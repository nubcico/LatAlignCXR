import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from anatomy_encoder import AnatomyEncoder
from models import SiT_models


class SiTAnatomyWrapper(nn.Module):
    def __init__(self, sit_model, anat_encoder, hidden_dim=1152):
        super().__init__()
        self.sit = sit_model
        self.anat_encoder = anat_encoder
        self.num_tokens = 256
        self.anat_proj = nn.Linear(256, hidden_dim)
        self.anat_pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim) * 0.02)
        self.uncond_anat = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim) * 0.02)

        for p in self.sit.parameters():
            p.requires_grad = True
        self.anat_encoder.eval()
        for p in self.anat_encoder.parameters():
            p.requires_grad = False

    def _get_anatomy_tokens(self, clean_img, drop_anat=False):
        b = clean_img.shape[0]
        if drop_anat:
            return self.uncond_anat.expand(b, -1, -1)
        if clean_img.shape[-2:] != (256, 256):
            clean_img = F.interpolate(clean_img, size=(256, 256), mode="bilinear", align_corners=False)
        with torch.no_grad():
            z_spatial, _ = self.anat_encoder(clean_img)
            z_spatial = F.adaptive_avg_pool2d(z_spatial, (16, 16))
        x = z_spatial.flatten(2).transpose(1, 2)
        return self.anat_proj(x) + self.anat_pos_embed

    def forward(self, x_noisy, t, severity, clean_img, drop_anat=False, drop_class=False):
        b = x_noisy.shape[0]
        anat_tokens = self._get_anatomy_tokens(clean_img, drop_anat=drop_anat)
        img_tokens = self.sit.x_embedder(x_noisy) + self.sit.pos_embed
        x = torch.cat([anat_tokens, img_tokens], dim=1)

        t_embed = self.sit.t_embedder(t)
        if drop_class:
            y_embed = self.sit.y_embedder(torch.full((b,), 2, dtype=torch.long, device=x_noisy.device), False)
        else:
            emb_healthy = self.sit.y_embedder(torch.zeros(b, dtype=torch.long, device=x_noisy.device), False)
            emb_sick = self.sit.y_embedder(torch.ones(b, dtype=torch.long, device=x_noisy.device), False)
            sev = severity.view(b, 1)
            y_embed = (1.0 - sev) * emb_healthy + sev * emb_sick

        c = t_embed + y_embed
        for block in self.sit.blocks:
            x = block(x, c)
        img_out = self.sit.final_layer(x[:, self.num_tokens:], c)
        return self.sit.unpatchify(img_out)[:, :4]


class InpaintingPipeline:
    def __init__(
        self,
        anat_ckpt: str,
        sit_ckpt: str = None,
        model_name: str = "SiT-XL/2",
        image_size: int = 256,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.latent_size = image_size // 8

        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device).eval()

        anat_enc = AnatomyEncoder().to(self.device).eval()
        anat_state = torch.load(anat_ckpt, map_location=self.device, weights_only=False)
        anat_enc.load_state_dict(anat_state.get("anatomy_enc", anat_state), strict=False)

        sit_base = SiT_models[model_name](num_classes=2).to(self.device)
        self.model = SiTAnatomyWrapper(sit_base, anat_enc).to(self.device).eval()

        if sit_ckpt and Path(sit_ckpt).exists():
            ckpt = torch.load(sit_ckpt, map_location=self.device, weights_only=False)
            sit_state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict({k.replace("module.", ""): v for k, v in sit_state.items()}, strict=False)

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def mask_to_latent(self, mask_pil, dilate_px=15, blur_px=25):
        m = transforms.functional.to_tensor(mask_pil.convert("L"))
        m = F.interpolate(m.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        m = (m > 0.5).float()

        if dilate_px > 0:
            k = 2 * dilate_px + 1
            m = F.max_pool2d(m, kernel_size=k, stride=1, padding=k // 2)
        if blur_px > 0:
            k = 2 * blur_px + 1
            m = transforms.functional.gaussian_blur(m, kernel_size=[k, k], sigma=[max(0.1, blur_px / 2.0)] * 2)

        return F.avg_pool2d(m, kernel_size=self.image_size // self.latent_size).to(self.device)

    @torch.no_grad()
    def generate(
        self,
        img_path,
        mask_path,
        severity: float = 1.5,
        cfg_scale: float = 5.0,
        steps: int = 50,
        edit_strength: float = 1.0,
        dilate_px: int = 15,
        blur_px: int = 25,
        seed: int = None,
    ) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)

        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")

        clean_img = self.transform(img_pil).unsqueeze(0).to(self.device)
        mask_latent = self.mask_to_latent(mask_pil, dilate_px, blur_px)
        sev_tensor = torch.tensor([severity], dtype=torch.float32, device=self.device)

        z_source = self.vae.encode(clean_img * 2.0 - 1.0).latent_dist.sample().mul_(0.18215)
        
        init_noise = torch.randn_like(z_source)
        dt = 1.0 / steps
        t_start = 1.0 - edit_strength
        sample_x = t_start * z_source + (1.0 - t_start) * init_noise
        start_step = int(steps * t_start)

        for i in range(start_step, steps):
            t_val = i / steps
            vec_t = torch.full((1,), t_val, device=self.device)
            noised_source = t_val * z_source + (1.0 - t_val) * init_noise
            sample_x = mask_latent * sample_x + (1.0 - mask_latent) * noised_source

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                p_cond = self.model(sample_x, vec_t, sev_tensor, clean_img, drop_anat=False, drop_class=False)
                p_uncond = self.model(sample_x, vec_t, sev_tensor, clean_img, drop_anat=True, drop_class=True)
                v_pred = p_uncond + cfg_scale * (p_cond - p_uncond)
                sample_x = sample_x + v_pred * dt

        sample_x = mask_latent * sample_x + (1.0 - mask_latent) * z_source
        decoded = torch.clamp((self.vae.decode(sample_x / 0.18215).sample + 1.0) / 2.0, 0.0, 1.0)
        return decoded[0]

    def run_directory(
        self,
        xrays_dir: str,
        masks_dir: str,
        out_dir: str,
        overwrite: bool = True,
        **gen_kwargs,
    ):
        xrays_dir, masks_dir, out_dir = Path(xrays_dir), Path(masks_dir), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mask_map = {m.stem: m for m in masks_dir.rglob("*") if m.suffix in (".png", ".jpg")}

        tasks = []
        for x_path in list(xrays_dir.rglob("*.png")) + list(xrays_dir.rglob("*.jpg")):
            m_path = mask_map.get(x_path.stem)
            if m_path:
                out_path = out_dir / f"{x_path.stem}_inpainted.png"
                if overwrite or not out_path.exists():
                    tasks.append((x_path, m_path, out_path))

        for x_path, m_path, out_path in tqdm(tasks, desc="Inpainting X-rays"):
            try:
                res = self.generate(x_path, m_path, **gen_kwargs)
                save_image(res, out_path)
            except Exception as e:
                print(f"Error on {x_path.name}: {e}")


if __name__ == "__main__":
    anat_ckpt = r"anatomy_checkpoints\best_anatomy_encoder.pt"
    sit_ckpt = r"checkpoints\best_checkpoint.pt"

    pipeline = InpaintingPipeline(anat_ckpt=anat_ckpt, sit_ckpt=sit_ckpt)

    pipeline.run_directory(
        xrays_dir=r"xrays",
        masks_dir=r"anatomic_masks",
        out_dir=r"inpainted_xrays",
    )