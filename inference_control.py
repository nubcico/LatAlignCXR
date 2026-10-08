from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from tqdm import tqdm

from models import SiT_models
from rela_ctrl_wrapper import SiTRelaCtrlWrapper
from transport import Sampler, create_transport


class XRayPipeline:
    def __init__(
        self,
        ckpt_path: str,
        vae_path: str = None,
        model_name: str = "SiT-XL/2",
        image_size: int = 256,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.latent_size = image_size // 8

        # VAE setup
        if vae_path and Path(vae_path).exists():
            self.vae = AutoencoderKL.from_single_file(vae_path)
        else:
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
        self.vae = self.vae.to(self.device, dtype=torch.float32).eval()

        # SiT + RelaCtrl setup
        self.base_model = SiT_models[model_name](input_size=self.latent_size, num_classes=2)
        self.model = SiTRelaCtrlWrapper(
            self.base_model, condition_channels=3, relevant_layers=[2, 4, 6, 8, 10, 12, 14]
        ).to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt.get("ema", ckpt.get("model_state_dict", ckpt)))
        self.model.eval()

        # Transport Sampler setup
        transport = create_transport("Linear", "velocity", "None", None, None)
        self.sampler = Sampler(transport)

        self.mask_transform = transforms.Compose([
            transforms.Resize((self.latent_size, self.latent_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def _forward_wrapper(self, x, t, **kwargs):
        out = self.model.forward_with_cfg(x, t, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        if out.ndim == 3:
            out = self.base_model.unpatchify(out)
        if out.ndim == 4 and out.size(1) == x.size(1) * 2:
            out, _ = out.chunk(2, dim=1)
        return out

    @torch.no_grad()
    def generate(
        self,
        inputs,
        labels=0,
        cfg_scale: float = 3.5,
        control_scale: float = 0.8,
        num_steps: int = 50,
        method: str = "dopri5",
        vae_scale: float = 0.18215,
    ) -> torch.Tensor:
        """Generates X-rays for a single input (PIL image/path) or a list/batch of inputs."""
        is_single = not isinstance(inputs, (list, tuple))
        masks = [inputs] if is_single else list(inputs)
        
        if isinstance(labels, int):
            labels = [labels] * len(masks)

        # Preprocess masks
        loaded_masks = []
        for m in masks:
            if isinstance(m, (str, Path)):
                m = Image.open(m).convert("RGB")
            loaded_masks.append(self.mask_transform(m))

        mask_tensor = torch.stack(loaded_masks).to(self.device)
        label_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        null_labels = torch.full_like(label_tensor, 2)

        # Batch setup
        b = len(masks)
        z = torch.randn(b, 4, self.latent_size, self.latent_size, device=self.device)

        zs_cfg = torch.cat([z, z], dim=0)
        maps_cfg = torch.cat([mask_tensor, mask_tensor], dim=0)
        ys_cfg = torch.cat([label_tensor, null_labels], dim=0)

        sample_fn = self.sampler.sample_ode(sampling_method=method, num_steps=num_steps, atol=1e-6, rtol=1e-3)
        sample_kwargs = dict(y=ys_cfg, condition_img=maps_cfg, cfg_scale=cfg_scale, control_scale=control_scale)

        # Sampling & decoding
        latents = sample_fn(zs_cfg, self._forward_wrapper, **sample_kwargs)[-1]
        latents, _ = latents.chunk(2, dim=0)
        
        gen_img = self.vae.decode(latents.float() / vae_scale).sample.float()

        return gen_img[0] if is_single else gen_img

    def run_directory(
        self,
        mask_dir: str,
        out_dir: str,
        label: int = 0,
        batch_size: int = 4,
        **gen_kwargs,
    ):
        mask_dir, out_dir = Path(mask_dir), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        mask_paths = list(mask_dir.glob("*.png"))

        # Process in batches
        for i in tqdm(range(0, len(mask_paths), batch_size), desc="Generating X-rays"):
            batch_paths = mask_paths[i : i + batch_size]
            results = self.generate(batch_paths, labels=label, **gen_kwargs)

            for path, img in zip(batch_paths, results):
                base_name = path.stem.replace("_anatomic_mask", "")
                save_image(img, out_dir / f"{base_name}_controlled_xray.png", normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    ckpt_path = r"control_checkpoints\relactrl.pt"
    vae_path = r"vae_checkpoints\vae_best.pt"

    pipeline = XRayPipeline(ckpt_path=ckpt_path, vae_path=vae_path)

    pipeline.run_directory(
        mask_dir=r"anatomic_masks",
        out_dir=r"controlled_xrays",
        label=0,
        batch_size=4
    )