import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from download import find_model
from models import SiT_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import create_transport, Sampler
import argparse
import sys
import os

def main(mode, args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = args.ckpt
    raw_state = find_model(ckpt_path)
    state_dict = None
    if isinstance(raw_state, dict):
        if "ema" in raw_state:
            state_dict = raw_state["ema"]
            print("Loaded EMA weights")
        elif "model" in raw_state:
            state_dict = raw_state["model"]
        elif "state_dict" in raw_state:
            state_dict = raw_state["state_dict"]
        else:
            state_dict = raw_state
    else:
        state_dict = raw_state

    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if "final_layer.linear.weight" in clean_state_dict:
        out_features = clean_state_dict["final_layer.linear.weight"].shape[0]
        learn_sigma = (out_features == 32)
        print(f"Auto-detected learn_sigma = {learn_sigma} (output channels: {out_features})")
    else:
        learn_sigma = False

    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    ).to(device)
    model.load_state_dict(clean_state_dict)
    model.eval()

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)

    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    if args.vae_ckpt:
        state = torch.load(args.vae_ckpt, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", "").replace("vae.", ""): v for k, v in state.items()}
        vae.load_state_dict(state, strict=False)
        print(f"  Loaded fine-tuned VAE from {args.vae_ckpt}")
        
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    out_base_dir = args.out_dir
    os.makedirs(out_base_dir, exist_ok=True)

    class_targets = {
        0: ("No Finding", args.num_no_finding),
        1: ("Pneumonia",   args.num_pneumonia),
    }
    batch_size = 25

    for class_id, (class_name, target_count) in class_targets.items():
        class_dir = os.path.join(out_base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        generation_plan = []
        for idx in range(target_count):
            filename = f"LatAlignSiTXray_{class_name.replace(' ', '_')}_{idx+1:04d}.png"
            generation_plan.append(os.path.join(class_dir, filename))

        pending_plan = [(idx, path) for idx, path in enumerate(generation_plan) if not os.path.exists(path)]
        already_done = target_count - len(pending_plan)

        if already_done >= target_count:
            print(f"\n[{class_name.upper()}] already complete ({already_done}/{target_count}), skipping.")
            continue

        print(f"\n[{class_name.upper()}] Resuming from {already_done}/{target_count} (Target: {target_count})")

        for i in range(0, len(pending_plan), batch_size):
            current_batch = pending_plan[i:i + batch_size]
            current_batch_size = len(current_batch)

            torch.manual_seed(args.seed + i + class_id * 100000)

            z = torch.randn(current_batch_size, 4, latent_size, latent_size, device=device)
            y = torch.tensor([class_id] * current_batch_size, device=device)

            z_cfg = torch.cat([z, z], 0)
            y_null = torch.tensor([args.num_classes] * current_batch_size, device=device)
            y_cfg = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y_cfg, cfg_scale=args.cfg_scale)

            samples = sample_fn(z_cfg, model.forward_with_cfg, **model_kwargs)[-1]
            samples, _ = samples.chunk(2, dim=0)
            samples = vae.decode(samples / 0.18215).sample

            for batch_idx, (global_idx, out_path) in enumerate(current_batch):
                save_image(samples[batch_idx], out_path, normalize=True, value_range=(-1, 1))

            print(f"   -> [{class_name.upper()}] [{already_done + i + current_batch_size}/{target_count}] saved to {class_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: inference.py <mode> [options]")
        sys.exit(1)

    mode = sys.argv[1]
    assert mode[:2] != "--", "Usage: inference.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Choose 'ODE' or 'SDE'"

    parser.add_argument("--model",               type=str,   choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae_ckpt",            type=str,   default=r"vae_checkpoints\vae_best.pt")
    parser.add_argument("--image-size",          type=int,   choices=[256, 512], default=256)
    parser.add_argument("--num-classes",         type=int,   default=2)   
    parser.add_argument("--cfg-scale",           type=float, default=4.0) 
    parser.add_argument("--num-sampling-steps",  type=int,   default=100)      
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--ckpt",                type=str,   default=r"checkpoints\model.pt")
    parser.add_argument("--out-dir",             type=str,   default=r"xrays")
    parser.add_argument("--num-no-finding",      type=int,   default=1000)
    parser.add_argument("--num-pneumonia",       type=int,   default=1000)

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
    elif mode == "SDE":
        parse_sde_args(parser)

    args = parser.parse_known_args()[0]
    main(mode, args)