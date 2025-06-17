import os
import argparse
import torch
import warnings

from diffusers import DDIMScheduler
from utils.scheduler import DDIMInverseScheduler
from utils.edit_pipeline import EditingPipeline

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_prompt', type=str, required=True)
    parser.add_argument('--target_prompt', type=str, required=True)
    parser.add_argument('--random_seed', default=0)
    parser.add_argument('--results_folder', type=str, default='output/test_cat')
    parser.add_argument('--num_ddim_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--negative_guidance_scale', default=5.0, type=float)
    parser.add_argument('--use_float_16', action='store_true')
    parser.add_argument('--cycle_guidance', type=float, default=0.0)
    parser.add_argument('--clip_guidance', type=float, default=0.1)
    parser.add_argument('--structure_guidance', type=float, default=1.0)
    parser.add_argument('--beta_p', type=float, default=0.5)
    parser.add_argument('--beta_f', type=float, default=2.5)

    args = parser.parse_args()

    os.makedirs(args.results_folder, exist_ok=True)

    if args.use_float_16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Make the editing pipeline
    pipe = EditingPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.cycle_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inv_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.cycle_inv_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    # make the input noise map
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    else:
        torch.manual_seed(args.random_seed)

    x = torch.randn((1,4,64,64), device=device)
    
    rec_pil, edit_pil = pipe(
        prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        num_inference_steps=args.num_ddim_steps,
        x_in=x,
        guidance_scale=args.negative_guidance_scale,
        negative_prompt="", # use the empty string for the negative prompt,
        cycle_guidance=args.cycle_guidance,
        clip_guidance=args.clip_guidance,
        structure_guidance=args.structure_guidance,
        clip_margin=args.beta_p,
        structure_margin=args.beta_f,
    )

    edit_pil[0].save(os.path.join(args.results_folder, f"edit.png"))
    rec_pil[0].save(os.path.join(args.results_folder, f"reconstruction.png"))

