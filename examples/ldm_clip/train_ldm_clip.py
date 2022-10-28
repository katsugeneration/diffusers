import argparse
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
import numpy as np
from torchvision import transforms
import datasets
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPFeatureExtractor
from clip_loss import CLIPLoss


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        default=None,
        required=True,
        help="Train use target dataset name.",
    )
    parser.add_argument(
        "--style_img_dir",
        type=str,
        default=None,
        required=True,
        help="Path to style images directory.",
    )
    parser.add_argument(
        "--target_text",
        type=str,
        default=None,
        help="The target text describing the output image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--emb_train_steps",
        type=int,
        default=500,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--emb_learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizing the embeddings.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate for fine tuning the model.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--train_inference_steps", type=int, default=50, help="DDIM steps")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_auth_token=True)

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", use_auth_token=True)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", use_auth_token=True)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", use_auth_token=True)
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
    clip_loss = CLIPLoss(clip, feature_extractor)

    vae.eval()
    unet.eval()
    text_encoder.eval()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.Adam8bit
    else:
        optimizer_class = torch.optim.Adam

    noise_scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    noise_scheduler.set_timesteps(args.train_inference_steps)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    clip.to(accelerator.device, dtype=weight_dtype)

    # Encode the input image.
    if args.target_dataset == "Imagenet":
        dataset = datasets.load_dataset("mrm8488/ImageNet1K-val", split="train")

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def encode_image(image):
        image_tensor = image_transforms(image)
        image_tensor = image_tensor[None].to(device=accelerator.device, dtype=weight_dtype)
        with torch.inference_mode():
            image_latents = vae.encode(image_tensor).latent_dist.sample()
            image_latents = 0.18215 * image_latents
        return image_latents

    def decode_image(latents):
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def denoise(latents, t_start, cond_embeddings=None):
        timesteps = noise_scheduler.timesteps[t_start:].to(accelerator.device)

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = noise_scheduler.scale_model_input(latents, t)

            # predict the noise residual
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=cond_embeddings).sample

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        return latents

    def denoise_one_step(latents, t_start, cond_embeddings=None):
        t = noise_scheduler.timesteps[t_start].to(accelerator.device)

        # expand the latents if we are doing classifier free guidance
        latent_model_input = noise_scheduler.scale_model_input(latents, t)

        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=cond_embeddings).sample

        # compute the previous noisy sample x_t -> x_t-1
        pred = noise_scheduler.step(noise_pred, t, latents).pred_original_sample
        return pred

    def save_logs(img, path, step, key="sample"):
        img = Image.fromarray((255 * img.permute(1, 2, 0).numpy()).astype(np.uint8))
        imgpath = os.path.join(path, f"{key}_step_{step:06}.png")
        img.save(imgpath)

    latents = torch.cat([encode_image(data) for data in dataset[:100]['image']])
    with torch.inference_mode():
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(latents, decode_image(latents).cpu()), batch_size=args.train_batch_size, shuffle=True)

    # Encode the target text.
    text_ids = tokenizer(
        args.target_text,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids

    text_ids = text_ids.to(device=accelerator.device)
    with torch.inference_mode():
        target_embeddings = text_encoder(text_ids)[0]

    # Compute CLIP Loss
    valid_exts = [".png", ".jpg", ".jpeg"]
    file_list = [
        os.path.join(args.style_img_dir, file_name)
        for file_name in os.listdir(args.style_img_dir)
        if os.path.splitext(file_name)[1].lower() in valid_exts
    ]
    with torch.inference_mode():
        decode_images = decode_image(latents[:100]).cpu()
        clip_loss.set_img2img_direction(list(decode_images), file_list)
        [save_logs(im, logging_dir, i, 'origin') for i, im in enumerate(decode_images[:10])]

    del text_encoder, dataset, decode_images
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_embeddings = target_embeddings.float()
    optimized_embeddings = target_embeddings.clone()

    # Optimize the text embeddings first.
    optimized_embeddings.requires_grad_(True)
    optimizer = optimizer_class(
        [optimized_embeddings],  # only optimize embeddings
        lr=args.emb_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        # weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    unet, vae, optimizer, clip_loss, loader = accelerator.prepare(unet, vae, optimizer, clip_loss, loader)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("ldm_clip", config=vars(args))

    def train_loop(pbar, optimizer):
        loss_avg = AverageMeter()
        for step in pbar:
            init_latents, origin_image = next(iter(loader))
            with accelerator.accumulate(optimized_embeddings):
                noise = torch.randn_like(init_latents)
                bsz = init_latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, args.train_inference_steps, (1,), device=accelerator.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                with torch.no_grad():
                    noisy_latents = noise_scheduler.add_noise(init_latents, noise, torch.tensor([noise_scheduler.timesteps[-timesteps]] * bsz, device=accelerator.device))

                sample_conditioned = denoise_one_step(noisy_latents, args.train_inference_steps - timesteps - 1, cond_embeddings=torch.cat([optimized_embeddings]*bsz))
                decode_conditioned = decode_image(sample_conditioned.to(dtype=weight_dtype))

                loss = clip_loss(list(origin_image.to(device=accelerator.device)), list(decode_conditioned))

                accelerator.backward(loss)
                # if accelerator.sync_gradients:     # results aren't good with it, may be will need more training with it.
                #     accelerator.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                loss_avg.update(loss.detach_(), bsz)

            if not step % args.log_interval:
                logs = {"loss": loss_avg.avg.item()}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=step)
                with torch.no_grad():
                    samples = torch.cat([decode_conditioned, origin_image.to(device=accelerator.device)], dim=-1).to("cpu")
                    samples = torch.cat(list(samples), dim=-2)
                    save_logs(samples, logging_dir, step)

        accelerator.wait_for_everyone()

    progress_bar = tqdm(range(args.emb_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Optimizing embedding")

    train_loop(progress_bar, optimizer)

    optimized_embeddings.requires_grad_(False)
    if accelerator.is_main_process:
        torch.save(target_embeddings.cpu(), os.path.join(args.output_dir, "target_embeddings.pt"))
        torch.save(optimized_embeddings.cpu(), os.path.join(args.output_dir, "optimized_embeddings.pt"))
        with open(os.path.join(args.output_dir, "target_text.txt"), "w") as f:
            f.write(args.target_text)

    # # Fine tune the diffusion model.
    # optimizer = optimizer_class(
    #     accelerator.unwrap_model(unet).parameters(),
    #     lr=args.learning_rate,
    #     betas=(args.adam_beta1, args.adam_beta2),
    #     # weight_decay=args.adam_weight_decay,
    #     eps=args.adam_epsilon,
    # )
    # optimizer = accelerator.prepare(optimizer)

    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    # progress_bar.set_description("Fine Tuning")
    # unet.train()

    # train_loop(progress_bar, optimizer, unet.parameters())

    # Create the pipeline using using the trained modules and save it.
    # if accelerator.is_main_process:
    #     pipeline = StableDiffusionPipeline.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         unet=accelerator.unwrap_model(unet),
    #         use_auth_token=True
    #     )
    #     pipeline.save_pretrained(args.output_dir)

    #     if args.push_to_hub:
    #         repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()