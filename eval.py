from ddpm.models import DiffusionModel

import os

import argparse
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, default="diffusion/model-312799.pt")
    parser.add_argument("--images", type=int, default=32)

    args = parser.parse_args()

    alphas = 1.0 - torch.linspace(1e-4, 0.02, 1000).cuda()
    alphas_tilde = alphas.cumprod(dim=0)

    model = torch.load(args.ckpt, map_location='cuda')["model"]

    model.eval()

    images = torch.randn(args.images, 3, 32, 32).cuda()

    for t in reversed(range(1000)):
        timestep = torch.tensor([t] * args.images).cuda()
        
        alpha_t = alphas[timestep].view(args.images, 1, 1, 1)
        alpha_tilde_t = alphas_tilde[timestep].view(args.images, 1, 1, 1)
        
        w_t = (1 - alpha_t) / torch.sqrt(1 - alpha_tilde_t)
        z_t = torch.sqrt(1 - alpha_t) * torch.randn(args.images, 3, 32, 32).cuda()
        
        if t == 0:
            z_t = 0.0
        
        with torch.no_grad():
            pred = model(images, timestep)
        
        images = (1 / torch.sqrt(alpha_t)) * (images - w_t * pred) + z_t

    import matplotlib.pyplot as plt

    for i in range(args.images):
        plt.clf()
        plt.imshow(images[i].clamp(-1, 1).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.savefig(f"image{i}.png")