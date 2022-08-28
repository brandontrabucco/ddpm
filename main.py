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

    parser.add_argument("--logdir", type=str, default="diffusion")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    try:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group(backend="nccl")

    except KeyError:
        rank, world_size = 0, 1

    print(f'Initialized process {rank} / {world_size}')
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")

    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if rank == 0 or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), 
                                         (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(
        "images/", download=True, train=True, transform=transform)

    sampler = None

    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=4, 
        pin_memory=True, sampler=sampler, shuffle=world_size == 1)

    unwrapped_model = model = DiffusionModel().to(device)

    print(model)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.train()

    iteration = -1

    for epoch in range(args.epochs):
        
        epoch_loss = 0.0
        epoch_size = 0.0

        for it, (images, labels) in enumerate(data_loader):

            iteration += 1

            images = images.to(device)
            noise = torch.randn_like(images)

            timestep = torch.rand(images.shape[0], device=device)

            alphas = torch.cos(timestep * 3.1415927410125732 / 2) ** 2
            alphas = alphas.view(images.shape[0], 1, 1, 1)

            images = (torch.sqrt(alphas) * images + 
                      torch.sqrt(1 - alphas) * noise)

            pred = model(images, timestep)
            loss = ((noise - pred) ** 2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss = loss.detach().cpu().numpy().item()
        
            epoch_loss += loss * float(images.shape[0])
            epoch_size += float(images.shape[0])
                
            print(f"Epoch {epoch} Iteration {iteration} " + 
                  f"Training Loss {epoch_loss / epoch_size}")

        print(f"Epoch {epoch} Iteration {iteration} " + 
              f"Training Loss {epoch_loss / epoch_size}")

        if rank == 0:
            torch.save(dict(model=unwrapped_model, iteration=iteration), 
                    os.path.join(args.logdir, f"model-{iteration}.pt"))
