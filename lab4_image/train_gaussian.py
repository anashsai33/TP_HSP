import torch
import torch.nn as nn
from torch.optim import AdamW
# NOUVEAU: Importer le scheduler et EMA
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import ExponentialMovingAverage 

from model_gaussian import GaussianDiffusion
from gaussian_dataset import create_gaussian_dataloader, visualize_comparison
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Simple 2D Gaussian VP-SDE")
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--batch_size', type=int, default=512) # Augmenté
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=2000)
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=5.0)
    parser.add_argument('--gaussian_samples', type=int, default=50000) # Augmenté
    parser.add_argument('--model_ema_decay', type=float, default=0.995) # NOUVEAU
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


def main(args):
    device = "cpu" if args.cpu else "cuda"
    
    # Dataloader
    train_dataloader = create_gaussian_dataloader(
        batch_size=args.batch_size,
        n_samples=args.gaussian_samples
    )
    
    true_samples = next(iter(train_dataloader))[:args.n_samples]
    
    # Model
    model = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_min=args.beta_min,
        beta_max=args.beta_max
    ).to(device)
    
    # NOUVEAU: Initialiser EMA
    model_ema = ExponentialMovingAverage(model, device=device, decay=args.model_ema_decay)
    
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    # NOUVEAU: Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_dataloader),
        eta_min=1e-6
    )
    
    os.makedirs("results_gaussian", exist_ok=True)
    global_steps = 0
    
    # Training
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for data in train_dataloader:
            noise = torch.randn_like(data).to(device)
            data = data.to(device)
            
            pred = model(data, noise)
            loss = loss_fn(pred, noise)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            # NOUVEAU: Mettre à jour EMA
            model_ema.update_parameters(model)
            
            epoch_loss += loss.item()
            global_steps += 1
        
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.5f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Sample et visualise en utilisant le modèle EMA (plus stable)
        model_ema.eval()
        samples = model_ema.module.sampling(args.n_samples, device=device)
        
        visualize_comparison(
            true_samples,
            samples,
            save_path=f"results_gaussian/epoch{epoch+1:03d}.png"
        )
        
        # NOUVEAU: Sauvegarder le checkpoint avec EMA
        ckpt = {
            "model": model.state_dict(),
            "ema_model": model_ema.state_dict()
        }
        torch.save(ckpt, f"results_gaussian/epoch{epoch+1:03d}.pt")


if __name__ == "__main__":
    args = parse_args()
    main(args)