"""
VP-SDE Forward Process:
    dx = -1/2 * β(t) * x * dt + √β(t) * dw
    
où:
    - β(t) = β_min + t(β_max - β_min) : Linear Schedule 
    - x(t) : data at time t
    - dw : Brownian motion

Metrics:
    1. FID (Fréchet Inception Distance) 
    2. Log-Likelihood
"""

import torch
import numpy as np
from scipy.linalg import sqrtm
import argparse
import os
from model_gaussian import GaussianDiffusion

from gaussian_dataset import GaussianMixture2D, DEFAULT_CENTERS, DEFAULT_COVARIANCES


try:
    from torchdiffeq import odeint
    ODEINT_AVAILABLE = True
except ImportError:
    ODEINT_AVAILABLE = False
    print("⚠️  Warning: torchdiffeq not installed. Install with: pip install torchdiffeq")


def calculate_fid_2d(real_samples, generated_samples):
    """
    Compute FID (Fréchet Inception Distance)
    
    Le FID gives the distance between two Gaussian distributions:
    
    FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))        
    
    smaller FID means better sample quality
    
    Args:
        real_samples: true samples [N, 2]
        generated_samples: generated samples [N, 2]
    
    Returns:
        fid: score FID (float)
    """
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.cpu().numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.cpu().numpy()
    
    # Calculer les statistiques des deux distributions
    mu1 = np.mean(real_samples, axis=0)
    mu2 = np.mean(generated_samples, axis=0)
    
    sigma1 = np.cov(real_samples, rowvar=False)
    sigma2 = np.cov(generated_samples, rowvar=False)
    
    # Régularisation pour la stabilité numérique
    epsilon = 1e-6
    sigma1 += np.eye(2) * epsilon
    sigma2 += np.eye(2) * epsilon
    
    # Calculer le FID selon la formule
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2*covmean)
    return float(fid)


def calculate_exact_likelihood_ode(model, x, device='cuda', rtol=1e-5, atol=1e-5):
    
    """
    
    1. Probability Flow ODE :       
       
       dx = [f(x,t) - 1/2 * g(t)² * ∇ₓlog pₜ(x)] dt
             
       - f(x,t) = -1/2 * β(t) * x           [drift]
       - g(t)² = β(t)                        [squared diffusion]
       - ∇ₓlog pₜ(x) = score function       [gradient of log-likelihood]
       
       Donc: dx = [-1/2*β(t)*x - 1/2*β(t)*∇ₓlog pₜ(x)] dt
    
    2. Change of Variables (Instantaneous Change of Variables):
       
       log p₀(x(0)) = log pₜ(x(T)) - ∫₀ᵀ ∇·f(x(t),t) dt
    
    3. Score Estimation :
       
       ∇ₓlog pₜ(x) = -ε_θ(x,t) / σ(t)
       
       
       - ε_θ : noise predicted by the network
       - σ(t) = √(1 - α(t)²) with α(t) = exp(-1/2 ∫₀ᵗ β(s)ds)       
    
    Returns:
        bits_per_dim: log-likelihood

    Note: To implement
    """    


    import torch
    import math

    model.eval()
    x = x.to(device)
    N, D = x.shape

    # Pas de l’ODE
    T = 1.0
    steps = model.timesteps
    dt = T / steps

    # Grille temporelle 0 → 1
    t_grid = torch.linspace(0, 1, steps, device=device)

    # On intègre : ∫ Tr(J_f) dt
    div_integral = torch.zeros(N, device=device)

    # x(t=0)
    x_t = x.clone().detach()
    x_t.requires_grad_(True)

    for t in t_grid:
        t_batch = torch.full((N,), t, device=device)

        # β(t)
        beta_t = model.beta(t_batch).view(-1, 1)  # [N,1]

        # σ(t)
        _, sigma_t = model.get_marginal_params(t_batch)
        sigma_t = sigma_t.view(-1, 1)

        # Embedding temporel
        t_emb = model.get_time_embedding(t_batch)

        # Bruit prédit par le réseau
        eps_pred = model.score_network(x_t, t_emb)

        # Score = ∇ log p_t(x)
        score = -eps_pred / (sigma_t + 1e-12)

        # Probability Flow drift :
        # dx/dt = -1/2 β(t) x - 1/2 β(t) score
        drift = -0.5 * beta_t * x_t - 0.5 * beta_t * score

        # Hutchinson estimator pour la divergence
        eps = torch.randn_like(x_t)
        score_eps = (score * eps).sum()
        grad = torch.autograd.grad(score_eps, x_t, create_graph=False)[0]
        div_score = (grad * eps).sum(dim=1)  # [N]

        # Tr(J_f)
        beta_flat = beta_t.view(-1)
        trace_f = -0.5 * beta_flat * D - 0.5 * beta_flat * div_score

        # Intégration ∫ Tr(J_f) dt
        div_integral += trace_f * dt

        # Intégration avant d’un pas d’Euler
        with torch.no_grad():
            x_t = x_t + drift * dt
        x_t.requires_grad_(True)

    # À t=1 → distribution ~ N(0, I)
    log_p1 = -0.5 * (x_t.pow(2).sum(dim=1) + D * math.log(2 * math.pi))

    # Log-likelihood à t=0
    log_p0 = log_p1 - div_integral  # [N]

    # Bits / dimension
    bits_per_dim = -log_p0.mean() / (math.log(2) * D)

    return bits_per_dim.item()



def evaluate_model(checkpoint_path, n_samples=10000, device="cuda"):
    
    print(f"\nEvaluating: {checkpoint_path}")
    print(f"Generating {n_samples} samples...\n")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Hyperparamètres VP-SDE (doivent correspondre à l'entraînement)
    timesteps = 100
    beta_min = 0.1  # β_min dans l'équation β(t) = β_min + t(β_max - β_min)
    beta_max = 5.0  # β_max
    
    model = GaussianDiffusion(
        timesteps=timesteps,
        beta_min=beta_min,
        beta_max=beta_max
    ).to(device)
    
    # Charger les poids du modèle
    if 'ema_model' in checkpoint:
        print("Loading EMA model state...")
        model_ema = ExponentialMovingAverage(model, device=device, decay=0.995) 
        model_ema.load_state_dict(checkpoint['ema_model'])
        model = model_ema.module
    elif 'model' in checkpoint:
        print("Loading standard model state...")
        model.load_state_dict(checkpoint['model'])
    else:
        print("Loading raw state dict...")
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # ====================================================================
    # ÉTAPE 1: Génération d'échantillons via Reverse-Time SDE
    # ====================================================================
    with torch.no_grad():
        generated_samples = model.sampling(n_samples, device=device)
    
    # ====================================================================
    # ÉTAPE 2: Chargement des données réelles (distribution cible)
    # ====================================================================
    ## PATCH : dataset réel dérivé automatiquement de gaussian_dataset.py
    dataset = GaussianMixture2D(DEFAULT_CENTERS, DEFAULT_COVARIANCES, N=n_samples)

    real_samples = dataset.data
    
    # ====================================================================
    # ÉTAPE 3: Calcul du FID (Qualité visuelle)
    # ====================================================================
    fid_score = calculate_fid_2d(real_samples, generated_samples)
    
    # ====================================================================
    # ÉTAPE 4: Calcul de la Log-Likelihood exacte (Probability Flow ODE)
    # ====================================================================
    bits_per_dim = calculate_exact_likelihood_ode(model, real_samples, device=device)
    
    print(f"{'='*50}")
    print(f"FID Score:          {fid_score:.6f}")
    print(f"Log-Likelihood:     {bits_per_dim:.6f} bits/dim")
    print(f"{'='*50}\n")
    
    return fid_score, bits_per_dim


def evaluate_directory(results_dir="results_gaussian", n_samples=10000, device="cuda"):
    from utils import ExponentialMovingAverage 
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} not found!")
        return
    
    checkpoints = sorted([f for f in os.listdir(results_dir) if f.endswith('.pt')])
    
    if not checkpoints:
        print(f"No checkpoint files found in {results_dir}")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoints\n")
    
    results = []
    
    for ckpt_file in checkpoints:
        ckpt_path = os.path.join(results_dir, ckpt_file)
        
        try:
            fid, log_lik = evaluate_model(ckpt_path, n_samples, device)
            results.append({
                'checkpoint': ckpt_file,
                'fid': fid,
                'log_likelihood_bits_dim': log_lik
            })
        except Exception as e:
            print(f"Error evaluating {ckpt_file}: {e}\n")
            continue
    
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        best_fid = min(results, key=lambda x: x['fid'])
        best_ll = max(results, key=lambda x: x['log_likelihood_bits_dim'])
        
        print(f"\nBest FID: {best_fid['checkpoint']}")
        print(f"  FID: {best_fid['fid']:.6f}")
        print(f"  Log-Likelihood: {best_fid['log_likelihood_bits_dim']:.6f} bits/dim")
        
        print(f"\nBest Log-Likelihood: {best_ll['checkpoint']}")
        print(f"  FID: {best_ll['fid']:.6f}")
        print(f"  Log-Likelihood: {best_ll['log_likelihood_bits_dim']:.6f} bits/dim")
        
        print(f"\n{'='*70}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Gaussian Mixture - FID & Log-Likelihood")
    parser.add_argument('--checkpoint', type=str, help='Path to specific checkpoint')
    parser.add_argument('--results_dir', type=str, default='results_gaussian', 
                        help='Directory with checkpoints')
    parser.add_argument('--n_samples', type=int, default=10000, 
                        help='Number of samples for evaluation')
    parser.add_argument('--cpu', action='store_true', help='Use CPU')
    parser.add_argument('--all', action='store_true', help='Evaluate all checkpoints')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cpu" if args.cpu else "cuda"
    
    from utils import ExponentialMovingAverage 
    
    if args.all:
        evaluate_directory(args.results_dir, args.n_samples, device)
    elif args.checkpoint:
        evaluate_model(args.checkpoint, args.n_samples, device)
    else:
        if os.path.exists(args.results_dir):
            checkpoints = sorted([f for f in os.listdir(args.results_dir) if f.endswith('.pt')])
            if checkpoints:
                latest = os.path.join(args.results_dir, checkpoints[-1])
                print(f"Latest checkpoint: {latest}")
                evaluate_model(latest, args.n_samples, device)
            else:
                print(f"No checkpoints in {args.results_dir}")
        else:
            print(f"Directory {args.results_dir} not found!")
