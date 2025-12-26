# flops_analysis/flops_lab4_simple.py
# Script simple pour la partie 3 du TP HSP
# Le but ici est juste de calculer les FLOPs du modele de diffusion du Lab 4
# On utilise seulement fvcore, comme vu dans le cours

import os
import sys
import math
import torch
import torch.nn as nn

from fvcore.nn import FlopCountAnalysis


# ----------------------------------
# Import du modele du Lab 4
# ----------------------------------
# Le code du Lab 4 est dans le dossier lab4_image/
# On ajoute le dossier au PYTHONPATH pour pouvoir importer le modele

this_dir = os.path.dirname(__file__)
lab4_dir = os.path.abspath(os.path.join(this_dir, "..", "lab4_image"))
sys.path.insert(0, lab4_dir)

# Dans notre TP, le reseau s'appelle SimpleMLP (pas ScoreNet)
from model_gaussian import SimpleMLP


# ----------------------------------
# Petit wrapper autour du MLP
# ----------------------------------
# Le MLP du Lab 4 prend (x, t_emb) en entree
# Ici on veut un forward(x, t) plus simple, donc on calcule t_emb a la main

class WrappedLab4MLP(nn.Module):
    def __init__(self, x_dim=2, t_dim=128, hidden_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.mlp = SimpleMLP(
            input_dim=x_dim,
            time_embedding_dim=t_dim,
            hidden_dim=hidden_dim
        )

    def time_embedding(self, t):
        # Encodage temporel sin/cos classique (comme dans les modeles de diffusion)
        # t est de taille (B,), on le transforme en vecteur
        if not torch.is_floating_point(t):
            t = t.float()

        half = self.t_dim // 2
        freqs = torch.exp(
            torch.arange(half) * (-math.log(10000.0) / (half - 1))
        )
        freqs = freqs.to(t.device)

        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x, t):
        # forward du modele: on calcule t_emb puis on appelle le MLP
        t_emb = self.time_embedding(t)
        return self.mlp(x, t_emb)


# ----------------------------------
# Main
# ----------------------------------
def main():
    print("=== Calcul des FLOPs (Lab 4 diffusion) ===")

    # On travaille sur CPU, comme dans le TP
    model = WrappedLab4MLP(
        x_dim=2,
        t_dim=128,
        hidden_dim=128
    ).eval()

    # On cree des donnees factices pour tester le modele
    # batch de 128 points en dimension 2
    B = 128
    x = torch.randn(B, 2)
    t = torch.randint(0, 1000, (B,))

    # Calcul des FLOPs pour un forward pass
    flops = FlopCountAnalysis(model, (x, t))
    total_flops = float(flops.total())

    print(f"FLOPs pour un forward pass : {total_flops:.3e}")
    print("Attention: fvcore ne compte pas toutes les ops (sin, cos, exp)")
    print("mais l ordre de grandeur reste correct pour analyser le cout du modele")


if __name__ == "__main__":
    main()
