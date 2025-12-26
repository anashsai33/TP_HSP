
import torch
import torch.nn as nn
import math
from tqdm import tqdm

class SimpleMLP(nn.Module):
    """
    simple MLP simple to predict noise.
    
    Architecture:
    - Input: [x, time_embedding] ∈ ℝ^(2 + time_embedding_dim)
    - Hidden layers: 3 × [Linear + ReLU]
    - Output: ε_θ(x,t) ∈ ℝ²
    
    Note: To implement
    """
    
    def __init__(self, input_dim=2, time_embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.time_embedding_dim = time_embedding_dim
        
        total_in_dim = input_dim + time_embedding_dim  # concat [x, t_emb]
        
        self.net = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, input_dim)  # prédire ε ∈ ℝ²
        )
    
    def forward(self, x, t_emb):
        # Concaténation des features spatiales et temporelles
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)



class GaussianDiffusion(nn.Module):    
    """
    Note: To implement
    """ 
    
    def __init__(self, timesteps=100, beta_min=0.1, beta_max=5.0,
                 time_embedding_dim=128, hidden_dim=128):
        super().__init__()
        
        self.timesteps = timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.time_embedding_dim = time_embedding_dim
        
        # Réseau qui prédit le bruit ε_θ(x,t)
        self.score_network = SimpleMLP(
            input_dim=2,
            time_embedding_dim=time_embedding_dim,
            hidden_dim=hidden_dim
        )
    
    def beta(self, t):
        """
        β(t) = β_min + t (β_max - β_min)
        t: tensor ∈ [0,1], shape [B]
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def get_marginal_params(self, t):
        """
        Calcule m_t et σ_t de l’équation (1) du poly:
        
        p(x_t | x_0) = N( x_t ; m_t x_0, (1 - exp(-1/2 t²(βmax-βmin) - t βmin)) I )
        
        où:
        m_t = exp( -1/4 t²(βmax-βmin) - 1/2 t βmin )
        σ_t² = 1 - exp( -1/2 t²(βmax-βmin) - t βmin )
        """
        # t shape [B]
        b = self.beta_max - self.beta_min
        
        # m_t
        mt = torch.exp(-0.25 * (t ** 2) * b - 0.5 * t * self.beta_min)
        
        # variance term inside (1 - exp(...))
        exp_term = torch.exp(-0.5 * (t ** 2) * b - t * self.beta_min)
        var = 1.0 - exp_term
        sigma_t = torch.sqrt(var + 1e-8)  # pour stabilité numérique
        
        return mt, sigma_t
    
    def forward(self, x0, noise):
        """
        Forward utilisé pendant l'entraînement.
        
        x0   : données réelles (batch, 2)
        noise: bruit gaussien ε ~ N(0, I) (batch, 2)
        
        Étapes:
        - échantillonner t ~ U(0,1)
        - calculer m_t, σ_t
        - construire x_t = m_t x0 + σ_t ε
        - calculer l'embedding temporel
        - prédire ε_θ(x_t, t)
        
        On renvoie ε_θ(x_t, t), qui est comparé à ε dans la loss MSE.
        """
        device = x0.device
        batch_size = x0.shape[0]
        
        # t ~ Uniform(0,1)
        t = torch.rand(batch_size, device=device)
        
        # m_t, σ_t
        mt, sigma_t = self.get_marginal_params(t)   # shape [B]
        mt = mt.view(-1, 1)
        sigma_t = sigma_t.view(-1, 1)
        
        # Construire x_t = m_t x0 + σ_t ε
        x_t = mt * x0 + sigma_t * noise
        
        # Embedding temporel
        t_emb = self.get_time_embedding(t)  # shape [B, time_embedding_dim]
        
        # Prédiction du bruit
        eps_pred = self.score_network(x_t, t_emb)
        
        return eps_pred
    
    @torch.no_grad()
    def sampling(self, n_samples, device="cuda"):
        """
        Génère des échantillons x_0 en intégrant l'équation (2) (discrétisation du 
        probability flow ODE / reverse process).
        
        On part de x_T ~ N(0, I) et on remonte jusqu'à t=0.
        """
        self.eval()
        
        x = torch.randn(n_samples, 2, device=device)  # x_T
        dt = 1.0 / self.timesteps
        
        for i in tqdm(range(self.timesteps, 0, -1), desc="Sampling"):
            # t dans [0,1]
            # on utilise (i / T) comme temps courant
            t = torch.full((n_samples,), i / self.timesteps, device=device)
            
            beta_t = self.beta(t)               # [B]
            mt, sigma_t = self.get_marginal_params(t)
            sigma_t = sigma_t.view(-1, 1)       # [B,1]
            
            # Embedding temporel
            t_emb = self.get_time_embedding(t)  # [B, D_t]
            
            # Prédiction du bruit
            eps_pred = self.score_network(x, t_emb)  # [B,2]
            
            # Approximation du score:
            # ∇_x log p_t(x) ≈ - ε_θ(x,t) / σ_t
            score = -eps_pred / (sigma_t + 1e-8)
            
            # Drift du reverse-time / probability flow ODE discretisé:
            # x_{i-1} = x_i - [ -1/2 β(t) x_i - β(t) ∇_x log p_t(x) ] dt + √β(t) √dt z
            beta_t_col = beta_t.view(-1, 1)
            
            drift = -0.5 * beta_t_col * x - beta_t_col * score
            
            # Bruit gaussien (z=0 au dernier step si tu veux du pur ODE)
            if i > 1:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            
            diffusion = torch.sqrt(beta_t_col) * math.sqrt(dt)
            
            x = x - drift * dt + diffusion * z
        
        return x
    
    def get_time_embedding(self, timesteps):
        
        half_dim = self.time_embedding_dim // 2
        
        # Calcul des fréquences: 1, 1/10000^(1/d), ..., 1/10000
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        
        # Multiplication timesteps × fréquences
        emb = timesteps[:, None] * emb[None, :]
        
        # Concaténation [sin, cos]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb    
