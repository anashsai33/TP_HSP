import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



DEFAULT_CENTERS = [
    # Cercle de 8 clusters
    (5,0), (3.5,3.5), (0,5), (-3.5,3.5),
    (-5,0), (-3.5,-3.5), (0,-5), (3.5,-3.5),

    # Mode très rare, isolé
    (15.0, 15.0)
]
DEFAULT_COVARIANCES = [0.3] * len(DEFAULT_CENTERS)


class GaussianMixture2D(Dataset):
    """
    Dataset :
      - centers : liste des μ, shape K x 2
      - covariances : liste de variances scalaires OU matrices 2x2
      - N : nombre d'échantillons
    """

    def __init__(self, centers, covariances, N=10000):
        super().__init__()

        self.centers = np.array(centers)  # shape (K, 2)
        self.K = len(centers)
        self.N = N

        # Convert covariances
        self.covariances = []
        for cov in covariances:
            if np.isscalar(cov):
                # variance scalaire -> matrice 2x2
                self.covariances.append(cov * np.eye(2))
            else:
                self.covariances.append(np.array(cov))

        # Generate samples
        self.data = self._generate()

    def _generate(self):
        samples = []
        for _ in range(self.N):
            k = np.random.randint(0, self.K)
            mu = self.centers[k]
            cov = self.covariances[k]
            x = np.random.multivariate_normal(mu, cov)
            samples.append(x)
        return torch.tensor(samples, dtype=torch.float32)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]


def create_gaussian_dataloader(
    batch_size: int = 128,
    n_samples: int = 10000,
    centers=None,
    covariances=None,
):
    """
    modification sans definir de distance entre cluster 
    """

    centers = DEFAULT_CENTERS   ## PATCH




    if covariances is None:
        # même variance pour chaque mode
        covariances = [0.5] * len(centers)

    dataset = GaussianMixture2D(centers, covariances, N=n_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # on retourne seulement le DataLoader
    # car train nious le demande 
    return loader


def visualize_comparison(real_samples, generated_samples, save_path):
    """
    Visualise les samples réels vs générés fenêtres auto
    """

    # CPU + numpy
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.detach().cpu().numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Fenêtres auto ---
    all_data = np.concatenate([real_samples, generated_samples], axis=0)
    xmin, ymin = all_data.min(axis=0)
    xmax, ymax = all_data.max(axis=0)

    # petite marge
    margin = 0.2 * max(xmax - xmin, ymax - ymin)

    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin

    # Real data
    axes[0].scatter(real_samples[:, 0], real_samples[:, 1], c='blue', s=5)
    axes[0].set_title("Real samples")
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(ymin, ymax)

    # Generated data
    axes[1].scatter(generated_samples[:, 0], generated_samples[:, 1], c='red', s=5)
    axes[1].set_title("Generated samples")
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
