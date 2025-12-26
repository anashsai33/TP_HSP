import torch


def vectorize_lower(M: torch.Tensor) -> torch.Tensor:
    # version rapide sans boucle python
    # torch s occupe de tout
    n = M.shape[-1]
    idx = torch.tril_indices(n, n, device=M.device)
    return M[..., idx[0], idx[1]]


def devectorize_lower(v: torch.Tensor, n: int) -> torch.Tensor:
    # on remet le vecteur dans une matrice
    idx = torch.tril_indices(n, n, device=v.device)
    out = torch.zeros(*v.shape[:-1], n, n, device=v.device, dtype=v.dtype)

    out[..., idx[0], idx[1]] = v
    # on force la symetrie
    out = out + out.transpose(-1, -2) - torch.diag_embed(
        torch.diagonal(out, dim1=-2, dim2=-1)
    )
    return out


def spd_matrix_sqrt(M: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    # version batchee donc beaucoup plus rapide
    d, U = torch.linalg.eigh(M)
    d = torch.clamp(d, min=eps)
    return U @ torch.diag_embed(torch.sqrt(d)) @ U.transpose(-1, -2)


def spd_matrix_log(M: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    # log matriciel en batch
    d, U = torch.linalg.eigh(M)
    d = torch.clamp(d, min=eps)
    return U @ torch.diag_embed(torch.log(d)) @ U.transpose(-1, -2)


def random_spd_batch(batch: int, n: int, device=None,
                     dtype=torch.float32, jitter: float = 1e-3) -> torch.Tensor:
    # genere des matrices spd aleatoires
    # a*a^t + un petit bruit sur la diagonale
    A = torch.randn(batch, n, n, device=device, dtype=dtype)
    I = torch.eye(n, device=device, dtype=dtype).expand(batch, n, n)
    return A @ A.transpose(-1, -2) + jitter * I
