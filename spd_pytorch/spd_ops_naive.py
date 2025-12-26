import torch


def vectorize_lower_naive(M: torch.Tensor) -> torch.Tensor:
    # ici on prend juste la partie basse de la matrice
    # et on la met dans un vecteur
    # c est pas opti mais facile a comprendre
    *batch, n, _ = M.shape
    out_dim = n * (n + 1) // 2
    out = torch.empty(*batch, out_dim, device=M.device, dtype=M.dtype)

    M2 = M.reshape(-1, n, n)
    out2 = out.reshape(-1, out_dim)

    # double boucle python -> tres lent
    for b in range(M2.shape[0]):
        k = 0
        for i in range(n):
            for j in range(i + 1):
                out2[b, k] = M2[b, i, j]
                k += 1
    return out


def devectorize_lower_naive(v: torch.Tensor, n: int) -> torch.Tensor:
    # on reconstruit une matrice symetrique a partir du vecteur
    *batch, d = v.shape
    out = torch.zeros(*batch, n, n, device=v.device, dtype=v.dtype)

    v2 = v.reshape(-1, d)
    out2 = out.reshape(-1, n, n)

    for b in range(v2.shape[0]):
        k = 0
        for i in range(n):
            for j in range(i + 1):
                out2[b, i, j] = v2[b, k]
                out2[b, j, i] = v2[b, k]  # symetrie
                k += 1
    return out


def spd_matrix_sqrt_naive(M: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    # calcul de la racine d une matrice spd
    # on passe par les valeurs propres
    *batch, n, _ = M.shape
    out = torch.empty(*batch, n, n, device=M.device, dtype=M.dtype)

    M2 = M.reshape(-1, n, n)
    out2 = out.reshape(-1, n, n)

    for b in range(M2.shape[0]):
        d, U = torch.linalg.eigh(M2[b])
        d = torch.clamp(d, min=eps)  # evite les pb numeriques
        out2[b] = U @ torch.diag(torch.sqrt(d)) @ U.t()
    return out


def spd_matrix_log_naive(M: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    # meme idee que pour la racine mais avec le log
    *batch, n, _ = M.shape
    out = torch.empty(*batch, n, n, device=M.device, dtype=M.dtype)

    M2 = M.reshape(-1, n, n)
    out2 = out.reshape(-1, n, n)

    for b in range(M2.shape[0]):
        d, U = torch.linalg.eigh(M2[b])
        d = torch.clamp(d, min=eps)
        out2[b] = U @ torch.diag(torch.log(d)) @ U.t()
    return out
