import numpy as np
from pygsp import filters, graphs
from scipy.spatial.distance import pdist, squareform


def upper_triangle_indices(m):
    return np.triu_indices(m, k=1)


def laplacian(W):
    return np.diag(W.sum(axis=1)) - W


def normalize_laplacian_norm(W):
    L = laplacian(W)
    norm = np.linalg.eigvalsh(L)[-1]
    return W / norm, L / norm


def random_geometric_graph(m, sigma, cutoff, rng):
    positions = rng.uniform(0.0, 1.0, size=(m, 2))
    distances2 = squareform(pdist(positions, metric="sqeuclidean"))
    W = np.exp(-distances2 / sigma**2)
    W[W < cutoff] = 0.0
    np.fill_diagonal(W, 0.0)
    return W, positions


def erdos_renyi_graph(m, p, rng):
    upper = np.triu(rng.random((m, m)) < p, k=1)
    return (upper + upper.T).astype(float)


def adjacency_to_vector(W):
    return W[upper_triangle_indices(W.shape[0])]


def vector_to_adjacency(w, m):
    W = np.zeros((m, m), dtype=float)
    i, j = upper_triangle_indices(m)
    W[i, j] = w
    W[j, i] = w
    return W


def filter_signal(W, X0, kind):
    G = graphs.Graph(W)
    G.compute_fourier_basis()
    if kind == "Tikhonov":
        filt = filters.Filter(G, lambda x: 1.0 / (1.0 + 10.0 * x))
    else:
        filt = filters.Heat(G, scale=10.0)

    return filt.filter(X0, method="exact")


def signal_distances(X):
    Z = squareform(pdist(X, metric="sqeuclidean"))
    i, j = upper_triangle_indices(X.shape[0])
    median = np.median(Z[i, j])
    return Z / median
