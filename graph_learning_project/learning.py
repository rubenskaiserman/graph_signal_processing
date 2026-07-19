import numpy as np

from graph_generation import upper_triangle_indices, vector_to_adjacency


def S_apply(w, i, j, m):
    return np.bincount(i, weights=w, minlength=m) + np.bincount(j, weights=w, minlength=m)


def ST_apply(d, i, j):
    return d[i] + d[j]


def objective(w, z, i, j, m, alpha, beta):
    degrees = S_apply(w, i, j, m)
    return 2.0 * np.dot(w, z) - alpha * np.log(degrees).sum() + beta * np.dot(w, w)


def solve_once(z, m, alpha, beta, max_iter, tol, step_fraction, w0=None, d0=None):
    i, j = upper_triangle_indices(m)
    n_edges = m * (m - 1) // 2
    w = np.full(n_edges, 1e-3) if w0 is None else w0.copy()
    d = S_apply(w, i, j, m) if d0 is None else d0.copy()

    gamma = step_fraction / (2.0 * beta + np.sqrt(2.0 * (m - 1)))
    rel_w = np.inf
    rel_d = np.inf

    for iteration in range(1, max_iter + 1):
        y = w - gamma * (2.0 * beta * w + ST_apply(d, i, j))
        y_bar = d + gamma * S_apply(w, i, j, m)

        p = np.maximum(0.0, y - 2.0 * gamma * z)
        p_bar = 0.5 * (y_bar - np.sqrt(y_bar**2 + 4.0 * alpha * gamma))

        q = p - gamma * (2.0 * beta * p + ST_apply(p_bar, i, j))
        q_bar = p_bar + gamma * S_apply(p, i, j, m)

        w_new = w - y + q
        d_new = d - y_bar + q_bar

        rel_w = np.linalg.norm(w_new - w) / max(1.0, np.linalg.norm(w))
        rel_d = np.linalg.norm(d_new - d) / max(1.0, np.linalg.norm(d))
        w, d = w_new, d_new

        if rel_w < tol and rel_d < tol:
            break

    info = {
        "iterations": iteration,
        "rel_w": float(rel_w),
        "rel_d": float(rel_d),
        "objective": float(objective(w, z, i, j, m, alpha, beta)),
        "converged": bool(rel_w < tol and rel_d < tol),
        "attempt": 0,
    }
    return vector_to_adjacency(w, m), w, d, info


def solve_kalofolias(z, m, alpha=1.0, beta=1.0, tol=1e-5):
    attempts = [(5000, 0.8), (10000, 0.8), (10000, 0.5)]
    w0 = None
    d0 = None
    last = None

    for attempt, (max_iter, step_fraction) in enumerate(attempts):
        W, w, d, info = solve_once(z, m, alpha, beta, max_iter, tol, step_fraction, w0, d0)
        info["attempt"] = attempt
        last = (W, w, info)
        if info["converged"]:
            return last
        w0, d0 = w, d

    return last
