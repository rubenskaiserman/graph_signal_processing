from collections import defaultdict

import numpy as np

from graph_generation import adjacency_to_vector


METRICS = [
    "edge_abs_l1_mean",
    "edge_abs_l2_mean",
    "edge_mean_relative_error",
    "degree_mean_relative_error",
    "precision",
    "sensibilidade",
    "f1",
]


def graph_metrics(W_true, W_hat, support_rtol=1e-4):
    w_true = adjacency_to_vector(W_true)
    w_hat = adjacency_to_vector(W_hat)
    edge_error = w_hat - w_true

    d_true = W_true.sum(axis=1)
    d_hat = W_hat.sum(axis=1)
    degree_error = d_hat - d_true

    true_support = w_true > 1e-12
    learned_threshold = support_rtol * W_hat.max() if W_hat.max() > 0 else np.inf
    learned_support = w_hat > learned_threshold

    tp = np.sum(true_support & learned_support)
    fp = np.sum(~true_support & learned_support)
    fn = np.sum(true_support & ~learned_support)

    precision = tp / (tp + fp) if tp + fp else 0.0
    sensibilidade = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * sensibilidade / (precision + sensibilidade) if precision + sensibilidade else 0.0

    true_edge_abs_error = np.abs(edge_error[true_support])
    true_edge_relative_error = np.abs(edge_error[true_support]) / np.maximum(np.abs(w_true[true_support]), 1e-12)
    degree_support = d_true > 1e-12
    degree_relative_error = np.abs(degree_error[degree_support]) / np.abs(d_true[degree_support])

    return {
        "edge_abs_l1_mean": float(np.mean(true_edge_abs_error)),
        "edge_abs_l2_mean": float(np.mean(true_edge_abs_error**2)),
        "edge_mean_relative_error": float(true_edge_relative_error.mean()) if true_edge_relative_error.size else np.nan,
        "degree_mean_relative_error": float(np.mean(degree_relative_error)) if degree_relative_error.size else np.nan,
        "precision": float(precision),
        "sensibilidade": float(sensibilidade),
        "f1": float(f1),
    }


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["graph"], row["filter"])].append(row)

    rows = []
    for (graph_name, filter_name), group in sorted(grouped.items()):
        for metric in METRICS:
            values = np.array([row[metric] for row in group], dtype=float)
            rows.append({
                "graph": graph_name,
                "filter": filter_name,
                "metric": metric,
                "mean": float(values.mean()),
                "min": float(values.min()),
                "median": float(np.median(values)),
                "max": float(values.max()),
            })
    return rows


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def print_table(rows, columns):
    widths = {col: max(len(col), *(len(format_value(row[col])) for row in rows)) for col in columns}
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(format_value(row[col]).ljust(widths[col]) for col in columns))
