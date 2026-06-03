import gc
import json
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from k_means_constrained import KMeansConstrained
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from src.utils_dipl import (
    calculate_multiple_run_statistics,
    collect_multiple_runs,
    create_windowed_dataset,
    split_time_series,
)


BASE_VARIANT_NAMES = ["raw_xy", "rel_xy", "dxy"]
VARIANT_NAMES = BASE_VARIANT_NAMES
SHUFFLE_CONTROL_SEED = 2026
CLUSTER_PLOT_DATA_SPLITS = ["train", "val", "test"]


def echo_number(echo_dir):
    match = re.search(r"echo(\d+)$", Path(echo_dir).name)
    return int(match.group(1))


def amplitude_result_name(amplitude_dir):
    match = re.search(r"(\d+)perc", Path(amplitude_dir).name)
    return f"gamma_{match.group(1)}perc"


def amplitude_percent(amplitude_dir):
    match = re.search(r"(\d+)perc", Path(amplitude_dir).name)
    return int(match.group(1))


def discover_echo_folders(data_root):
    data_root = Path(data_root)
    echo_jobs = []

    for amplitude_dir in sorted(
        (path for path in data_root.iterdir() if path.is_dir()),
        key=amplitude_percent,
    ):
        echo_dirs = [
            path
            for path in amplitude_dir.iterdir()
            if path.is_dir() and path.name.startswith("individual_beads_echo")
        ]
        echo_dirs = sorted(echo_dirs, key=echo_number)

        for echo_dir in echo_dirs:
            echo_jobs.append(
                {
                    "amplitude_name": amplitude_dir.name,
                    "amplitude_result_name": amplitude_result_name(amplitude_dir),
                    "echo_name": f"echo{echo_number(echo_dir)}",
                    "echo_dir": echo_dir,
                }
            )

    return echo_jobs


def make_echo_result_dirs(results_root, amplitude_result_name, echo_name):
    results_dir = Path(results_root) / amplitude_result_name / echo_name

    dirs = {
        "root": results_dir,
        "individual_clusters": results_dir / "individual_clusters",
        "individual_cluster_time_series": results_dir / "individual_cluster_time_series",
        "acf": results_dir / "acf",
        "training": results_dir / "training_cluster_results",
        "heatmaps": results_dir / "heatmaps",
        "self_alpha": results_dir / "self_alpha_lag1",
        "cluster_plot_data": results_dir / "cluster_plot_data",
        "event_interpretability": results_dir / "event_interpretability_plots",
        "event_conditioned_summary": results_dir / "event_conditioned_summary",
        "event_raw_line_plots": results_dir / "event_raw_line_plots",
    }

    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    return dirs


def load_particles_full_frame(echo_dir, full_frame_length=599):
    files = sorted(Path(echo_dir).glob("bead_*.mat"))

    particle_ids = []
    x_list = []
    y_list = []
    frame_list = []

    for file in files:
        data = loadmat(file)["bsec"][:, :5]

        if len(data) != full_frame_length:
            continue

        data = data[np.argsort(data[:, 2])]

        particle_ids.append(int(data[0, 3]))
        x_list.append(data[:, 0])
        y_list.append(data[:, 1])
        frame_list.append(data[:, 2].astype(int))

    particle_ids = np.array(particle_ids, dtype=int)
    x_matrix = np.column_stack(x_list)
    y_matrix = np.column_stack(y_list)
    frames = frame_list[0]

    return frames, particle_ids, x_matrix, y_matrix


def make_cluster_from_indices(particle_ids, x_matrix, y_matrix, positions, selected_idx):
    selected_idx = np.asarray(selected_idx, dtype=int)
    cluster_positions = positions[selected_idx]

    if len(selected_idx) == 1:
        center_local_idx = 0
        score = 0.0
    else:
        dist = cdist(cluster_positions, cluster_positions)
        mean_dist = dist.sum(axis=1) / (len(selected_idx) - 1)
        center_local_idx = np.argmin(mean_dist)
        score = mean_dist[center_local_idx]

    center_idx = selected_idx[center_local_idx]

    return {
        "center_particle_id": int(particle_ids[center_idx]),
        "selected_particle_ids": particle_ids[selected_idx],
        "selected_indices": selected_idx,
        "score": float(score),
        "x_selected": x_matrix[:, selected_idx],
        "y_selected": y_matrix[:, selected_idx],
        "x0": positions[selected_idx, 0],
        "y0": positions[selected_idx, 1],
        "cluster_size": int(len(selected_idx)),
    }


def select_clusters_constrained_kmeans(
    particle_ids,
    x_matrix,
    y_matrix,
    n_particles,
    frame_index=0,
    random_state=0,
    n_init=20,
):
    positions = np.column_stack([x_matrix[frame_index], y_matrix[frame_index]])
    n_total = len(particle_ids)

    if n_total == 0:
        return []

    n_clusters = int(np.ceil(n_total / n_particles))
    size_min = n_total // n_clusters
    size_max = int(np.ceil(n_total / n_clusters))

    model = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        random_state=random_state,
        n_init=n_init,
    )

    labels = model.fit_predict(positions)

    clusters = []
    for label in range(n_clusters):
        selected_idx = np.where(labels == label)[0]
        cluster = make_cluster_from_indices(
            particle_ids=particle_ids,
            x_matrix=x_matrix,
            y_matrix=y_matrix,
            positions=positions,
            selected_idx=selected_idx,
        )
        clusters.append(cluster)

    clusters = sorted(clusters, key=lambda cluster: cluster["score"])

    for cluster_id, cluster in enumerate(clusters, start=1):
        cluster["cluster_id"] = cluster_id

    return clusters


def save_cluster_members(clusters, save_path, metadata=None):
    rows = []

    for cluster_idx, cluster in enumerate(clusters):
        row = {
            "cluster_idx": cluster_idx,
            "cluster_id": cluster["cluster_id"],
            "center_particle_id": cluster["center_particle_id"],
            "cluster_size": cluster["cluster_size"],
            "selected_particle_ids": " ".join(
                str(int(particle_id)) for particle_id in cluster["selected_particle_ids"]
            ),
            "score": cluster["score"],
        }

        if metadata is not None:
            row.update(metadata)

        rows.append(row)

    pd.DataFrame(rows).to_csv(save_path, index=False)


def shuffle_columns(matrix, rng):
    shuffled = np.empty_like(matrix)
    for col_idx in range(matrix.shape[1]):
        shuffled[:, col_idx] = matrix[rng.permutation(matrix.shape[0]), col_idx]
    return shuffled


def variant_base_name(variant_name):
    return variant_name.replace("_shuffled", "")


def variant_is_shuffled(variant_name):
    return variant_name.endswith("_shuffled")


def make_variant_names(include_shuffled=True):
    names = BASE_VARIANT_NAMES.copy()
    if include_shuffled:
        names += [f"{name}_shuffled" for name in BASE_VARIANT_NAMES]
    return names


def build_dcits_input(x_block, y_block, frames, mode="raw_xy", shuffle_seed=None):
    x_block = np.asarray(x_block)
    y_block = np.asarray(y_block)

    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        x_block = shuffle_columns(x_block, rng)
        y_block = shuffle_columns(y_block, rng)

    if mode == "raw_xy":
        x_part = x_block
        y_part = y_block
        frames_used = frames
    elif mode == "rel_xy":
        x_part = x_block - x_block[0:1, :]
        y_part = y_block - y_block[0:1, :]
        frames_used = frames
    elif mode == "dxy":
        x_part = np.diff(x_block, axis=0)
        y_part = np.diff(y_block, axis=0)
        frames_used = frames[1:]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    data = np.concatenate([x_part, y_part], axis=1).T
    time_series = torch.tensor(data, dtype=torch.float32)

    return time_series, frames_used


def build_training_variants(x_selected, y_selected, frames, include_shuffled=True, shuffle_seed=SHUFFLE_CONTROL_SEED):
    variants = {
        "raw_xy": build_dcits_input(x_selected, y_selected, frames, mode="raw_xy"),
        "rel_xy": build_dcits_input(x_selected, y_selected, frames, mode="rel_xy"),
        "dxy": build_dcits_input(x_selected, y_selected, frames, mode="dxy"),
    }

    if include_shuffled:
        for mode in BASE_VARIANT_NAMES:
            variants[f"{mode}_shuffled"] = build_dcits_input(
                x_selected,
                y_selected,
                frames,
                mode=mode,
                shuffle_seed=shuffle_seed,
            )

    return variants


def build_variants(x_selected, y_selected, frames, variant_names=VARIANT_NAMES):
    include_shuffled = any(variant_is_shuffled(name) for name in variant_names)
    variants = build_training_variants(
        x_selected,
        y_selected,
        frames,
        include_shuffled=include_shuffled,
    )
    return {variant_name: variants[variant_name] for variant_name in variant_names}


def make_label_map(selected_ids):
    labels_raw = [f"x_{pid}" for pid in selected_ids] + [f"y_{pid}" for pid in selected_ids]
    labels_rel = [f"xrel_{pid}" for pid in selected_ids] + [f"yrel_{pid}" for pid in selected_ids]
    labels_dxy = [f"dx_{pid}" for pid in selected_ids] + [f"dy_{pid}" for pid in selected_ids]

    return {
        "raw_xy": labels_raw,
        "rel_xy": labels_rel,
        "dxy": labels_dxy,
        "raw_xy_shuffled": [f"{label}_shuf" for label in labels_raw],
        "rel_xy_shuffled": [f"{label}_shuf" for label in labels_rel],
        "dxy_shuffled": [f"{label}_shuf" for label in labels_dxy],
    }


def summarize_loss_curves(curves):
    max_len = max(len(curve) for curve in curves)
    curve_matrix = np.full((len(curves), max_len), np.nan)

    for run_idx, curve in enumerate(curves):
        curve_matrix[run_idx, : len(curve)] = curve

    return {
        "mean": np.nanmean(curve_matrix, axis=0),
        "std": np.nanstd(curve_matrix, axis=0),
    }


def run_multiple_dcits(
    time_series,
    name,
    device,
    seed,
    n_runs,
    window_size,
    temperature,
    order,
    epochs,
):
    train_config = {
        "verbose": False,
        "device": device,
        "seed": seed,
        "learning_rate": 1e-3,
        "scheduler_patience": 5,
        "early_stopping_modifier": 2,
        "criterion": nn.MSELoss(),
        "epochs": epochs,
        "batch_size": 64,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
    }

    run_results = collect_multiple_runs(
        n_runs=n_runs,
        time_series=time_series,
        window_size=window_size,
        temperature=temperature,
        order=order,
        config=train_config,
        seed=seed,
        verbose=False,
    )

    split_stats = {
        split_name: calculate_multiple_run_statistics(run_results, split=split_name)
        for split_name in ["train", "val", "test"]
    }

    run_keys = [key for key in run_results.keys() if key.startswith("run_")]
    train_curves = [run_results[run_key]["train_losses"] for run_key in run_keys]
    val_curves = [run_results[run_key]["val_losses"] for run_key in run_keys]

    return {
        "name": name,
        "n_runs": n_runs,
        "run_keys": run_keys,
        "runs": run_results,
        "summary": run_results["summary"],
        "split_stats": split_stats,
        "train_curve": summarize_loss_curves(train_curves),
        "val_curve": summarize_loss_curves(val_curves),
    }


def plot_all_clusters(x_full, y_full, clusters, save_path, figure_dpi):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x_full[0], y_full[0], s=10, alpha=0.15, color="gray")

    for cluster in clusters:
        ax.scatter(cluster["x0"], cluster["y0"], s=35, label=f'C{cluster["cluster_id"]}')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Constrained k-means clusters")
    ax.axis("equal")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_individual_clusters(
    x_full,
    y_full,
    particle_ids,
    clusters,
    output_dir,
    figure_dpi,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for cluster_idx, cluster in enumerate(clusters):
        x_selected = cluster["x_selected"]
        y_selected = cluster["y_selected"]
        selected_ids = cluster["selected_particle_ids"]
        cluster_size = cluster["cluster_size"]

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        axes[0].scatter(x_full[0], y_full[0], s=12, alpha=0.35, label="All beads")
        axes[0].scatter(cluster["x0"], cluster["y0"], s=45, label=f"{cluster_size} compact beads")

        center_idx = np.flatnonzero(particle_ids == cluster["center_particle_id"])[0]
        axes[0].scatter(
            x_full[0, center_idx],
            y_full[0, center_idx],
            s=90,
            marker="x",
            label="Center bead",
        )
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title(f"Cluster {cluster_idx}: {cluster_size} tracer beads")
        axes[0].legend()
        axes[0].axis("equal")

        for particle_idx in range(x_selected.shape[1]):
            axes[1].plot(
                x_selected[:, particle_idx],
                y_selected[:, particle_idx],
                lw=1,
                label=f"Particle {selected_ids[particle_idx]}",
            )

        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title(f"Local traces for {cluster_size} beads")
        axes[1].axis("equal")
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        axes[2].scatter(x_full[0], y_full[0], s=10, alpha=0.2)

        for particle_idx in range(x_selected.shape[1]):
            axes[2].plot(x_selected[:, particle_idx], y_selected[:, particle_idx], lw=1)

        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title(f"Global traces for {cluster_size} beads")
        axes[2].axis("equal")

        fig.tight_layout()
        save_path = output_dir / f"cluster_{cluster_idx}.png"
        fig.savefig(save_path, dpi=figure_dpi, bbox_inches="tight")
        plt.close(fig)


def plot_cluster_time_series(frames, clusters, output_dir, figure_dpi, event_percentile=99.6):
    output_dir.mkdir(parents=True, exist_ok=True)

    for cluster_idx, cluster in enumerate(clusters):
        y_selected = cluster["y_selected"]
        selected_ids = cluster["selected_particle_ids"]
        cluster_size = cluster["cluster_size"]

        y_rel = y_selected - y_selected[0:1, :]
        dy = np.diff(y_selected, axis=0)
        frames_dy = frames[1:]
        abs_dy = np.abs(dy)
        event_threshold = np.percentile(abs_dy, event_percentile)
        event_mask = abs_dy >= event_threshold

        fig, axes = plt.subplots(3, 1, figsize=(12, 15))

        for particle_idx in range(y_selected.shape[1]):
            line, = axes[0].plot(
                frames,
                y_selected[:, particle_idx],
                lw=1.5,
                label=f"bead {selected_ids[particle_idx]}",
            )
            color = line.get_color()
            event_time_idx = np.where(event_mask[:, particle_idx])[0]
            axes[0].scatter(
                frames_dy[event_time_idx],
                y_selected[event_time_idx + 1, particle_idx],
                color=color,
                edgecolor="black",
                s=35,
                zorder=5,
            )

        axes[0].set_xlabel("Frame")
        axes[0].set_ylabel("y")
        axes[0].set_title(f"Cluster {cluster_idx}: y(t) for {cluster_size} beads")
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        for particle_idx in range(y_rel.shape[1]):
            line, = axes[1].plot(
                frames,
                y_rel[:, particle_idx],
                lw=1.5,
                label=f"bead {selected_ids[particle_idx]}",
            )
            color = line.get_color()
            event_time_idx = np.where(event_mask[:, particle_idx])[0]
            axes[1].scatter(
                frames_dy[event_time_idx],
                y_rel[event_time_idx + 1, particle_idx],
                color=color,
                edgecolor="black",
                s=35,
                zorder=5,
            )

        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("y - y0")
        axes[1].set_title(f"Cluster {cluster_idx}: y(t) - y0 for {cluster_size} beads")
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        for particle_idx in range(dy.shape[1]):
            line, = axes[2].plot(
                frames_dy,
                dy[:, particle_idx],
                lw=1.2,
                label=f"bead {selected_ids[particle_idx]}",
            )
            color = line.get_color()
            event_time_idx = np.where(event_mask[:, particle_idx])[0]
            axes[2].scatter(
                frames_dy[event_time_idx],
                dy[event_time_idx, particle_idx],
                color=color,
                edgecolor="black",
                s=35,
                zorder=5,
            )

        axes[2].set_xlabel("Frame")
        axes[2].set_ylabel("dy = y(t) - y(t-1)")
        axes[2].set_title(
            f"Cluster {cluster_idx}: frame-to-frame dy for {cluster_size} beads "
            f"(|dy| >= {event_threshold:.4g}, {event_percentile}th percentile)"
        )
        axes[2].axhline(event_threshold, color="black", linestyle="--", lw=1)
        axes[2].axhline(-event_threshold, color="black", linestyle="--", lw=1)
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        for ax in axes:
            ax.grid(True)

        fig.tight_layout()
        save_path = output_dir / f"cluster_series_{cluster_idx}.png"
        fig.savefig(save_path, dpi=figure_dpi, bbox_inches="tight")
        plt.close(fig)


def run_adf_tests(clusters, frames):
    counts = {"raw_xy": 0, "rel_xy": 0, "dxy": 0}
    n_tested_series = 0

    for cluster in clusters:
        x_selected = cluster["x_selected"]
        y_selected = cluster["y_selected"]

        time_series_raw, _ = build_dcits_input(x_selected, y_selected, frames, mode="raw_xy")
        time_series_rel, _ = build_dcits_input(x_selected, y_selected, frames, mode="rel_xy")
        time_series_diff, _ = build_dcits_input(x_selected, y_selected, frames, mode="dxy")

        n_series = time_series_raw.shape[0]
        n_tested_series += n_series

        for series_idx in range(n_series):
            p_raw = adfuller(time_series_raw[series_idx].numpy())[1]
            p_rel = adfuller(time_series_rel[series_idx].numpy())[1]
            p_diff = adfuller(time_series_diff[series_idx].numpy())[1]

            if p_raw < 0.05:
                counts["raw_xy"] += 1
            if p_rel < 0.05:
                counts["rel_xy"] += 1
            if p_diff < 0.05:
                counts["dxy"] += 1

    return {
        "num_raw_stationary": counts["raw_xy"],
        "num_rel_stationary": counts["rel_xy"],
        "num_diff_stationary": counts["dxy"],
        "n_tested_series": n_tested_series,
        "n_adf_tests": n_tested_series * 3,
    }


def plot_acf_for_clusters(
    clusters,
    frames,
    output_dir,
    figure_dpi,
    lags=10,
    include_shuffled=True,
    shuffle_seed=SHUFFLE_CONTROL_SEED,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for cluster_idx, cluster in enumerate(clusters):
        x_selected = cluster["x_selected"]
        y_selected = cluster["y_selected"]

        variants = build_training_variants(
            x_selected,
            y_selected,
            frames,
            include_shuffled=include_shuffled,
            shuffle_seed=shuffle_seed + cluster_idx,
        )
        variants = {variant_name: bundle[0] for variant_name, bundle in variants.items()}

        n_particles = x_selected.shape[1]
        n_series = variants["raw_xy"].shape[0]

        cluster_dir = output_dir / f"cluster_{cluster_idx}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        for series_idx in range(n_series):
            if series_idx < n_particles:
                series_name = f"x_{series_idx + 1}"
            else:
                series_name = f"y_{series_idx + 1 - n_particles}"

            for variant_name, time_series in variants.items():
                variant_dir = cluster_dir / variant_name
                variant_dir.mkdir(parents=True, exist_ok=True)

                fig = plot_acf(
                    time_series[series_idx].numpy(),
                    lags=lags,
                    title=f"cluster_{cluster_idx}_{series_name}_ACF_{variant_name}",
                )
                fig.savefig(
                    variant_dir / f"{series_name}_acf.png",
                    dpi=figure_dpi,
                    bbox_inches="tight",
                )
                plt.close(fig)

def plot_acf_pacf_for_clusters(*args, **kwargs):
    return plot_acf_for_clusters(*args, **kwargs)


def run_granger_y_tests(clusters, lag=3, alpha=0.05):
    results = []

    for cluster_idx, cluster in enumerate(clusters):
        y_selected = cluster["y_selected"]
        selected_ids = cluster["selected_particle_ids"]

        y_diff = np.diff(y_selected, axis=0).T
        n_particles = y_diff.shape[0]

        for a, b in combinations(range(n_particles), 2):
            data_a_causes_b = np.column_stack([y_diff[b], y_diff[a]])
            result_a_b = grangercausalitytests(
                data_a_causes_b,
                maxlag=[lag],
                verbose=False,
            )
            p_a_b = result_a_b[lag][0]["ssr_ftest"][1]

            data_b_causes_a = np.column_stack([y_diff[a], y_diff[b]])
            result_b_a = grangercausalitytests(
                data_b_causes_a,
                maxlag=[lag],
                verbose=False,
            )
            p_b_a = result_b_a[lag][0]["ssr_ftest"][1]

            if p_a_b < alpha and p_b_a < alpha:
                results.append(
                    {
                        "cluster_idx": cluster_idx,
                        "bead_a": int(selected_ids[a]),
                        "bead_b": int(selected_ids[b]),
                        "p_a_to_b": float(p_a_b),
                        "p_b_to_a": float(p_b_a),
                    }
                )

    return results


def plot_granger_y_pairs(x_full, y_full, clusters, granger_results, save_path, figure_dpi):
    fig, ax = plt.subplots(figsize=(9, 9))

    for cluster in clusters:
        x_selected = cluster["x_selected"]
        y_selected = cluster["y_selected"]

        for particle_idx in range(x_selected.shape[1]):
            ax.plot(
                x_selected[:, particle_idx],
                y_selected[:, particle_idx],
                color="black",
                lw=2,
                alpha=0.5,
            )

    if len(granger_results) > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, len(granger_results)))

        for pair_idx, result in enumerate(granger_results):
            cluster_idx = result["cluster_idx"]
            bead_1 = result["bead_a"]
            bead_2 = result["bead_b"]
            cluster = clusters[cluster_idx]

            x_selected = cluster["x_selected"]
            y_selected = cluster["y_selected"]
            selected_ids = cluster["selected_particle_ids"]

            idx_1 = np.where(selected_ids == bead_1)[0][0]
            idx_2 = np.where(selected_ids == bead_2)[0][0]
            pair_color = colors[pair_idx]

            ax.plot(x_selected[:, idx_1], y_selected[:, idx_1], color=pair_color, lw=2.5)
            ax.plot(x_selected[:, idx_2], y_selected[:, idx_2], color=pair_color, lw=2.5)
            ax.scatter(x_selected[0, idx_1], y_selected[0, idx_1], color=pair_color, s=45)
            ax.scatter(x_selected[0, idx_2], y_selected[0, idx_2], color=pair_color, s=45)
            ax.plot(
                [x_selected[0, idx_1], x_selected[0, idx_2]],
                [y_selected[0, idx_1], y_selected[0, idx_2]],
                color=pair_color,
                lw=1.5,
                alpha=0.8,
            )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("All clusters with mutual y-Granger bead pairs highlighted")
    ax.axis("equal")
    fig.savefig(save_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_heatmaps(
    alpha_mean,
    labels,
    window_size,
    figure_dpi,
    title="",
    threshold=0.0,
    flip_lag_axis=True,
    cmap="coolwarm",
    save_path=None,
):
    alpha_plot = np.array(alpha_mean, copy=True)

    if flip_lag_axis:
        alpha_plot = np.flip(alpha_plot, axis=2)

    if threshold > 0:
        alpha_plot[np.abs(alpha_plot) < threshold] = 0.0

    n_series = alpha_plot.shape[0]
    vmax = np.max(np.abs(alpha_plot))
    if vmax == 0:
        vmax = 1.0

    fig, axes = plt.subplots(
        1,
        window_size,
        figsize=(4 * window_size, 5),
        constrained_layout=True,
    )

    if window_size == 1:
        axes = [axes]

    split = n_series // 2

    for lag_idx, ax in enumerate(axes):
        matrix = alpha_plot[:, :, lag_idx]
        im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"lag {lag_idx + 1}")
        ax.set_xlabel("source i")
        ax.set_ylabel("target j")
        ax.set_xticks(range(n_series))
        ax.set_yticks(range(n_series))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)

        if split < n_series:
            ax.axvline(split - 0.5, color="black", linewidth=1)
            ax.axhline(split - 0.5, color="black", linewidth=1)

    fig.suptitle(title)
    fig.colorbar(im, ax=axes, shrink=0.85, label=r"mean $alpha$")

    if save_path is not None:
        fig.savefig(save_path, dpi=figure_dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def get_alpha_seq_physical_split_stats(all_results, variant_name, split_name="test", order_idx=1):
    result = all_results[variant_name]["result"]
    alpha_runs = []

    for run_key in result["run_keys"]:
        split_result = result["runs"][run_key]["split_results"][split_name]
        alpha_seq = to_numpy_array(split_result["focus"][order_idx] * split_result["coefficients"][order_idx])
        alpha_runs.append(np.flip(alpha_seq, axis=3))

    alpha_runs = np.stack(alpha_runs, axis=0)
    return alpha_runs.mean(axis=0), alpha_runs.std(axis=0)


def get_alpha_seq_all_sets_stats(all_results, variant_name, order_idx=1):
    alpha_train_mean, alpha_train_std = get_alpha_seq_physical_split_stats(
        all_results, variant_name, split_name="train", order_idx=order_idx
    )
    alpha_val_mean, alpha_val_std = get_alpha_seq_physical_split_stats(
        all_results, variant_name, split_name="val", order_idx=order_idx
    )
    alpha_test_mean, alpha_test_std = get_alpha_seq_physical_split_stats(
        all_results, variant_name, split_name="test", order_idx=order_idx
    )

    alpha_mean_all = np.concatenate([alpha_train_mean, alpha_val_mean, alpha_test_mean], axis=0)
    alpha_std_all = np.concatenate([alpha_train_std, alpha_val_std, alpha_test_std], axis=0)
    test_start = alpha_train_mean.shape[0] + alpha_val_mean.shape[0]

    return alpha_mean_all, alpha_std_all, test_start


def to_numpy_array(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def plot_data_metric_arrays_from_split(split_result, order_idx=1):
    focus = to_numpy_array(split_result["focus"][order_idx])
    coefficients = to_numpy_array(split_result["coefficients"][order_idx])
    alpha = focus * coefficients
    q_values = to_numpy_array(split_result["inputs"])
    alpha_times_q = alpha * q_values[:, None, :, :]

    return {
        "focuser": np.flip(focus, axis=3).astype(np.float32),
        "alpha": np.flip(alpha, axis=3).astype(np.float32),
        "alpha_times_Q": np.flip(alpha_times_q, axis=3).astype(np.float32),
    }


def display_metric_arrays_from_split(split_result, order_idx=1):
    arrays = plot_data_metric_arrays_from_split(split_result, order_idx=order_idx)
    return {
        "Focuser": arrays["focuser"],
        "alpha": arrays["alpha"],
        "alpha * Q": arrays["alpha_times_Q"],
    }


def mean_display_metrics_across_runs(all_results, variant_name, split_names, order_idx=1):
    result = all_results[variant_name]["result"]
    metric_runs = {"Focuser": [], "alpha": [], "alpha * Q": []}

    for run_key in result["run_keys"]:
        metric_parts = {"Focuser": [], "alpha": [], "alpha * Q": []}

        for split_name in split_names:
            split_result = result["runs"][run_key]["split_results"][split_name]
            split_metrics = display_metric_arrays_from_split(split_result, order_idx=order_idx)

            for metric_name, values in split_metrics.items():
                metric_parts[metric_name].append(values)

        for metric_name in metric_runs:
            metric_runs[metric_name].append(np.concatenate(metric_parts[metric_name], axis=0))

    return {
        metric_name: np.mean(np.stack(run_values, axis=0), axis=0)
        for metric_name, run_values in metric_runs.items()
    }


def plot_data_metric_stats(result, split_names=CLUSTER_PLOT_DATA_SPLITS, order_idx=1):
    metric_runs = {"focuser": [], "alpha": [], "alpha_times_Q": []}

    for run_key in result["run_keys"]:
        metric_parts = {"focuser": [], "alpha": [], "alpha_times_Q": []}

        for split_name in split_names:
            split_result = result["runs"][run_key]["split_results"][split_name]
            split_metrics = plot_data_metric_arrays_from_split(split_result, order_idx=order_idx)

            for metric_name, values in split_metrics.items():
                metric_parts[metric_name].append(values)

        for metric_name in metric_runs:
            metric_runs[metric_name].append(np.concatenate(metric_parts[metric_name], axis=0))

    metric_stats = {}
    for metric_name, run_values in metric_runs.items():
        stacked = np.stack(run_values, axis=0)
        metric_stats[f"{metric_name}_mean"] = stacked.mean(axis=0).astype(np.float32)
        metric_stats[f"{metric_name}_std"] = stacked.std(axis=0).astype(np.float32)

    return metric_stats


def window_target_indices_from_time_series(time_series, window_size):
    train_series, val_series, test_series = split_time_series(
        time_series,
        train_ratio=0.8,
        val_ratio=0.1,
        window_size=window_size,
    )
    split_lengths = [train_series.shape[1], val_series.shape[1], test_series.shape[1]]
    split_starts = [0, split_lengths[0], split_lengths[0] + split_lengths[1]]
    target_indices = []
    split_boundaries = []
    total_windows = 0

    for split_start, split_length in zip(split_starts, split_lengths):
        n_windows = split_length - window_size
        target_indices.append(split_start + window_size + np.arange(n_windows))
        total_windows += n_windows
        split_boundaries.append(total_windows)

    return np.concatenate(target_indices), np.array(split_boundaries[:-1], dtype=int)


def window_target_indices_for_cluster_variant(
    cluster,
    variant_name,
    cluster_idx,
    frames,
    window_size,
    shuffle_seed=SHUFFLE_CONTROL_SEED,
):
    variants = build_training_variants(
        cluster["x_selected"],
        cluster["y_selected"],
        frames,
        include_shuffled=variant_is_shuffled(variant_name),
        shuffle_seed=shuffle_seed + cluster_idx,
    )
    return window_target_indices_from_time_series(variants[variant_name][0], window_size)


def save_cluster_plot_data(
    cluster_idx,
    cluster,
    all_results,
    label_map,
    variants,
    save_dir,
    variant_names,
    frames,
    window_size,
    metadata=None,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    save_arrays = {
        "frames": np.asarray(frames),
        "selected_ids": np.asarray(cluster["selected_particle_ids"]),
        "x_selected": np.asarray(cluster["x_selected"], dtype=np.float32),
        "y_selected": np.asarray(cluster["y_selected"], dtype=np.float32),
    }

    saved_variants = []
    plot_metadata = {
        "cluster_idx": int(cluster_idx),
        "window_size": int(window_size),
        "order_idx": 1,
        "splits": CLUSTER_PLOT_DATA_SPLITS,
        "metrics": ["focuser", "alpha", "alpha_times_Q"],
        "description": "Run-mean/run-std arrays for recreating event plots without retraining.",
    }
    if metadata is not None:
        plot_metadata.update(metadata)

    for variant_name in variant_names:
        if variant_name not in all_results:
            continue

        saved_variants.append(variant_name)
        result = all_results[variant_name]["result"]
        time_series, frames_used = variants[variant_name]

        for array_name, values in plot_data_metric_stats(result, order_idx=1).items():
            save_arrays[f"{variant_name}__{array_name}"] = values

        target_indices, split_boundaries = window_target_indices_from_time_series(time_series, window_size)
        save_arrays[f"{variant_name}__frames_used"] = np.asarray(frames_used)
        save_arrays[f"{variant_name}__target_indices"] = target_indices.astype(int)
        save_arrays[f"{variant_name}__split_boundaries"] = split_boundaries.astype(int)
        save_arrays[f"{variant_name}__labels"] = np.asarray(label_map[variant_name], dtype=str)

    plot_metadata["variants"] = saved_variants
    save_arrays["metadata_json"] = np.asarray(json.dumps(plot_metadata))

    save_path = save_dir / f"cluster_{cluster_idx}_plot_data.npz"
    np.savez_compressed(save_path, **save_arrays)
    return save_path


def plot_training_curves(all_results, cluster_idx, save_path, figure_dpi):
    loss_ylims = {
        "raw_xy": (0, 10),
        "rel_xy": (0, 0.15),
        "dxy": (0, 0.02),
    }

    fig, axes = plt.subplots(1, len(all_results), figsize=(4.5 * len(all_results), 4))
    if len(all_results) == 1:
        axes = [axes]

    for ax, (variant_name, bundle) in zip(axes, all_results.items()):
        result = bundle["result"]
        train_mean = result["train_curve"]["mean"]
        train_std = result["train_curve"]["std"]
        val_mean = result["val_curve"]["mean"]
        val_std = result["val_curve"]["std"]

        train_x = np.arange(len(train_mean))
        val_x = np.arange(len(val_mean))

        train_line, = ax.plot(train_x, train_mean, label="train mean")
        ax.fill_between(
            train_x,
            train_mean - train_std,
            train_mean + train_std,
            color=train_line.get_color(),
            alpha=0.2,
        )

        val_line, = ax.plot(val_x, val_mean, linestyle="--", label="val mean")
        ax.fill_between(
            val_x,
            val_mean - val_std,
            val_mean + val_std,
            color=val_line.get_color(),
            alpha=0.15,
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(variant_name)
        if not variant_is_shuffled(variant_name):
            ax.set_ylim(*loss_ylims[variant_base_name(variant_name)])
        ax.legend()

    fig.suptitle("Train-validation curves (mean +/- std across runs)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps_for_cluster(
    all_results,
    label_map,
    cluster_idx,
    output_dir,
    window_size,
    figure_dpi,
    alpha_threshold=0.1,
):
    cluster_heatmap_dir = output_dir / f"cluster_{cluster_idx}"
    cluster_heatmap_dir.mkdir(parents=True, exist_ok=True)

    for variant_name, bundle in all_results.items():
        labels = label_map[variant_name]

        alpha_mean_test = bundle["result"]["split_stats"]["test"]["alpha"][1]["mean"]
        save_path = cluster_heatmap_dir / f"alpha_heatmap_test_{variant_name}.png"
        plot_alpha_heatmaps(
            alpha_mean=alpha_mean_test,
            labels=labels,
            window_size=window_size,
            figure_dpi=figure_dpi,
            title=f"Cluster {cluster_idx}: mean alpha heatmaps ({variant_name}, test)",
            threshold=alpha_threshold,
            flip_lag_axis=True,
            save_path=save_path,
        )

        alpha_mean_train = bundle["result"]["split_stats"]["train"]["alpha"][1]["mean"]
        save_path = cluster_heatmap_dir / f"alpha_heatmap_train_{variant_name}.png"
        plot_alpha_heatmaps(
            alpha_mean=alpha_mean_train,
            labels=labels,
            window_size=window_size,
            figure_dpi=figure_dpi,
            title=f"Cluster {cluster_idx}: mean alpha heatmaps ({variant_name}, train)",
            threshold=alpha_threshold,
            flip_lag_axis=True,
            save_path=save_path,
        )


def plot_self_alpha_lag1(
    all_results,
    label_map,
    cluster_idx,
    output_dir,
    figure_dpi,
    variants_to_plot=VARIANT_NAMES,
    batch_size=1,
    y_limits=(-1.2, 1.2),
):
    cluster_self_alpha_dir = output_dir / f"cluster_{cluster_idx}"
    cluster_self_alpha_dir.mkdir(parents=True, exist_ok=True)

    alpha_stats_by_variant = {}
    for variant_name in variants_to_plot:
        alpha_stats_by_variant[variant_name] = get_alpha_seq_all_sets_stats(
            all_results,
            variant_name,
            order_idx=1,
        )

    n_series = alpha_stats_by_variant[variants_to_plot[0]][0].shape[1]

    for batch_start in range(0, n_series, batch_size):
        batch_end = min(batch_start + batch_size, n_series)

        fig, axes = plt.subplots(len(variants_to_plot), 1, figsize=(12, 10), sharex=True)

        if len(variants_to_plot) == 1:
            axes = [axes]

        for ax, variant_name in zip(axes, variants_to_plot):
            labels = label_map[variant_name]
            alpha_mean_all, alpha_std_all, test_start = alpha_stats_by_variant[variant_name]
            x = np.arange(alpha_mean_all.shape[0])

            for series_idx in range(batch_start, batch_end):
                mean_trace = alpha_mean_all[:, series_idx, series_idx, 0]
                std_trace = alpha_std_all[:, series_idx, series_idx, 0]

                line, = ax.plot(
                    x,
                    mean_trace,
                    lw=1.5,
                    label=labels[series_idx],
                )

                ax.fill_between(
                    x,
                    mean_trace - std_trace,
                    mean_trace + std_trace,
                    color=line.get_color(),
                    alpha=0.08,
                )

            ax.axvline(test_start - 0.5, color="black", linestyle="--", lw=1.5)
            ax.axhline(0, color="black", linestyle="--", lw=1.5)
            ax.set_title(f"i=j, lag=1 ({variant_name}, mean +/- std across runs)")
            ax.set_ylabel("alpha")
            ax.set_ylim(*y_limits)
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)

        axes[-1].set_xlabel("Window index")
        fig.suptitle(f"Cluster {cluster_idx}: self-alpha lag 1", y=1.02)
        plt.tight_layout()

        save_path = cluster_self_alpha_dir / f"self_alpha_lag1_batch_{batch_start}_{batch_end - 1}.png"
        fig.savefig(save_path, dpi=figure_dpi, bbox_inches="tight")
        plt.close(fig)


def safe_file_part(value):
    return (
        str(value)
        .replace(" ", "_")
        .replace("*", "times")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(">", "to")
        .replace("<", "from")
    )


def save_figure(fig, save_path, figure_dpi):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


def select_event_beads(selected_ids, event_counts, max_abs_dy_by_particle, requested_bead_ids, cluster_idx):
    event_particle_order = np.argsort(-max_abs_dy_by_particle)
    quiet_particle_order = np.lexsort((max_abs_dy_by_particle, event_counts))
    beads_to_plot = [
        ("biggest-event bead", int(event_particle_order[0])),
        ("second-biggest-event bead", int(event_particle_order[1])),
        ("quietest bead", int(quiet_particle_order[0])),
        ("second-quietest bead", int(quiet_particle_order[1])),
    ]

    for requested_bead_id in requested_bead_ids:
        matches = np.where(selected_ids == requested_bead_id)[0]
        if len(matches) == 0:
            print(f"Requested bead {requested_bead_id} is not in cluster {cluster_idx}.")
            continue

        requested_particle_idx = int(matches[0])
        if requested_particle_idx not in [particle_idx for _, particle_idx in beads_to_plot]:
            beads_to_plot.append((f"requested bead {requested_bead_id}", requested_particle_idx))

    unique_beads = []
    seen = set()
    for bead_kind, particle_idx in beads_to_plot:
        if particle_idx not in seen:
            unique_beads.append((bead_kind, particle_idx))
            seen.add(particle_idx)

    return unique_beads


def get_event_plot_setup(cluster, frames, event_percentile, requested_bead_ids, cluster_idx):
    y_selected = cluster["y_selected"]
    selected_ids = np.asarray(cluster["selected_particle_ids"])
    n_particles = len(selected_ids)
    y_series_indices = np.arange(n_particles, 2 * n_particles)
    y_rel = y_selected - y_selected[0:1, :]
    dy = np.diff(y_selected, axis=0)
    frames_dy = frames[1:]
    abs_dy = np.abs(dy)
    event_threshold = np.percentile(abs_dy, event_percentile)
    event_mask = abs_dy >= event_threshold
    event_counts = event_mask.sum(axis=0)
    max_abs_dy_by_particle = abs_dy.max(axis=0)
    beads_to_plot = select_event_beads(
        selected_ids=selected_ids,
        event_counts=event_counts,
        max_abs_dy_by_particle=max_abs_dy_by_particle,
        requested_bead_ids=requested_bead_ids,
        cluster_idx=cluster_idx,
    )

    return {
        "y_selected": y_selected,
        "selected_ids": selected_ids,
        "n_particles": n_particles,
        "y_series_indices": y_series_indices,
        "y_rel": y_rel,
        "dy": dy,
        "frames_dy": frames_dy,
        "abs_dy": abs_dy,
        "event_threshold": event_threshold,
        "event_mask": event_mask,
        "beads_to_plot": beads_to_plot,
    }


def save_selected_event_time_series(cluster_idx, event_setup, frames, cluster_plot_dir, figure_dpi):
    y_selected = event_setup["y_selected"]
    y_rel = event_setup["y_rel"]
    dy = event_setup["dy"]
    frames_dy = event_setup["frames_dy"]
    event_mask = event_setup["event_mask"]
    event_threshold = event_setup["event_threshold"]
    selected_ids = event_setup["selected_ids"]
    beads_to_plot = event_setup["beads_to_plot"]

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)

    for bead_kind, particle_idx in beads_to_plot:
        bead_id = selected_ids[particle_idx]
        label = f"{bead_kind}: bead {bead_id}"
        line, = axes[0].plot(frames, y_selected[:, particle_idx], lw=1.5, label=label)
        color = line.get_color()
        axes[1].plot(frames, y_rel[:, particle_idx], lw=1.5, color=color, label=label)
        axes[2].plot(frames_dy, dy[:, particle_idx], lw=1.4, color=color, label=label)

        event_time_idx = np.where(event_mask[:, particle_idx])[0]
        axes[0].scatter(
            frames_dy[event_time_idx],
            y_selected[event_time_idx + 1, particle_idx],
            color=color,
            edgecolor="black",
            s=45,
            zorder=5,
        )
        axes[1].scatter(
            frames_dy[event_time_idx],
            y_rel[event_time_idx + 1, particle_idx],
            color=color,
            edgecolor="black",
            s=45,
            zorder=5,
        )
        axes[2].scatter(
            frames_dy[event_time_idx],
            dy[event_time_idx, particle_idx],
            color=color,
            edgecolor="black",
            s=45,
            zorder=5,
        )

    axes[0].set_title(f"Cluster {cluster_idx}: selected event/quiet bead raw y(t)")
    axes[0].set_ylabel("y")
    axes[1].set_title("relative y(t) - y(0)")
    axes[1].set_ylabel("y - y0")
    axes[2].set_title("dy with large |dy| events marked")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("dy")
    axes[2].axhline(event_threshold, color="black", linestyle="--", lw=1)
    axes[2].axhline(-event_threshold, color="black", linestyle="--", lw=1)

    for ax in axes:
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    save_figure(fig, cluster_plot_dir / "selected_event_time_series.png", figure_dpi)


def event_target_indices_for_variant(event_mask, particle_idx, variant_name):
    event_time_idx = np.where(event_mask[:, particle_idx])[0]
    if variant_base_name(variant_name) == "dxy":
        return event_time_idx
    return event_time_idx + 1


def event_windows_for_particle(event_mask, particle_idx, variant_name, target_indices):
    target_to_window = {int(target_idx): window_idx for window_idx, target_idx in enumerate(target_indices)}
    event_target_indices = event_target_indices_for_variant(event_mask, particle_idx, variant_name)
    return np.array(
        [
            target_to_window[int(target_idx)]
            for target_idx in event_target_indices
            if int(target_idx) in target_to_window
        ],
        dtype=int,
    )


def biggest_event_window_for_particle(event_setup, particle_idx, variant_name, target_indices):
    event_mask = event_setup["event_mask"]
    abs_dy = event_setup["abs_dy"]
    particle_event_times = np.where(event_mask[:, particle_idx])[0]
    if len(particle_event_times) == 0:
        return None

    biggest_local_time = particle_event_times[np.argmax(abs_dy[particle_event_times, particle_idx])]
    target_idx = biggest_local_time if variant_base_name(variant_name) == "dxy" else biggest_local_time + 1
    matches = np.where(target_indices == target_idx)[0]
    if len(matches) == 0:
        return None
    return int(matches[0])


def y_only_values(values, y_series_indices):
    return values[:, y_series_indices][:, :, y_series_indices, :]


def metric_plot_settings_y_only(metric_values, metric_name, y_series_indices):
    if metric_name == "Focuser":
        return "viridis", 0.0, 1.0

    y_values = y_only_values(metric_values, y_series_indices).reshape(-1)
    vmax = np.percentile(np.abs(y_values), 99)
    if vmax == 0:
        vmax = 1.0
    return "coolwarm", -vmax, vmax


def rows_for_selected_y_bead(particle_idx, labels, n_particles, y_series_indices):
    selected_y_idx = particle_idx + n_particles
    rows = []
    row_labels = []

    for source_idx in y_series_indices:
        rows.append((selected_y_idx, source_idx))
        row_labels.append(f"{labels[selected_y_idx]} <- {labels[source_idx]}")

    for target_idx in y_series_indices:
        if target_idx == selected_y_idx:
            continue
        rows.append((target_idx, selected_y_idx))
        row_labels.append(f"{labels[target_idx]} <- {labels[selected_y_idx]}")

    return rows, row_labels


def build_event_window_maps(
    cluster,
    cluster_idx,
    event_setup,
    event_plot_variants,
    frames,
    window_size,
    shuffle_seed,
):
    event_windows_by_variant = {}
    biggest_event_window_by_variant = {}
    split_boundaries_by_variant = {}

    for variant_name in event_plot_variants:
        target_indices, split_boundaries = window_target_indices_for_cluster_variant(
            cluster,
            variant_name,
            cluster_idx,
            frames,
            window_size,
            shuffle_seed=shuffle_seed,
        )
        split_boundaries_by_variant[variant_name] = split_boundaries
        event_windows_by_variant[variant_name] = {}
        biggest_event_window_by_variant[variant_name] = {}

        for _, particle_idx in event_setup["beads_to_plot"]:
            event_windows_by_variant[variant_name][particle_idx] = event_windows_for_particle(
                event_setup["event_mask"],
                particle_idx,
                variant_name,
                target_indices,
            )
            biggest_event_window_by_variant[variant_name][particle_idx] = biggest_event_window_for_particle(
                event_setup,
                particle_idx,
                variant_name,
                target_indices,
            )

    return event_windows_by_variant, biggest_event_window_by_variant, split_boundaries_by_variant


def save_average_y_heatmaps_comparison(
    metric_values_by_variant,
    metric_name,
    labels_by_variant,
    cluster_idx,
    n_particles,
    y_series_indices,
    split_name,
    cluster_plot_dir,
    figure_dpi,
):
    variant_names = list(metric_values_by_variant.keys())
    n_lags = metric_values_by_variant[variant_names[0]].shape[3]
    fig, axes = plt.subplots(
        len(variant_names),
        n_lags,
        figsize=(4.0 * n_lags, 3.6 * len(variant_names)),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(variant_names), n_lags)

    for row_idx, variant_name in enumerate(variant_names):
        metric_values = metric_values_by_variant[variant_name]
        metric_mean = metric_values.mean(axis=0)
        y_labels = labels_by_variant[variant_name][n_particles:]
        cmap, vmin, vmax = metric_plot_settings_y_only(metric_values, metric_name, y_series_indices)

        for lag_idx in range(n_lags):
            ax = axes[row_idx, lag_idx]
            matrix = metric_mean[y_series_indices[:, None], y_series_indices[None, :], lag_idx]
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"{variant_name}, lag {lag_idx + 1}")
            ax.set_xlabel("source y")
            ax.set_ylabel("target y")
            ax.set_xticks(range(n_particles))
            ax.set_yticks(range(n_particles))
            ax.set_xticklabels(y_labels, rotation=90, fontsize=7)
            ax.set_yticklabels(y_labels, fontsize=7)

        fig.colorbar(im, ax=axes[row_idx, :], shrink=0.85)

    fig.suptitle(f"Cluster {cluster_idx}: mean {metric_name} heatmaps ({split_name}, y only)")
    save_figure(fig, cluster_plot_dir / "average_heatmaps" / f"{safe_file_part(metric_name)}.png", figure_dpi)


def save_selected_bead_window_heatmaps(
    metric_values_by_variant,
    metric_name,
    particle_idx,
    bead_kind,
    event_setup,
    labels_by_variant,
    event_plot_variants,
    event_windows_by_variant,
    biggest_event_window_by_variant,
    split_boundaries_by_variant,
    cluster_idx,
    cluster_plot_dir,
    figure_dpi,
):
    n_particles = event_setup["n_particles"]
    y_series_indices = event_setup["y_series_indices"]
    selected_ids = event_setup["selected_ids"]
    n_lags = metric_values_by_variant[event_plot_variants[0]].shape[3]
    bead_id = selected_ids[particle_idx]
    fig, axes = plt.subplots(
        len(event_plot_variants),
        n_lags,
        figsize=(4.2 * n_lags, 5.8 * len(event_plot_variants)),
        sharey=False,
    )
    axes = np.asarray(axes).reshape(len(event_plot_variants), n_lags)

    for row_idx, variant_name in enumerate(event_plot_variants):
        metric_values = metric_values_by_variant[variant_name]
        labels = labels_by_variant[variant_name]
        rows, row_labels = rows_for_selected_y_bead(particle_idx, labels, n_particles, y_series_indices)
        event_windows = event_windows_by_variant[variant_name][particle_idx]
        biggest_event_window = biggest_event_window_by_variant[variant_name][particle_idx]
        split_boundaries = split_boundaries_by_variant[variant_name]
        cmap, vmin, vmax = metric_plot_settings_y_only(metric_values, metric_name, y_series_indices)

        for lag_idx in range(n_lags):
            ax = axes[row_idx, lag_idx]
            matrix = np.array(
                [
                    metric_values[:, target_idx, source_idx, lag_idx]
                    for target_idx, source_idx in rows
                ]
            )
            im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{variant_name}, lag {lag_idx + 1}")
            ax.set_xlabel("Window index")

            for boundary in split_boundaries:
                ax.axvline(boundary - 0.5, color="black", linestyle="--", lw=1)
            for event_window in event_windows:
                ax.axvline(event_window, color="red", alpha=0.25, lw=1)
            if biggest_event_window is not None:
                ax.axvline(biggest_event_window, color="red", linestyle="--", lw=1.2, alpha=0.8)

        axes[row_idx, 0].set_yticks(np.arange(len(row_labels)))
        axes[row_idx, 0].set_yticklabels(row_labels, fontsize=6)
        fig.colorbar(im, ax=axes[row_idx, :], shrink=0.85)

    fig.suptitle(
        f"Cluster {cluster_idx}, {bead_kind} {bead_id}: {metric_name} over all windows (y only)",
        y=1.0,
    )
    save_figure(
        fig,
        cluster_plot_dir
        / "selected_bead_relations"
        / f"{safe_file_part(bead_kind)}_bead_{bead_id}"
        / f"{safe_file_part(metric_name)}.png",
        figure_dpi,
    )


def save_self_bead_window_heatmaps(
    metric_values_by_variant,
    metric_name,
    event_setup,
    event_plot_variants,
    event_windows_by_variant,
    biggest_event_window_by_variant,
    split_boundaries_by_variant,
    cluster_idx,
    cluster_plot_dir,
    figure_dpi,
):
    n_particles = event_setup["n_particles"]
    selected_ids = event_setup["selected_ids"]
    beads_to_plot = event_setup["beads_to_plot"]
    n_lags = metric_values_by_variant[event_plot_variants[0]].shape[3]
    row_info = []
    row_labels = []

    for _, particle_idx in beads_to_plot:
        selected_y_idx = particle_idx + n_particles
        bead_id = selected_ids[particle_idx]
        for lag_idx in range(n_lags):
            row_info.append((particle_idx, selected_y_idx, lag_idx))
            row_labels.append(f"{bead_id}, lag {lag_idx + 1}")

    fig, axes = plt.subplots(
        len(event_plot_variants),
        1,
        figsize=(14, max(4.0, 0.35 * len(row_labels)) * len(event_plot_variants)),
        sharex=False,
    )
    axes = np.atleast_1d(axes)

    for ax, variant_name in zip(axes, event_plot_variants):
        metric_values = metric_values_by_variant[variant_name]
        cmap, vmin, vmax = metric_plot_settings_y_only(
            metric_values,
            metric_name,
            event_setup["y_series_indices"],
        )
        matrix = np.array(
            [
                metric_values[:, selected_y_idx, selected_y_idx, lag_idx]
                for _, selected_y_idx, lag_idx in row_info
            ]
        )
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{variant_name}: self y -> y")
        ax.set_xlabel("Window index")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)

        for boundary in split_boundaries_by_variant[variant_name]:
            ax.axvline(boundary - 0.5, color="black", linestyle="--", lw=1)

        for row_idx, (particle_idx, _, _) in enumerate(row_info):
            event_windows = event_windows_by_variant[variant_name][particle_idx]
            biggest_event_window = biggest_event_window_by_variant[variant_name][particle_idx]
            if len(event_windows) > 0:
                ax.scatter(event_windows, np.full(len(event_windows), row_idx), color="red", s=8, alpha=0.35, zorder=5)
            if biggest_event_window is not None:
                ax.scatter([biggest_event_window], [row_idx], color="red", s=35, marker="x", zorder=6)

        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"Cluster {cluster_idx}: selected beads self {metric_name} over all windows", y=1.0)
    plt.tight_layout()
    save_figure(fig, cluster_plot_dir / "self_bead_windows" / f"{safe_file_part(metric_name)}.png", figure_dpi)


def save_self_lag1_line_plots(
    metric_values_by_variant,
    metric_name,
    event_setup,
    event_plot_variants,
    event_windows_by_variant,
    biggest_event_window_by_variant,
    split_boundaries_by_variant,
    cluster_idx,
    cluster_plot_dir,
    figure_dpi,
    min_alpha_limit=0.3,
    min_alpha_q_limit=0.3,
):
    lag_idx = 0
    n_particles = event_setup["n_particles"]
    selected_ids = event_setup["selected_ids"]
    beads_to_plot = event_setup["beads_to_plot"]
    fig, axes = plt.subplots(len(event_plot_variants), 1, figsize=(13, 4.2 * len(event_plot_variants)), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, variant_name in zip(axes, event_plot_variants):
        metric_values = metric_values_by_variant[variant_name]
        x = np.arange(metric_values.shape[0])
        traces = []

        for bead_kind, particle_idx in beads_to_plot:
            selected_y_idx = particle_idx + n_particles
            bead_id = selected_ids[particle_idx]
            trace = metric_values[:, selected_y_idx, selected_y_idx, lag_idx]
            traces.append(trace)
            line, = ax.plot(x, trace, lw=1.5, label=f"{bead_kind}: bead {bead_id}")

            event_windows = event_windows_by_variant[variant_name][particle_idx]
            if len(event_windows) > 0:
                ax.scatter(event_windows, trace[event_windows], color=line.get_color(), edgecolor="black", s=35, zorder=5)

            biggest_event_window = biggest_event_window_by_variant[variant_name][particle_idx]
            if biggest_event_window is not None:
                ax.scatter([biggest_event_window], [trace[biggest_event_window]], color="red", marker="x", s=70, zorder=6)

        for boundary in split_boundaries_by_variant[variant_name]:
            ax.axvline(boundary - 0.5, color="black", linestyle="--", lw=1)

        ax.axhline(0, color="black", linestyle="--", lw=0.8)
        if metric_name == "Focuser":
            ax.set_ylim(0.0, 1.0)
        elif metric_name == "alpha":
            ax.set_ylim(*symmetric_metric_limits(np.concatenate(traces), min_alpha_limit))
        elif metric_name == "alpha * Q":
            ax.set_ylim(*symmetric_metric_limits(np.concatenate(traces), min_alpha_q_limit))
        ax.set_title(f"{variant_name}: self y -> y, lag 1")
        ax.set_xlabel("Window index")
        ax.set_ylabel(metric_name)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)

    fig.suptitle(f"Cluster {cluster_idx}: self lag-1 {metric_name} traces", y=1.02)
    plt.tight_layout()
    save_figure(fig, cluster_plot_dir / "self_lag1_lines" / f"{safe_file_part(metric_name)}.png", figure_dpi)


def save_focal_lag1_relation_line_plots(
    metric_values_by_variant,
    metric_name,
    event_setup,
    event_plot_variants,
    split_boundaries_by_variant,
    cluster_idx,
    cluster_plot_dir,
    figure_dpi,
    min_alpha_limit=0.3,
    min_alpha_q_limit=0.3,
):
    lag_idx = 0
    n_particles = event_setup["n_particles"]
    selected_ids = event_setup["selected_ids"]
    focal_particles = [particle_idx for _, particle_idx in event_setup["beads_to_plot"]]
    fig, axes = plt.subplots(len(event_plot_variants), 1, figsize=(14, 5.0 * len(event_plot_variants)), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, variant_name in zip(axes, event_plot_variants):
        metric_values = metric_values_by_variant[variant_name]
        x = np.arange(metric_values.shape[0])
        traces = []

        for target_particle_idx in focal_particles:
            for source_particle_idx in focal_particles:
                target_idx = target_particle_idx + n_particles
                source_idx = source_particle_idx + n_particles
                trace = metric_values[:, target_idx, source_idx, lag_idx]
                traces.append(trace)
                label = f"{selected_ids[source_particle_idx]} -> {selected_ids[target_particle_idx]}"
                ax.plot(x, trace, lw=1.2, label=label)

        for boundary in split_boundaries_by_variant[variant_name]:
            ax.axvline(boundary - 0.5, color="black", linestyle="--", lw=1)

        ax.axhline(0, color="black", linestyle="--", lw=0.8)
        if metric_name == "Focuser":
            ax.set_ylim(0.0, 1.0)
        elif metric_name == "alpha":
            ax.set_ylim(*symmetric_metric_limits(np.concatenate(traces), min_alpha_limit))
        elif metric_name == "alpha * Q":
            ax.set_ylim(*symmetric_metric_limits(np.concatenate(traces), min_alpha_q_limit))
        ax.set_title(f"{variant_name}: focal bead y -> y relations, lag 1")
        ax.set_xlabel("Window index")
        ax.set_ylabel(metric_name)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, ncol=2)

    fig.suptitle(f"Cluster {cluster_idx}: focal lag-1 {metric_name} relations", y=1.02)
    plt.tight_layout()
    save_figure(fig, cluster_plot_dir / "self_lag1_lines" / f"focal_relations_{safe_file_part(metric_name)}.png", figure_dpi)


def save_event_interpretability_plots_for_cluster(
    cluster_idx,
    cluster,
    all_results,
    label_map,
    frames,
    save_dir,
    window_size,
    figure_dpi,
    event_plot_variants=("rel_xy", "dxy"),
    average_heatmap_variants=("raw_xy", "raw_xy_shuffled", "rel_xy", "rel_xy_shuffled", "dxy", "dxy_shuffled"),
    metrics=("Focuser", "alpha", "alpha * Q"),
    event_percentile=99.6,
    order_idx=1,
    average_heatmap_split="test",
    all_window_splits=("train", "val", "test"),
    requested_bead_ids=(),
    shuffle_seed=SHUFFLE_CONTROL_SEED,
):
    event_plot_variants = [variant_name for variant_name in event_plot_variants if variant_name in all_results]
    average_heatmap_variants = [variant_name for variant_name in average_heatmap_variants if variant_name in all_results]
    if len(event_plot_variants) == 0:
        print(f"Cluster {cluster_idx}: no event plot variants available.")
        return

    cluster_plot_dir = save_dir / f"cluster_{cluster_idx}"
    cluster_plot_dir.mkdir(parents=True, exist_ok=True)
    event_setup = get_event_plot_setup(
        cluster=cluster,
        frames=frames,
        event_percentile=event_percentile,
        requested_bead_ids=requested_bead_ids,
        cluster_idx=cluster_idx,
    )
    save_selected_event_time_series(cluster_idx, event_setup, frames, cluster_plot_dir, figure_dpi)

    all_plot_variants = list(dict.fromkeys(average_heatmap_variants + event_plot_variants))
    labels_by_variant = {variant_name: label_map[variant_name] for variant_name in all_plot_variants}
    average_metrics_by_variant = {
        variant_name: mean_display_metrics_across_runs(
            all_results,
            variant_name,
            split_names=[average_heatmap_split],
            order_idx=order_idx,
        )
        for variant_name in average_heatmap_variants
    }
    all_window_metrics_by_variant = {
        variant_name: mean_display_metrics_across_runs(
            all_results,
            variant_name,
            split_names=all_window_splits,
            order_idx=order_idx,
        )
        for variant_name in event_plot_variants
    }
    event_windows_by_variant, biggest_event_window_by_variant, split_boundaries_by_variant = build_event_window_maps(
        cluster=cluster,
        cluster_idx=cluster_idx,
        event_setup=event_setup,
        event_plot_variants=event_plot_variants,
        frames=frames,
        window_size=window_size,
        shuffle_seed=shuffle_seed,
    )

    for metric_name in metrics:
        if len(average_metrics_by_variant) > 0:
            metric_values_by_variant = {
                variant_name: average_metrics_by_variant[variant_name][metric_name]
                for variant_name in average_heatmap_variants
            }
            save_average_y_heatmaps_comparison(
                metric_values_by_variant=metric_values_by_variant,
                metric_name=metric_name,
                labels_by_variant=labels_by_variant,
                cluster_idx=cluster_idx,
                n_particles=event_setup["n_particles"],
                y_series_indices=event_setup["y_series_indices"],
                split_name=average_heatmap_split,
                cluster_plot_dir=cluster_plot_dir,
                figure_dpi=figure_dpi,
            )

        metric_values_by_variant = {
            variant_name: all_window_metrics_by_variant[variant_name][metric_name]
            for variant_name in event_plot_variants
        }
        save_self_bead_window_heatmaps(
            metric_values_by_variant=metric_values_by_variant,
            metric_name=metric_name,
            event_setup=event_setup,
            event_plot_variants=event_plot_variants,
            event_windows_by_variant=event_windows_by_variant,
            biggest_event_window_by_variant=biggest_event_window_by_variant,
            split_boundaries_by_variant=split_boundaries_by_variant,
            cluster_idx=cluster_idx,
            cluster_plot_dir=cluster_plot_dir,
            figure_dpi=figure_dpi,
        )
        save_self_lag1_line_plots(
            metric_values_by_variant=metric_values_by_variant,
            metric_name=metric_name,
            event_setup=event_setup,
            event_plot_variants=event_plot_variants,
            event_windows_by_variant=event_windows_by_variant,
            biggest_event_window_by_variant=biggest_event_window_by_variant,
            split_boundaries_by_variant=split_boundaries_by_variant,
            cluster_idx=cluster_idx,
            cluster_plot_dir=cluster_plot_dir,
            figure_dpi=figure_dpi,
        )
        save_focal_lag1_relation_line_plots(
            metric_values_by_variant=metric_values_by_variant,
            metric_name=metric_name,
            event_setup=event_setup,
            event_plot_variants=event_plot_variants,
            split_boundaries_by_variant=split_boundaries_by_variant,
            cluster_idx=cluster_idx,
            cluster_plot_dir=cluster_plot_dir,
            figure_dpi=figure_dpi,
        )

        for bead_kind, particle_idx in event_setup["beads_to_plot"]:
            save_selected_bead_window_heatmaps(
                metric_values_by_variant=metric_values_by_variant,
                metric_name=metric_name,
                particle_idx=particle_idx,
                bead_kind=bead_kind,
                event_setup=event_setup,
                labels_by_variant=labels_by_variant,
                event_plot_variants=event_plot_variants,
                event_windows_by_variant=event_windows_by_variant,
                biggest_event_window_by_variant=biggest_event_window_by_variant,
                split_boundaries_by_variant=split_boundaries_by_variant,
                cluster_idx=cluster_idx,
                cluster_plot_dir=cluster_plot_dir,
                figure_dpi=figure_dpi,
            )


def event_y_for_variant(cluster, variant_name, cluster_idx, shuffle_seed):
    y_event_source = cluster["y_selected"]
    if variant_is_shuffled(variant_name):
        rng = np.random.default_rng(shuffle_seed + cluster_idx)
        _ = shuffle_columns(cluster["x_selected"], rng)
        y_event_source = shuffle_columns(cluster["y_selected"], rng)
    return y_event_source


def focal_beads_by_movement(cluster):
    dy = np.diff(cluster["y_selected"], axis=0)
    max_abs_dy_by_particle = np.abs(dy).max(axis=0)
    biggest_order = np.argsort(-max_abs_dy_by_particle)
    smallest_order = np.argsort(max_abs_dy_by_particle)
    focal_beads = [
        ("biggest_event", int(biggest_order[0])),
        ("second_biggest_event", int(biggest_order[1])),
        ("smallest_movement", int(smallest_order[0])),
        ("second_smallest_movement", int(smallest_order[1])),
    ]

    unique_focal_beads = []
    seen = set()
    for focal_kind, particle_idx in focal_beads:
        if particle_idx not in seen:
            unique_focal_beads.append((focal_kind, particle_idx))
            seen.add(particle_idx)

    return unique_focal_beads, max_abs_dy_by_particle


def target_index_from_dy_time(event_time_idx, variant_name):
    if variant_base_name(variant_name) == "dxy":
        return event_time_idx
    return event_time_idx + 1


def largest_valid_event_window_for_bead(cluster, variant_name, cluster_idx, particle_idx, target_indices, shuffle_seed):
    y_event_source = event_y_for_variant(cluster, variant_name, cluster_idx, shuffle_seed)
    dy = np.diff(y_event_source, axis=0)
    abs_dy = np.abs(dy)
    target_to_window = {int(target_idx): window_idx for window_idx, target_idx in enumerate(target_indices)}

    for event_time_idx in np.argsort(-abs_dy[:, particle_idx]):
        event_time_idx = int(event_time_idx)
        target_idx = target_index_from_dy_time(event_time_idx, variant_name)
        if int(target_idx) in target_to_window:
            return (
                int(target_to_window[int(target_idx)]),
                event_time_idx,
                dy[event_time_idx, particle_idx],
                abs_dy[event_time_idx, particle_idx],
            )

    event_time_idx = int(np.argmax(abs_dy[:, particle_idx]))
    return None, event_time_idx, dy[event_time_idx, particle_idx], abs_dy[event_time_idx, particle_idx]


def large_event_windows_for_variant(cluster, variant_name, cluster_idx, target_indices, percentile, shuffle_seed):
    y_event_source = event_y_for_variant(cluster, variant_name, cluster_idx, shuffle_seed)
    abs_dy = np.abs(np.diff(y_event_source, axis=0))
    threshold = np.percentile(abs_dy, percentile)
    event_time_idx, _ = np.where(abs_dy >= threshold)
    target_to_window = {int(target_idx): window_idx for window_idx, target_idx in enumerate(target_indices)}
    event_windows = []

    for time_idx in event_time_idx:
        target_idx = target_index_from_dy_time(int(time_idx), variant_name)
        if int(target_idx) in target_to_window:
            event_windows.append(target_to_window[int(target_idx)])

    return np.array(sorted(set(event_windows)), dtype=int), threshold


def window_context(center_window, n_windows, radius):
    if center_window is None:
        return np.array([], dtype=int)

    start = max(0, center_window - radius)
    end = min(n_windows, center_window + radius + 1)
    return np.arange(start, end, dtype=int)


def quiet_windows_excluding_events(n_windows, event_windows, exclusion_radius):
    quiet_mask = np.ones(n_windows, dtype=bool)
    for event_window in event_windows:
        start = max(0, event_window - exclusion_radius)
        end = min(n_windows, event_window + exclusion_radius + 1)
        quiet_mask[start:end] = False
    return np.where(quiet_mask)[0]


def self_lag1_values(metric_values, windows, particle_idx, n_particles):
    if len(windows) == 0:
        return np.array([])

    y_idx = particle_idx + n_particles
    return metric_values[windows, y_idx, y_idx, 0].reshape(-1)


def event_quiet_stats(event_values, quiet_values):
    if len(event_values) == 0 or len(quiet_values) == 0:
        return None

    event_abs_mean = np.mean(np.abs(event_values))
    quiet_abs_mean = np.mean(np.abs(quiet_values))
    return {
        "event_mean": np.mean(event_values),
        "quiet_mean": np.mean(quiet_values),
        "event_abs_mean": event_abs_mean,
        "quiet_abs_mean": quiet_abs_mean,
        "event_median": np.median(event_values),
        "quiet_median": np.median(quiet_values),
        "event_abs_median": np.median(np.abs(event_values)),
        "quiet_abs_median": np.median(np.abs(quiet_values)),
        "abs_mean_difference": event_abs_mean - quiet_abs_mean,
        "abs_mean_ratio": event_abs_mean / quiet_abs_mean if quiet_abs_mean > 0 else np.nan,
    }


def build_cluster_event_summary_rows(
    cluster_idx,
    cluster,
    all_results,
    variants,
    metrics,
    split_names,
    frames,
    window_size,
    percentile,
    event_context_radius,
    quiet_exclusion_radius,
    metadata=None,
    order_idx=1,
    shuffle_seed=SHUFFLE_CONTROL_SEED,
):
    selected_ids = np.asarray(cluster["selected_particle_ids"])
    n_particles = len(selected_ids)
    focal_beads, original_max_abs_dy_by_particle = focal_beads_by_movement(cluster)
    rows = []

    for variant_name in variants:
        if variant_name not in all_results:
            continue

        metrics_by_name = mean_display_metrics_across_runs(
            all_results,
            variant_name,
            split_names=split_names,
            order_idx=order_idx,
        )
        target_indices, _ = window_target_indices_for_cluster_variant(
            cluster,
            variant_name,
            cluster_idx,
            frames,
            window_size,
            shuffle_seed=shuffle_seed,
        )
        n_windows = metrics_by_name["alpha"].shape[0]
        large_event_windows, event_threshold = large_event_windows_for_variant(
            cluster,
            variant_name,
            cluster_idx,
            target_indices,
            percentile=percentile,
            shuffle_seed=shuffle_seed,
        )
        base_quiet_windows = quiet_windows_excluding_events(
            n_windows,
            large_event_windows,
            exclusion_radius=quiet_exclusion_radius,
        )

        for focal_kind, particle_idx in focal_beads:
            biggest_event_window, biggest_event_time_idx, biggest_dy, biggest_abs_dy = largest_valid_event_window_for_bead(
                cluster,
                variant_name,
                cluster_idx,
                particle_idx,
                target_indices,
                shuffle_seed=shuffle_seed,
            )
            event_windows = window_context(biggest_event_window, n_windows, event_context_radius)
            quiet_windows = np.setdiff1d(base_quiet_windows, event_windows)
            if len(event_windows) == 0:
                continue

            base_info = {
                "cluster_idx": cluster_idx,
                "variant": variant_name,
                "base_variant": variant_base_name(variant_name),
                "is_shuffled_control": variant_is_shuffled(variant_name),
                "event_source": "shuffled_y" if variant_is_shuffled(variant_name) else "original_y",
                "event_percentile": percentile,
                "large_event_threshold_abs_dy": event_threshold,
                "event_context_radius": event_context_radius,
                "quiet_exclusion_radius": quiet_exclusion_radius,
                "focal_kind": focal_kind,
                "event_bead_id": int(selected_ids[particle_idx]),
                "event_particle_idx": int(particle_idx),
                "biggest_event_window": biggest_event_window,
                "biggest_event_time_idx": int(biggest_event_time_idx),
                "biggest_event_dy": biggest_dy,
                "biggest_event_abs_dy": biggest_abs_dy,
                "event_window_start": int(event_windows[0]),
                "event_window_end": int(event_windows[-1]),
                "relation_group": "self_y_to_y",
                "lag": 1,
            }
            if metadata is not None:
                base_info.update(metadata)

            for metric_name in metrics:
                metric_values = metrics_by_name[metric_name]
                event_values = self_lag1_values(metric_values, event_windows, particle_idx, n_particles)
                quiet_values = self_lag1_values(metric_values, quiet_windows, particle_idx, n_particles)
                stats = event_quiet_stats(event_values, quiet_values)
                if stats is None:
                    continue

                rows.append(
                    {
                        **base_info,
                        "metric": metric_name,
                        "n_event_windows": len(event_windows),
                        "n_quiet_windows": len(quiet_windows),
                        "n_event_values": len(event_values),
                        "n_quiet_values": len(quiet_values),
                        **stats,
                    }
                )

    return rows


def save_cluster_event_summary(rows, save_dir, cluster_idx):
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"cluster_{cluster_idx}_event_vs_quiet_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_event_summary_outputs(rows, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_path = save_dir / "event_vs_quiet_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved event-vs-quiet summary: {summary_path} ({len(summary_df)} rows)")
    return summary_df, summary_path


def symmetric_metric_limits(values, minimum_vmax):
    vmax = np.max(np.abs(values))
    vmax = max(vmax, minimum_vmax)
    return -1.10 * vmax, 1.10 * vmax


def save_raw_line_plots_for_cluster(
    cluster_idx,
    cluster,
    all_results,
    label_map,
    frames,
    output_dir,
    window_size,
    figure_dpi,
    variants=("rel_xy", "dxy"),
    metrics=("Focuser", "alpha", "alpha * Q"),
    split_names=("train", "val", "test"),
    event_percentile=99.6,
    order_idx=1,
    min_alpha_limit=0.3,
    min_alpha_q_limit=0.3,
    shuffle_seed=SHUFFLE_CONTROL_SEED,
):
    x_selected = cluster["x_selected"]
    y_selected = cluster["y_selected"]
    selected_ids = np.asarray(cluster["selected_particle_ids"])
    n_particles = len(selected_ids)
    y_series_indices = np.arange(n_particles, 2 * n_particles)
    dy = np.diff(y_selected, axis=0)
    abs_dy = np.abs(dy)
    event_threshold = np.percentile(abs_dy, event_percentile)
    event_mask = abs_dy >= event_threshold
    event_counts = event_mask.sum(axis=0)
    max_abs_dy_by_particle = abs_dy.max(axis=0)
    event_particle_order = np.argsort(-max_abs_dy_by_particle)
    small_movement_order = np.argsort(max_abs_dy_by_particle)
    focal_beads = [
        ("biggest_event", int(event_particle_order[0])),
        ("second_biggest_event", int(event_particle_order[1])),
        ("smallest_movement", int(small_movement_order[0])),
        ("second_smallest_movement", int(small_movement_order[1])),
    ]

    unique_focal_beads = []
    seen = set()
    for focal_kind, particle_idx in focal_beads:
        if particle_idx not in seen:
            unique_focal_beads.append((focal_kind, particle_idx))
            seen.add(particle_idx)
    focal_beads = unique_focal_beads

    position0 = np.column_stack([x_selected[0], y_selected[0]])
    base_raw_plot_dir = output_dir / f"cluster_{cluster_idx}"
    base_raw_plot_dir.mkdir(parents=True, exist_ok=True)

    def closeness_rank_info(focal_particle_idx, other_particle_idx):
        distances = np.linalg.norm(position0 - position0[focal_particle_idx], axis=1)
        order = np.argsort(distances)
        ranks = {int(particle_idx): rank for rank, particle_idx in enumerate(order)}
        return ranks[int(other_particle_idx)], distances[other_particle_idx]

    def relations_touching_focal_bead(focal_particle_idx):
        focal_y_idx = focal_particle_idx + n_particles
        focal_bead_id = selected_ids[focal_particle_idx]
        relations = []

        for source_particle_idx, source_idx in enumerate(y_series_indices):
            rank, distance = closeness_rank_info(focal_particle_idx, source_particle_idx)
            relations.append(
                {
                    "direction": "incoming",
                    "target_idx": focal_y_idx,
                    "source_idx": source_idx,
                    "target_bead_id": focal_bead_id,
                    "source_bead_id": selected_ids[source_particle_idx],
                    "other_rank": rank,
                    "other_distance": distance,
                }
            )

        for target_particle_idx, target_idx in enumerate(y_series_indices):
            if target_particle_idx == focal_particle_idx:
                continue

            rank, distance = closeness_rank_info(focal_particle_idx, target_particle_idx)
            relations.append(
                {
                    "direction": "outgoing",
                    "target_idx": target_idx,
                    "source_idx": focal_y_idx,
                    "target_bead_id": selected_ids[target_particle_idx],
                    "source_bead_id": focal_bead_id,
                    "other_rank": rank,
                    "other_distance": distance,
                }
            )

        return relations

    def focal_event_windows(focal_particle_idx, variant_name, target_indices):
        event_time_idx = np.where(event_mask[:, focal_particle_idx])[0]
        event_target_indices = event_time_idx if variant_base_name(variant_name) == "dxy" else event_time_idx + 1
        target_to_window = {int(target_idx): window_idx for window_idx, target_idx in enumerate(target_indices)}
        return np.array(
            [
                target_to_window[int(target_idx)]
                for target_idx in event_target_indices
                if int(target_idx) in target_to_window
            ],
            dtype=int,
        )

    def focal_biggest_event_window(focal_particle_idx, variant_name, target_indices):
        event_time_idx = np.where(event_mask[:, focal_particle_idx])[0]
        if len(event_time_idx) == 0:
            return None

        biggest_local_time = event_time_idx[np.argmax(abs_dy[event_time_idx, focal_particle_idx])]
        target_idx = biggest_local_time if variant_base_name(variant_name) == "dxy" else biggest_local_time + 1
        matches = np.where(target_indices == target_idx)[0]
        if len(matches) == 0:
            return None
        return int(matches[0])

    def y_limits_for_relation_group(metric_values, metric_name, relations, variant_name, global_alpha_q_limits):
        if metric_name == "Focuser":
            return 0.0, 1.0
        if metric_name == "alpha * Q":
            return global_alpha_q_limits[variant_name]

        values = np.concatenate(
            [
                metric_values[:, relation["target_idx"], relation["source_idx"], :].reshape(-1)
                for relation in relations
            ]
        )
        if metric_name == "alpha":
            return symmetric_metric_limits(values, min_alpha_limit)

        low, high = np.percentile(values, [1, 99])
        pad = 0.05 * (high - low)
        if pad == 0:
            pad = 1.0
        return low - pad, high + pad

    def save_raw_relation_plot(
        metric_values,
        metric_name,
        variant_name,
        relation,
        focal_kind,
        focal_particle_idx,
        event_windows,
        biggest_event_window,
        split_boundaries,
        labels,
        raw_plot_dir,
        y_limits,
    ):
        target_idx = relation["target_idx"]
        source_idx = relation["source_idx"]
        target_bead_id = relation["target_bead_id"]
        source_bead_id = relation["source_bead_id"]
        direction = relation["direction"]
        other_rank = relation["other_rank"]
        other_distance = relation["other_distance"]
        x = np.arange(metric_values.shape[0])
        n_lags = metric_values.shape[3]
        fig, ax = plt.subplots(figsize=(12, 4.5))

        for lag_idx in range(n_lags):
            trace = metric_values[:, target_idx, source_idx, lag_idx]
            ax.plot(x, trace, lw=1.4, label=f"lag {lag_idx + 1}")

        for boundary in split_boundaries:
            ax.axvline(boundary - 0.5, color="black", linestyle="--", lw=1)
        for event_window in event_windows:
            ax.axvline(event_window, color="red", alpha=0.20, lw=1)
        if biggest_event_window is not None:
            ax.axvline(biggest_event_window, color="red", linestyle="--", lw=1.2, alpha=0.8)

        ax.axhline(0, color="black", linestyle="--", lw=0.8)
        ax.set_ylim(*y_limits)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax.set_xlabel("Window index")
        ax.set_ylabel(metric_name)
        ax.set_title(
            f"{variant_name} {metric_name}: {labels[source_idx]} -> {labels[target_idx]}\n"
            f"focal {focal_kind} bead {selected_ids[focal_particle_idx]}, "
            f"other bead rank {other_rank} by distance (distance={other_distance:.3f})"
        )
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)

        metric_dir = raw_plot_dir / variant_name / safe_file_part(metric_name) / direction
        metric_dir.mkdir(parents=True, exist_ok=True)
        save_path = metric_dir / f"rank_{other_rank:02d}_source_{source_bead_id}_target_{target_bead_id}.png"
        save_figure(fig, save_path, figure_dpi)

    variants = [variant_name for variant_name in variants if variant_name in all_results]
    metrics_by_variant = {
        variant_name: mean_display_metrics_across_runs(
            all_results,
            variant_name,
            split_names=split_names,
            order_idx=order_idx,
        )
        for variant_name in variants
    }
    global_alpha_q_limits = {}
    for variant_name in variants:
        values = metrics_by_variant[variant_name]["alpha * Q"]
        y_values = values[:, y_series_indices][:, :, y_series_indices, :].reshape(-1)
        global_alpha_q_limits[variant_name] = symmetric_metric_limits(y_values, min_alpha_q_limit)

    window_info_by_variant = {}
    for variant_name in variants:
        target_indices, split_boundaries = window_target_indices_for_cluster_variant(
            cluster,
            variant_name,
            cluster_idx,
            frames,
            window_size,
            shuffle_seed=shuffle_seed,
        )
        window_info_by_variant[variant_name] = {
            "target_indices": target_indices,
            "split_boundaries": split_boundaries,
        }

    saved_count = 0
    for focal_kind, focal_particle_idx in focal_beads:
        focal_bead_id = selected_ids[focal_particle_idx]
        raw_plot_dir = base_raw_plot_dir / f"{focal_kind}_bead_{focal_bead_id}"
        raw_plot_dir.mkdir(parents=True, exist_ok=True)
        relations = relations_touching_focal_bead(focal_particle_idx)
        for variant_name in variants:
            labels = label_map[variant_name]
            target_indices = window_info_by_variant[variant_name]["target_indices"]
            split_boundaries = window_info_by_variant[variant_name]["split_boundaries"]
            event_windows = focal_event_windows(focal_particle_idx, variant_name, target_indices)
            biggest_event_window = focal_biggest_event_window(focal_particle_idx, variant_name, target_indices)

            for metric_name in metrics:
                metric_values = metrics_by_variant[variant_name][metric_name]
                y_limits = y_limits_for_relation_group(
                    metric_values,
                    metric_name,
                    relations,
                    variant_name,
                    global_alpha_q_limits,
                )

                for relation in relations:
                    save_raw_relation_plot(
                        metric_values=metric_values,
                        metric_name=metric_name,
                        variant_name=variant_name,
                        relation=relation,
                        focal_kind=focal_kind,
                        focal_particle_idx=focal_particle_idx,
                        event_windows=event_windows,
                        biggest_event_window=biggest_event_window,
                        split_boundaries=split_boundaries,
                        labels=labels,
                        raw_plot_dir=raw_plot_dir,
                        y_limits=y_limits,
                    )
                    saved_count += 1

    print(f"Saved {saved_count} raw line plots for cluster {cluster_idx}.")


def train_clusters_for_echo(
    clusters,
    frames,
    result_dirs,
    device,
    seed,
    n_runs,
    window_size,
    temperature,
    order,
    epochs,
    figure_dpi,
    cluster_limit=None,
    variant_names=None,
    alpha_heatmap_threshold=0.1,
    metadata=None,
    include_shuffled=True,
    shuffle_seed=SHUFFLE_CONTROL_SEED,
    event_percentile=99.6,
    event_context_radius=2,
    quiet_exclusion_radius=2,
    requested_bead_ids=(109,),
):
    if variant_names is None:
        variant_names = make_variant_names(include_shuffled=include_shuffled)

    original_variant_names = [name for name in variant_names if not variant_is_shuffled(name)]
    training_rows = []
    event_summary_rows = []

    clusters_to_run = clusters
    if cluster_limit is not None:
        clusters_to_run = clusters[:cluster_limit]

    for cluster_idx, cluster in enumerate(clusters_to_run):
        x_selected = cluster["x_selected"]
        y_selected = cluster["y_selected"]
        selected_ids = cluster["selected_particle_ids"]
        variants = build_training_variants(
            x_selected,
            y_selected,
            frames,
            include_shuffled=include_shuffled,
            shuffle_seed=shuffle_seed + cluster_idx,
        )
        variants = {variant_name: variants[variant_name] for variant_name in variant_names}
        raw_time_series = variants["raw_xy"][0].detach().cpu()
        label_map = make_label_map(selected_ids)
        all_results = {}

        for variant_name, (time_series, frames_used) in variants.items():
            base_name = variant_base_name(variant_name)
            is_shuffled = variant_is_shuffled(variant_name)
            print("=" * 80)
            print(f"Cluster {cluster_idx}: running variant {variant_name}")
            print("time_series shape [N, T]:", tuple(time_series.shape))

            _, _, test_series = split_time_series(
                time_series,
                train_ratio=0.8,
                val_ratio=0.1,
                window_size=window_size,
            )
            test_windows = create_windowed_dataset(test_series, window_size)
            y_true = test_windows[:, :, -1]
            y_pred_baseline = test_windows[:, :, -2]
            baseline_mse = torch.mean((y_true - y_pred_baseline) ** 2).item()

            result = run_multiple_dcits(
                time_series=time_series,
                name=variant_name,
                device=device,
                seed=seed,
                n_runs=n_runs,
                window_size=window_size,
                temperature=temperature,
                order=order,
                epochs=epochs,
            )

            raw_target_idx = (
                (time_series.shape[1] - test_series.shape[1])
                + window_size
                + torch.arange(test_windows.shape[0])
            )
            if base_name == "dxy":
                raw_target_idx = raw_target_idx + 1

            if is_shuffled:
                y_true_raw = None
                baseline_raw_mse = np.nan
            else:
                y_true_raw = raw_time_series[:, raw_target_idx].T
                y_baseline_raw = raw_time_series[:, raw_target_idx - 1].T
                baseline_raw_mse = torch.mean((y_true_raw - y_baseline_raw) ** 2).item()

            variant_model_mses = []
            raw_model_mses = []
            ljung_fracs_this_variant = []

            for run_key in result["run_keys"]:
                y_pred = result["runs"][run_key]["split_results"]["test"]["predictions"].detach().cpu()
                variant_model_mses.append(torch.mean((y_true - y_pred) ** 2).item())

                if is_shuffled:
                    residuals = (y_true - y_pred).numpy()
                else:
                    if base_name == "raw_xy":
                        y_pred_raw = y_pred
                    elif base_name == "rel_xy":
                        offset = raw_time_series[:, 0].reshape(1, -1)
                        y_pred_raw = y_pred + offset
                    elif base_name == "dxy":
                        previous_raw = raw_time_series[:, raw_target_idx - 1].T
                        y_pred_raw = previous_raw + y_pred
                    else:
                        raise ValueError(f"Unknown variant: {variant_name}")

                    raw_model_mse = torch.mean((y_true_raw - y_pred_raw) ** 2).item()
                    raw_model_mses.append(raw_model_mse)
                    residuals = (y_true_raw - y_pred_raw).numpy()

                significant_count = 0
                for series_idx in range(residuals.shape[1]):
                    lb_result = acorr_ljungbox(
                        residuals[:, series_idx],
                        lags=[window_size],
                        return_df=True,
                    )
                    if lb_result["lb_pvalue"].iloc[0] < 0.05:
                        significant_count += 1

                ljung_fracs_this_variant.append(significant_count / residuals.shape[1])

            mean_variant_model_mse = np.mean(variant_model_mses)
            std_variant_model_mse = np.std(variant_model_mses)
            variant_improvement_percent = 100 * (baseline_mse - mean_variant_model_mse) / baseline_mse
            mean_ljung_frac = np.mean(ljung_fracs_this_variant)

            if is_shuffled:
                mean_raw_model_mse = np.nan
                std_raw_model_mse = np.nan
                raw_space_improvement_percent = np.nan
            else:
                mean_raw_model_mse = np.mean(raw_model_mses)
                std_raw_model_mse = np.std(raw_model_mses)
                raw_space_improvement_percent = 100 * (baseline_raw_mse - mean_raw_model_mse) / baseline_raw_mse

            all_results[variant_name] = {
                "result": result,
                "frames_used": frames_used,
            }

            summary = result["summary"]
            row = {
                "cluster_idx": cluster_idx,
                "variant": variant_name,
                "base_variant": base_name,
                "is_shuffled_control": is_shuffled,
                "mean_test_loss": summary["mean_test_loss"],
                "std_test_loss": summary["std_test_loss"],
                "baseline_mse": baseline_mse,
                "mean_variant_model_mse": mean_variant_model_mse,
                "std_variant_model_mse": std_variant_model_mse,
                "variant_space_improvement_percent": variant_improvement_percent,
                "baseline_raw_mse": baseline_raw_mse,
                "mean_raw_model_mse": mean_raw_model_mse,
                "std_raw_model_mse": std_raw_model_mse,
                "raw_space_improvement_percent": raw_space_improvement_percent,
                "mean_ljung_frac": mean_ljung_frac,
            }
            if metadata is not None:
                row.update(metadata)
            training_rows.append(row)

            print(
                f"Mean test loss ({variant_name}): "
                f"{summary['mean_test_loss']:.6f} +/- {summary['std_test_loss']:.6f}"
            )
            print(
                f"Variant-space MSE ({variant_name}): "
                f"DCIts={mean_variant_model_mse:.6f} +/- {std_variant_model_mse:.6f}, "
                f"baseline={baseline_mse:.6f}, improvement={variant_improvement_percent:.2f}%"
            )
            if is_shuffled:
                print("Shuffled control: raw-space comparison skipped because time order is artificial.")
            else:
                print(
                    f"Raw-space MSE ({variant_name}): "
                    f"DCIts={mean_raw_model_mse:.6f} +/- {std_raw_model_mse:.6f}, "
                    f"raw baseline={baseline_raw_mse:.6f}, improvement={raw_space_improvement_percent:.2f}%"
                )
            print(f"Ljung-Box significant residual series ({variant_name}): {100 * mean_ljung_frac:.1f}%")

        plot_training_curves(
            all_results=all_results,
            cluster_idx=cluster_idx,
            save_path=result_dirs["training"] / f"cluster_{cluster_idx}_loss_curves.png",
            figure_dpi=figure_dpi,
        )
        plot_heatmaps_for_cluster(
            all_results=all_results,
            label_map=label_map,
            cluster_idx=cluster_idx,
            output_dir=result_dirs["heatmaps"],
            window_size=window_size,
            figure_dpi=figure_dpi,
            alpha_threshold=alpha_heatmap_threshold,
        )
        plot_self_alpha_lag1(
            all_results=all_results,
            label_map=label_map,
            cluster_idx=cluster_idx,
            output_dir=result_dirs["self_alpha"],
            figure_dpi=figure_dpi,
            variants_to_plot=variant_names,
            batch_size=1,
        )
        save_cluster_plot_data(
            cluster_idx=cluster_idx,
            cluster=cluster,
            all_results=all_results,
            label_map=label_map,
            variants=variants,
            save_dir=result_dirs["cluster_plot_data"],
            variant_names=variant_names,
            frames=frames,
            window_size=window_size,
            metadata=metadata,
        )
        save_event_interpretability_plots_for_cluster(
            cluster_idx=cluster_idx,
            cluster=cluster,
            all_results=all_results,
            label_map=label_map,
            frames=frames,
            save_dir=result_dirs["event_interpretability"],
            window_size=window_size,
            figure_dpi=figure_dpi,
            event_plot_variants=("rel_xy", "dxy"),
            average_heatmap_variants=variant_names,
            metrics=("Focuser", "alpha", "alpha * Q"),
            event_percentile=event_percentile,
            average_heatmap_split="test",
            all_window_splits=("train", "val", "test"),
            requested_bead_ids=requested_bead_ids,
            shuffle_seed=shuffle_seed,
        )
        save_raw_line_plots_for_cluster(
            cluster_idx=cluster_idx,
            cluster=cluster,
            all_results=all_results,
            label_map=label_map,
            frames=frames,
            output_dir=result_dirs["event_raw_line_plots"],
            window_size=window_size,
            figure_dpi=figure_dpi,
            variants=("rel_xy", "dxy"),
            metrics=("Focuser", "alpha", "alpha * Q"),
            split_names=("train", "val", "test"),
            event_percentile=event_percentile,
            shuffle_seed=shuffle_seed,
        )
        cluster_event_summary_rows = build_cluster_event_summary_rows(
            cluster_idx=cluster_idx,
            cluster=cluster,
            all_results=all_results,
            variants=variant_names,
            metrics=("Focuser", "alpha", "alpha * Q"),
            split_names=("train", "val", "test"),
            frames=frames,
            window_size=window_size,
            percentile=event_percentile,
            event_context_radius=event_context_radius,
            quiet_exclusion_radius=quiet_exclusion_radius,
            metadata=metadata,
            order_idx=1,
            shuffle_seed=shuffle_seed,
        )
        event_summary_rows.extend(cluster_event_summary_rows)
        save_cluster_event_summary(
            cluster_event_summary_rows,
            result_dirs["event_conditioned_summary"],
            cluster_idx,
        )

        del all_results
        del variants
        del raw_time_series
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    training_df = pd.DataFrame(training_rows)
    training_df.to_csv(result_dirs["root"] / "training_summary.csv", index=False)
    save_event_summary_outputs(event_summary_rows, result_dirs["event_conditioned_summary"])

    original_df = training_df[~training_df["is_shuffled_control"]]
    training_summary = {
        "n_clusters_analyzed": len(clusters_to_run),
    }

    for variant_name in variant_names:
        variant_df = training_df[training_df["variant"] == variant_name]
        if len(variant_df) > 0:
            training_summary[f"{variant_name}_mean_variant_improvement_percent"] = float(
                variant_df["variant_space_improvement_percent"].mean()
            )
            training_summary[f"{variant_name}_mean_ljung_percent"] = float(
                100 * variant_df["mean_ljung_frac"].mean()
            )

    for variant_name in original_variant_names:
        variant_df = original_df[original_df["variant"] == variant_name]
        if len(variant_df) > 0:
            training_summary[f"{variant_name}_mean_raw_improvement_percent"] = float(
                variant_df["raw_space_improvement_percent"].mean()
            )

    pd.DataFrame([training_summary]).to_csv(result_dirs["root"] / "training_overview.csv", index=False)

    return {
        "training_rows": training_rows,
        "training_summary": training_summary,
        "event_summary_rows": event_summary_rows,
    }
