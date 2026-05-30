import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from statsmodels.tsa.seasonal import STL

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import calculate_multiple_run_statistics, collect_multiple_runs


DEFAULT_OUTPUT_DIR = Path(r"C:\Users\dujme\Desktop\dipl\Zadnje poglavlje\Rezultati\Hidden_drive")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config_task1.json")


def time_series_hidden_trend(
    mean,
    ts_length,
    A=1.0,
    P=200,
    rho_U=0.3,
    sigma_U=0.1,
    sigma_X=0.1,
    sigma_Y=0.1,
    sigma_Z=0.1,
    burn_in=1000,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    total_length = ts_length + burn_in
    time_series = torch.zeros(4, total_length)

    time_series[:, 0:2] = torch.tensor(
        np.random.normal(0, 1, (4, 2)),
        dtype=torch.float32,
    )

    for t in range(2, total_length):
        U_prev = time_series[0, t - 1]
        X_prev = time_series[1, t - 1]
        Y_prev = time_series[2, t - 1]
        Z_prev = time_series[3, t - 1]
        U_lag2 = time_series[0, t - 2]

        eta_U = np.random.normal(mean, sigma_U)
        eta_X = np.random.normal(mean, sigma_X)
        eta_Y = np.random.normal(mean, sigma_Y)
        eta_Z = np.random.normal(mean, sigma_Z)

        seasonal_drive = A * np.sin(2 * np.pi * t / P)

        time_series[0, t] = seasonal_drive + rho_U * U_prev + eta_U
        time_series[1, t] = 0.6 * X_prev + 0.8 * U_prev + eta_X
        time_series[2, t] = 0.5 * Y_prev + 0.8 * U_lag2 + eta_Y
        time_series[3, t] = 0.4 * Z_prev + 0.7 * X_prev + eta_Z

    return time_series[:, burn_in:]


def make_ground_truth(window_length, rho_U):
    ground_truth_alpha = torch.zeros(4, 4, window_length)

    if rho_U != 0:
        ground_truth_alpha[0, 0, 0] = rho_U

    ground_truth_alpha[1, 1, 0] = 0.6
    ground_truth_alpha[1, 0, 0] = 0.8

    ground_truth_alpha[2, 2, 0] = 0.5
    ground_truth_alpha[2, 0, 1] = 0.8

    ground_truth_alpha[3, 3, 0] = 0.4
    ground_truth_alpha[3, 1, 0] = 0.7

    ground_truth_alpha_hidden_U = torch.zeros(3, 3, window_length)
    ground_truth_alpha_hidden_U[0, 0, 0] = 0.6
    ground_truth_alpha_hidden_U[1, 1, 0] = 0.5
    ground_truth_alpha_hidden_U[2, 2, 0] = 0.4
    ground_truth_alpha_hidden_U[2, 0, 0] = 0.7

    return {
        "observed_alpha": ground_truth_alpha,
        "observed_bias": torch.zeros(4),
        "observed_mask": (ground_truth_alpha != 0).float(),
        "hidden_alpha": ground_truth_alpha_hidden_U,
        "hidden_bias": torch.zeros(3),
        "hidden_mask": (ground_truth_alpha_hidden_U != 0).float(),
    }


def remove_seasonal_component(time_series_hidden_U, period):
    time_series_without_seasonal = torch.zeros_like(time_series_hidden_U)
    seasonal_components = torch.zeros_like(time_series_hidden_U)

    for series_idx in range(time_series_hidden_U.shape[0]):
        series = time_series_hidden_U[series_idx].detach().cpu().numpy()
        stl_result = STL(series, period=period).fit()

        seasonal = stl_result.seasonal
        series_without_seasonal = series - seasonal

        seasonal_components[series_idx] = torch.tensor(seasonal, dtype=torch.float32)
        time_series_without_seasonal[series_idx] = torch.tensor(series_without_seasonal, dtype=torch.float32)

    return time_series_without_seasonal, seasonal_components


def lag_accuracy_table(stats, ground_truth_alpha, alpha_key=1, names=None):
    alpha_mean = stats["alpha"][alpha_key]["mean"]
    ground_truth_alpha_np = ground_truth_alpha.detach().cpu().numpy()

    alpha_for_lags = np.flip(alpha_mean, axis=2)
    rows = []

    for target_idx, source_idx, true_lag_idx in np.argwhere(ground_truth_alpha_np != 0):
        estimated_lag_values = alpha_for_lags[target_idx, source_idx]
        best_lag_idx = np.argmax(np.abs(estimated_lag_values))

        rows.append(
            {
                "source": names[source_idx] if names is not None else source_idx + 1,
                "target": names[target_idx] if names is not None else target_idx + 1,
                "true_lag": true_lag_idx + 1,
                "estimated_best_lag": best_lag_idx + 1,
                "correct_lag": best_lag_idx == true_lag_idx,
                "ground_truth_alpha": ground_truth_alpha_np[target_idx, source_idx, true_lag_idx],
                "estimated_alpha_true_lag": estimated_lag_values[true_lag_idx],
                "estimated_alpha_best_lag": estimated_lag_values[best_lag_idx],
            }
        )

    return pd.DataFrame(rows)


def make_stable_links_table(stats, ground_truth_alpha, names, alpha_key=1, threshold=0.04, c=1.95):
    alpha_mean = np.flip(stats["alpha"][alpha_key]["mean"], axis=2)
    alpha_std = np.flip(stats["alpha"][alpha_key]["std"], axis=2)
    alpha_gt = ground_truth_alpha.detach().cpu().numpy()

    stable_mask = (np.abs(alpha_mean) > c * alpha_std) & (np.abs(alpha_mean) >= threshold)
    true_mask = alpha_gt != 0
    rows = []

    for target_idx, source_idx, lag_idx in np.argwhere(stable_mask | true_mask):
        stable = bool(stable_mask[target_idx, source_idx, lag_idx])
        true_link = bool(true_mask[target_idx, source_idx, lag_idx])

        if stable and true_link:
            link_type = "true_positive"
        elif stable and not true_link:
            link_type = "false_positive"
        elif not stable and true_link:
            link_type = "missed_true_link"
        else:
            link_type = "true_negative"

        rows.append(
            {
                "source": names[source_idx],
                "target": names[target_idx],
                "lag": lag_idx + 1,
                "mean_alpha": alpha_mean[target_idx, source_idx, lag_idx],
                "std_alpha": alpha_std[target_idx, source_idx, lag_idx],
                "ground_truth_alpha": alpha_gt[target_idx, source_idx, lag_idx],
                "stable": stable,
                "true_link": true_link,
                "link_type": link_type,
            }
        )

    return pd.DataFrame(rows)


def make_false_positive_table(stable_links_table):
    if stable_links_table.empty:
        return pd.DataFrame(columns=["source", "target", "lag", "mean_alpha", "std_alpha"])

    table = stable_links_table[stable_links_table["link_type"] == "false_positive"].copy()

    if table.empty:
        return table[["source", "target", "lag", "mean_alpha", "std_alpha"]]

    table["_abs_alpha"] = table["mean_alpha"].abs()
    table = (
        table.sort_values("_abs_alpha", ascending=False)
        .groupby(["source", "target"], as_index=False)
        .first()
        .sort_values("_abs_alpha", ascending=False)
        .drop(columns="_abs_alpha")
    )

    return table[["source", "target", "lag", "mean_alpha", "std_alpha"]]


def sample_std(values):
    values = np.asarray(values)
    ddof = 1 if values.size > 1 else 0
    return values.std(ddof=ddof)


def calculate_case_summary(results, stats, ground_truth_alpha, names, threshold):
    mse_values = np.array(
        [results[run_key]["test_loss"] for run_key in results if run_key.startswith("run_")]
    )
    rmse_values = np.sqrt(mse_values)
    mae_values = np.array(
        [results[run_key]["test_mae"] for run_key in results if run_key.startswith("run_")]
    )

    stable_links = make_stable_links_table(stats, ground_truth_alpha, names, threshold=threshold)
    lag_table = lag_accuracy_table(stats, ground_truth_alpha, names=names)
    lag_table["correct_sign"] = (
        np.sign(lag_table["estimated_alpha_true_lag"]) == np.sign(lag_table["ground_truth_alpha"])
    )

    alpha_est = np.flip(stats["alpha"][1]["mean"], axis=2).flatten()
    alpha_gt = ground_truth_alpha.detach().cpu().numpy().flatten()
    alpha_correlation = np.corrcoef(alpha_est, alpha_gt)[0, 1]

    false_positive_rows = stable_links[stable_links["link_type"] == "false_positive"]

    if false_positive_rows.empty:
        max_false_positive_strength = 0.0
        mean_false_positive_strength = 0.0
    else:
        false_positive_abs = false_positive_rows["mean_alpha"].abs()
        max_false_positive_strength = false_positive_abs.max()
        mean_false_positive_strength = false_positive_abs.mean()

    summary = {
        "mean_MSE": mse_values.mean(),
        "std_MSE": sample_std(mse_values),
        "mean_RMSE": rmse_values.mean(),
        "std_RMSE": sample_std(rmse_values),
        "mean_MAE": mae_values.mean(),
        "std_MAE": sample_std(mae_values),
        "found_stable_links": int(stable_links["stable"].sum()),
        "true_positives": int((stable_links["link_type"] == "true_positive").sum()),
        "false_positives": int((stable_links["link_type"] == "false_positive").sum()),
        "missed_true_links": int((stable_links["link_type"] == "missed_true_link").sum()),
        "max_false_positive_strength": max_false_positive_strength,
        "mean_false_positive_strength": mean_false_positive_strength,
        "lag_accuracy": lag_table["correct_lag"].mean(),
        "sign_accuracy": lag_table["correct_sign"].mean(),
        "alpha_correlation": alpha_correlation,
    }

    return summary, stable_links, make_false_positive_table(stable_links), lag_table


def tensor_to_numpy(series):
    if isinstance(series, torch.Tensor):
        return series.detach().cpu().numpy()
    return np.asarray(series)


def save_series_plot(series, labels, title, save_path):
    data = tensor_to_numpy(series)

    fig, ax = plt.subplots(figsize=(12, 4))

    for series_idx, label in enumerate(labels):
        ax.plot(data[series_idx], lw=0.7, alpha=0.8, label=label)

    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)

    if len(labels) <= 6:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_time_series_figures(run_dir, observed, hidden, hidden_without_seasonal):
    time_dir = run_dir / "time_series"
    time_dir.mkdir(parents=True, exist_ok=True)

    save_series_plot(observed, ["U", "X", "Y", "Z"], "U, X, Y, Z", time_dir / "all_UXYZ.pdf")
    save_series_plot(observed[0:1], ["U"], "U", time_dir / "U.pdf")
    save_series_plot(observed[1:2], ["X"], "X", time_dir / "X.pdf")
    save_series_plot(observed[2:3], ["Y"], "Y", time_dir / "Y.pdf")
    save_series_plot(observed[3:4], ["Z"], "Z", time_dir / "Z.pdf")
    save_series_plot(hidden, ["X", "Y", "Z"], "X, Y, Z", time_dir / "XYZ.pdf")
    save_series_plot(
        hidden_without_seasonal,
        ["X", "Y", "Z"],
        "X, Y, Z without seasonal component",
        time_dir / "XYZ_without_seasonal.pdf",
    )


def save_stl_example(hidden, period, save_path):
    stl_example = STL(hidden[0].detach().cpu().numpy(), period=period).fit()
    fig = stl_example.plot()
    fig.suptitle("STL decomposition example: X", y=1.02)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_alphas_one_canvas(
    alpha,
    ground_truth_alpha,
    cmap="seismic",
    figsize=(6, 5),
    font_size=14,
    title=r"\alpha",
    space=0.1,
    cbar_font=11,
    save_path=None,
):
    if isinstance(ground_truth_alpha, torch.Tensor):
        ground_truth_alpha = ground_truth_alpha.detach().cpu().numpy()

    n_targets = alpha.shape[0]

    fig, axes = plt.subplots(
        n_targets,
        2,
        figsize=(figsize[0] * 2, figsize[1] * n_targets),
        gridspec_kw={"wspace": space, "hspace": space},
    )

    if n_targets == 1:
        axes = np.array([axes])

    for i in range(n_targets):
        for idx, (data, title_prefix, needs_flip) in enumerate(
            zip(
                [alpha[i], ground_truth_alpha[i]],
                [rf"${title}_{{{i+1},j,l}}$", rf"Ground truth ${title}_{{{i+1},j,l}}$"],
                [True, False],
            )
        ):
            ax = axes[i, idx]

            if needs_flip:
                if data.ndim < 2:
                    raise ValueError(f"Expected 2D data, got {data.ndim}D data with shape {data.shape}")
                data = np.flip(data, axis=1)

            abs_max_value = np.max(np.abs(data))
            abs_max_value = max(abs_max_value, 1e-12)

            im = ax.imshow(data, cmap=cmap, vmin=-abs_max_value, vmax=abs_max_value)
            data_rounded = np.round(data, 2)

            for ii in range(data_rounded.shape[0]):
                for jj in range(data_rounded.shape[1]):
                    value = data_rounded[ii, jj]
                    if value != 0:
                        text_color = "black" if abs(value) < 0.2 * abs_max_value else "white"
                        ax.text(
                            jj,
                            ii,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=font_size,
                        )

            ax.set_title(title_prefix, fontsize=font_size + 2)
            ax.set_xlabel("$l$", fontsize=font_size)
            ax.set_ylabel("$j$", rotation=0, fontsize=font_size)
            ax.set_xticks(range(data.shape[-1]))
            ax.set_xticklabels(range(1, data.shape[-1] + 1), fontsize=font_size - 2)
            ax.set_yticks(range(data.shape[0]))
            ax.set_yticklabels(range(1, data.shape[0] + 1), fontsize=font_size - 2)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            cbar.ax.tick_params(labelsize=cbar_font)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_metric_lag_comparison(synthetic_cases, metric="alpha", alpha_key=1, save_path=None, cmap="seismic"):
    case_names = ["observed_U", "hidden_U", "hidden_U_without_seasonal"]
    case_titles = ["observed U", "hidden U", "hidden U - seasonal removed"]

    estimated_by_case = []
    truth_by_case = []
    labels_by_case = []

    for case_name in case_names:
        case = synthetic_cases[case_name]

        if metric == "alpha":
            estimated = case["stats"]["alpha"][alpha_key]["mean"]
            truth = case["ground_truth_alpha"].detach().cpu().numpy()
            colorbar_label = r"$\alpha$"
        elif metric == "focuser":
            estimated = case["stats"]["f"][alpha_key]["mean"]
            truth = case["alpha_mask"].detach().cpu().numpy()
            colorbar_label = "Focuser"
        elif metric == "coefficient":
            estimated = case["stats"]["c"][alpha_key]["mean"]
            truth = case["alpha_mask"].detach().cpu().numpy()
            colorbar_label = "Coefficient"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        estimated_by_case.append(np.flip(estimated, axis=2))
        truth_by_case.append(truth)
        labels_by_case.append(case["names"])

    n_cases = len(case_names)
    n_lags = max(values.shape[2] for values in estimated_by_case)
    n_rows = 2 * n_cases

    vmax = max(
        max(np.max(np.abs(est)), np.max(np.abs(truth)))
        for est, truth in zip(estimated_by_case, truth_by_case)
    )
    vmax = max(vmax, 1e-12)

    fig, axes = plt.subplots(
        n_rows,
        n_lags,
        figsize=(3.2 * n_lags, 2.8 * n_rows),
        constrained_layout=True,
    )

    for case_idx, (case_title, estimated, truth, labels) in enumerate(
        zip(case_titles, estimated_by_case, truth_by_case, labels_by_case)
    ):
        for row_offset, row_label, values in [(0, "estimated", estimated), (1, "ground truth", truth)]:
            row_idx = 2 * case_idx + row_offset
            n_series = values.shape[0]

            for lag_idx in range(n_lags):
                ax = axes[row_idx, lag_idx]
                matrix = values[:, :, lag_idx]

                im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
                data_rounded = np.round(matrix, 2)

                for target_idx in range(data_rounded.shape[0]):
                    for source_idx in range(data_rounded.shape[1]):
                        value = data_rounded[target_idx, source_idx]
                        if value != 0:
                            text_color = "black" if abs(value) < 0.2 * vmax else "white"
                            ax.text(
                                source_idx,
                                target_idx,
                                f"{value:.2f}",
                                ha="center",
                                va="center",
                                color=text_color,
                                fontsize=8,
                            )

                if row_idx == 0:
                    ax.set_title(f"lag {lag_idx + 1}", fontsize=12)

                if lag_idx == 0:
                    ax.set_ylabel(f"{case_title}\n{row_label}\ntarget", fontsize=10)

                ax.set_xticks(range(n_series))
                ax.set_yticks(range(n_series))
                ax.set_xticklabels(labels, fontsize=9)
                ax.set_yticklabels(labels, fontsize=9)
                ax.set_xlabel("source", fontsize=9)

                ax.set_xticks(np.arange(-0.5, n_series, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, n_series, 1), minor=True)
                ax.grid(which="minor", color="black", linestyle="-", linewidth=0.4)
                ax.tick_params(which="minor", bottom=False, left=False)

    fig.colorbar(im, ax=axes, shrink=0.85, label=colorbar_label)
    fig.suptitle(f"{metric}: estimated vs ground truth across cases", fontsize=14)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def float_token(value):
    return f"{value:.1f}".replace("-", "m").replace(".", "p")


def make_run_id(A, P, rho_U, is_control=False):
    if is_control:
        return f"A_{float_token(A)}_rho_{float_token(rho_U)}"
    return f"A_{float_token(A)}_P_{int(P)}_rho_{float_token(rho_U)}"


def save_json(data, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_training_bundle(case_dir, case):
    with open(case_dir / "training_results.pkl", "wb") as f:
        pickle.dump(
            {
                "results": case["results"],
                "stats": case["stats"],
                "ground_truth_alpha": case["ground_truth_alpha"],
                "ground_truth_bias": case["ground_truth_bias"],
                "alpha_mask": case["alpha_mask"],
                "names": case["names"],
            },
            f,
        )


def train_one_case(case, n_runs, window_length, temperature, order, train_config):
    results = collect_multiple_runs(
        n_runs=n_runs,
        time_series=case["time_series"],
        window_size=window_length,
        temperature=temperature,
        order=order,
        config=train_config,
        verbose=False,
    )

    stats = calculate_multiple_run_statistics(results)
    case["results"] = results
    case["stats"] = stats


def build_cases(observed, hidden, hidden_without_seasonal, truth):
    return {
        "observed_U": {
            "time_series": observed,
            "ground_truth_alpha": truth["observed_alpha"],
            "ground_truth_bias": truth["observed_bias"],
            "alpha_mask": truth["observed_mask"],
            "names": ["U", "X", "Y", "Z"],
        },
        "hidden_U": {
            "time_series": hidden,
            "ground_truth_alpha": truth["hidden_alpha"],
            "ground_truth_bias": truth["hidden_bias"],
            "alpha_mask": truth["hidden_mask"],
            "names": ["X", "Y", "Z"],
        },
        "hidden_U_without_seasonal": {
            "time_series": hidden_without_seasonal,
            "ground_truth_alpha": truth["hidden_alpha"],
            "ground_truth_bias": truth["hidden_bias"],
            "alpha_mask": truth["hidden_mask"],
            "names": ["X", "Y", "Z"],
        },
    }


def save_case_outputs(case_dir, case):
    case_dir.mkdir(parents=True, exist_ok=True)

    save_training_bundle(case_dir, case)
    case["stable_links"].to_csv(case_dir / "stable_links.csv", index=False)
    case["false_positives"].to_csv(case_dir / "false_positives.csv", index=False)
    case["lag_table"].to_csv(case_dir / "lag_table.csv", index=False)

    plot_alphas_one_canvas(
        case["stats"]["alpha"][1]["mean"],
        case["ground_truth_alpha"],
        figsize=(8, 7),
        font_size=14,
        space=0.15,
        save_path=case_dir / "alpha_heatmaps.pdf",
    )

    plot_alphas_one_canvas(
        case["stats"]["f"][1]["mean"],
        case["alpha_mask"],
        title=r"f",
        figsize=(8, 7),
        font_size=14,
        space=0.15,
        save_path=case_dir / "focuser_heatmaps.pdf",
    )

    plot_alphas_one_canvas(
        case["stats"]["c"][1]["mean"],
        case["alpha_mask"],
        title=r"C",
        figsize=(8, 7),
        font_size=14,
        space=0.15,
        save_path=case_dir / "coefficient_heatmaps.pdf",
    )


def write_global_summaries(summary_dir, case_summary_rows, stable_rows, lag_rows):
    summary_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(case_summary_rows).to_csv(summary_dir / "all_case_summary.csv", index=False)
    pd.DataFrame(stable_rows).to_csv(summary_dir / "all_stable_links.csv", index=False)
    pd.DataFrame(lag_rows).to_csv(summary_dir / "all_lag_tables.csv", index=False)


def run_parameter_set(args, group_name, run_dir, A, P, rho_U, is_control, device):
    run_dir.mkdir(parents=True, exist_ok=True)

    truth = make_ground_truth(args.window_length, rho_U)

    time_series = time_series_hidden_trend(
        mean=0,
        ts_length=args.ts_length,
        A=A,
        P=P,
        rho_U=rho_U,
        sigma_U=args.sigma_U,
        sigma_X=args.sigma_X,
        sigma_Y=args.sigma_Y,
        sigma_Z=args.sigma_Z,
        burn_in=args.burn_in,
        seed=args.seed,
    )

    observed = time_series
    hidden = time_series[1:, :]
    hidden_without_seasonal, seasonal_components = remove_seasonal_component(hidden, P)

    np.savez_compressed(
        run_dir / "generated_series.npz",
        observed_U=tensor_to_numpy(observed),
        hidden_U=tensor_to_numpy(hidden),
        hidden_U_without_seasonal=tensor_to_numpy(hidden_without_seasonal),
        seasonal_components=tensor_to_numpy(seasonal_components),
    )

    run_config = {
        "group": group_name,
        "A": A,
        "P": P,
        "rho_U": rho_U,
        "is_control_A0": is_control,
        "T": args.ts_length,
        "L": args.window_length,
        "burn_in": args.burn_in,
        "seed": args.seed,
        "n_runs": args.n_runs,
        "noise": {
            "sigma_U": args.sigma_U,
            "sigma_X": args.sigma_X,
            "sigma_Y": args.sigma_Y,
            "sigma_Z": args.sigma_Z,
        },
        "model": {
            "temperature": args.temperature,
            "order": [1, 1],
            "learning_rate": args.learning_rate,
            "scheduler_patience": args.scheduler_patience,
            "early_stopping_modifier": args.early_stopping_modifier,
            "criterion": "MSELoss",
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1 - args.train_ratio - args.val_ratio,
            "device": str(device),
        },
        "cases": ["observed_U", "hidden_U", "hidden_U_without_seasonal"],
        "alpha_threshold": args.alpha_threshold,
        "stability_c": 1.95,
        "seasonal_removal": {
            "method": "STL",
            "period": P,
            "removed_component": "seasonal",
        },
    }
    save_json(run_config, run_dir / "config.json")

    save_time_series_figures(run_dir, observed, hidden, hidden_without_seasonal)
    save_stl_example(hidden, P, run_dir / "time_series" / "STL_X.pdf")

    cases = build_cases(observed, hidden, hidden_without_seasonal, truth)

    train_config = {
        "verbose": False,
        "device": device,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "scheduler_patience": args.scheduler_patience,
        "early_stopping_modifier": args.early_stopping_modifier,
        "criterion": nn.MSELoss(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
    }

    case_summary_rows = []
    stable_rows = []
    lag_rows = []

    for case_name, case in cases.items():
        print(f"  case: {case_name}")

        train_one_case(
            case=case,
            n_runs=args.n_runs,
            window_length=args.window_length,
            temperature=args.temperature,
            order=[1, 1],
            train_config=train_config,
        )

        summary, stable_links, false_positives, lag_table = calculate_case_summary(
            case["results"],
            case["stats"],
            case["ground_truth_alpha"],
            case["names"],
            args.alpha_threshold,
        )

        case["stable_links"] = stable_links
        case["false_positives"] = false_positives
        case["lag_table"] = lag_table

        case_summary = {
            "group": group_name,
            "run_id": run_dir.name,
            "A": A,
            "P": P,
            "rho_U": rho_U,
            "case": case_name,
            **summary,
        }
        case_summary_rows.append(case_summary)

        stable_links_with_meta = stable_links.copy()
        stable_links_with_meta.insert(0, "case", case_name)
        stable_links_with_meta.insert(0, "rho_U", rho_U)
        stable_links_with_meta.insert(0, "P", P)
        stable_links_with_meta.insert(0, "A", A)
        stable_links_with_meta.insert(0, "run_id", run_dir.name)
        stable_links_with_meta.insert(0, "group", group_name)
        stable_rows.extend(stable_links_with_meta.to_dict("records"))

        lag_table_with_meta = lag_table.copy()
        lag_table_with_meta.insert(0, "case", case_name)
        lag_table_with_meta.insert(0, "rho_U", rho_U)
        lag_table_with_meta.insert(0, "P", P)
        lag_table_with_meta.insert(0, "A", A)
        lag_table_with_meta.insert(0, "run_id", run_dir.name)
        lag_table_with_meta.insert(0, "group", group_name)
        lag_rows.extend(lag_table_with_meta.to_dict("records"))

        save_case_outputs(run_dir / case_name, case)

        del case["results"]
        del case["time_series"]
        del case["stable_links"]
        del case["false_positives"]
        del case["lag_table"]
        del stable_links, false_positives, lag_table
        del stable_links_with_meta, lag_table_with_meta
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    pd.DataFrame(case_summary_rows).to_csv(run_dir / "case_summary.csv", index=False)

    comparison_dir = run_dir / "comparison_heatmaps"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_lag_comparison(cases, metric="alpha", save_path=comparison_dir / "alpha_all_cases.pdf")
    plot_metric_lag_comparison(cases, metric="focuser", save_path=comparison_dir / "focuser_all_cases.pdf")
    plot_metric_lag_comparison(cases, metric="coefficient", save_path=comparison_dir / "coefficient_all_cases.pdf")

    del cases, time_series, observed, hidden, hidden_without_seasonal, seasonal_components
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return case_summary_rows, stable_rows, lag_rows


def parse_float_list(value):
    return [float(part) for part in value.split(",")]


def parse_int_list(value):
    return [int(part) for part in value.split(",")]


def load_config_defaults(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    aliases = {
        "rho_U_values": "rho_values",
        "T": "ts_length",
        "L": "window_length",
    }

    for config_key, arg_key in aliases.items():
        if config_key in config and arg_key not in config:
            config[arg_key] = config[config_key]

    if "output_dir" in config:
        config["output_dir"] = Path(config["output_dir"])

    return config


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Optional JSON config file, e.g. {DEFAULT_CONFIG_PATH}",
    )
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Run DCIts hidden-drive synthetic Task 1 pipeline.",
        parents=[config_parser],
    )

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--combo-limit", type=int, default=None, help="Optional limit for testing.")

    parser.add_argument("--A-values", type=parse_float_list, default=[0.5, 1.0, 2.0])
    parser.add_argument("--P-values", type=parse_int_list, default=[50, 200, 500])
    parser.add_argument("--rho-values", type=parse_float_list, default=[0.0, 0.3, 0.7])
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--only-control", action="store_true")

    parser.add_argument("--ts-length", type=int, default=20000)
    parser.add_argument("--window-length", type=int, default=5)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sigma-U", type=float, default=0.1)
    parser.add_argument("--sigma-X", type=float, default=0.05)
    parser.add_argument("--sigma-Y", type=float, default=0.2)
    parser.add_argument("--sigma-Z", type=float, default=0.5)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--early-stopping-modifier", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha-threshold", type=float, default=0.04)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    if config_args.config is not None:
        parser.set_defaults(**load_config_defaults(config_args.config))

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)

    return args


def select_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    device = select_device(args.device)

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = args.output_dir / "summary_tables"
    main_sweep_dir = args.output_dir / "main_sweep"
    control_dir = args.output_dir / "control_A0"

    top_level_config = {
        "task": "Hidden_drive",
        "config_file": str(args.config) if args.config is not None else None,
        "A_values": args.A_values,
        "P_values": args.P_values,
        "rho_U_values": args.rho_values,
        "control_A": 0.0,
        "T": args.ts_length,
        "L": args.window_length,
        "burn_in": args.burn_in,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "sigma_U": args.sigma_U,
        "sigma_X": args.sigma_X,
        "sigma_Y": args.sigma_Y,
        "sigma_Z": args.sigma_Z,
        "alpha_threshold": args.alpha_threshold,
        "output_format": "pdf",
    }
    save_json(top_level_config, args.output_dir / "run_config.json")

    jobs = []

    if not args.only_control:
        for A in args.A_values:
            for P in args.P_values:
                for rho_U in args.rho_values:
                    run_id = make_run_id(A, P, rho_U, is_control=False)
                    jobs.append(("main_sweep", main_sweep_dir / run_id, A, P, rho_U, False))

    if not args.skip_control:
        for rho_U in args.rho_values:
            A = 0.0
            P = args.P_values[0]
            run_id = make_run_id(A, P, rho_U, is_control=True)
            jobs.append(("control_A0", control_dir / run_id, A, P, rho_U, True))

    if args.combo_limit is not None:
        jobs = jobs[: args.combo_limit]

    all_case_rows = []
    all_stable_rows = []
    all_lag_rows = []

    print(f"Total parameter jobs: {len(jobs)}")

    for job_idx, (group_name, run_dir, A, P, rho_U, is_control) in enumerate(jobs, start=1):
        print(f"Job {job_idx}/{len(jobs)}: {group_name}, A={A}, P={P}, rho_U={rho_U}")

        case_rows, stable_rows, lag_rows = run_parameter_set(
            args=args,
            group_name=group_name,
            run_dir=run_dir,
            A=A,
            P=P,
            rho_U=rho_U,
            is_control=is_control,
            device=device,
        )

        all_case_rows.extend(case_rows)
        all_stable_rows.extend(stable_rows)
        all_lag_rows.extend(lag_rows)

        write_global_summaries(summary_dir, all_case_rows, all_stable_rows, all_lag_rows)

    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
