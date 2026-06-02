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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import calculate_multiple_run_statistics, collect_multiple_runs


DEFAULT_OUTPUT_DIR = Path("results/regime_change")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("regime_change_config.json")
NAMES = ["X1", "X2", "X3", "X4"]


def sample_std(values, axis=None):
    values = np.asarray(values)
    n = values.size if axis is None else values.shape[axis]
    ddof = 1 if n > 1 else 0
    return np.std(values, axis=axis, ddof=ddof)


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def safe_name(value):
    return str(value).replace("-", "m").replace(".", "p")


def sigma_token(sigma_set):
    return f"sigma_{safe_name(sigma_set['name'])}"


def b_token(B_min, B_max):
    return f"B_{B_min}_{B_max}"


def time_series_regime_change(
    mean,
    ts_length,
    sigma_X1=0.1,
    sigma_X2=0.1,
    sigma_X3=0.1,
    sigma_X4=0.1,
    B_min=50,
    B_max=150,
    burn_in=1000,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    total_length = ts_length + burn_in
    regimes = np.zeros(total_length, dtype=int)

    t = 0
    current_regime = 0
    while t < total_length:
        block_length = np.random.randint(B_min, B_max + 1)
        end = min(t + block_length, total_length)
        regimes[t:end] = current_regime
        t = end
        current_regime = 1 - current_regime

    source_change = torch.zeros(4, total_length)
    lag_change = torch.zeros(4, total_length)
    sign_change = torch.zeros(4, total_length)

    initial_values = torch.tensor(np.random.normal(0, 1, (4, 4)), dtype=torch.float32)
    source_change[:, 0:4] = initial_values
    lag_change[:, 0:4] = initial_values
    sign_change[:, 0:4] = initial_values

    for t in range(4, total_length):
        eta_X1 = np.random.normal(mean, sigma_X1)
        eta_X2 = np.random.normal(mean, sigma_X2)
        eta_X3 = np.random.normal(mean, sigma_X3)
        eta_X4 = np.random.normal(mean, sigma_X4)

        source_change[0, t] = 0.4 * source_change[0, t - 1] + eta_X1
        source_change[3, t] = 0.5 * source_change[3, t - 1] + eta_X4

        lag_change[0, t] = 0.4 * lag_change[0, t - 1] + eta_X1
        lag_change[3, t] = 0.5 * lag_change[3, t - 1] + eta_X4

        sign_change[0, t] = 0.4 * sign_change[0, t - 1] + eta_X1
        sign_change[3, t] = 0.5 * sign_change[3, t - 1] + eta_X4

        if regimes[t] == 0:
            source_change[1, t] = 0.7 * source_change[0, t - 2] + eta_X2
            source_change[2, t] = 0.6 * source_change[1, t - 1] + eta_X3

            lag_change[1, t] = 0.7 * lag_change[0, t - 1] + eta_X2
            lag_change[2, t] = 0.6 * lag_change[1, t - 1] + eta_X3

            sign_change[1, t] = 0.7 * sign_change[0, t - 2] + eta_X2
            sign_change[2, t] = 0.6 * sign_change[1, t - 1] + eta_X3
        else:
            source_change[1, t] = 0.7 * source_change[3, t - 2] + eta_X2
            source_change[2, t] = -0.6 * source_change[0, t - 3] + eta_X3

            lag_change[1, t] = 0.7 * lag_change[0, t - 4] + eta_X2
            lag_change[2, t] = 0.6 * lag_change[1, t - 1] + eta_X3

            sign_change[1, t] = 0.7 * sign_change[0, t - 2] + eta_X2
            sign_change[2, t] = -0.6 * sign_change[1, t - 1] + eta_X3

    regimes = regimes[burn_in:]

    return {
        "source_change": (source_change[:, burn_in:], regimes),
        "lag_change": (lag_change[:, burn_in:], regimes),
        "sign_change": (sign_change[:, burn_in:], regimes),
    }


def pad_lag_axis(values, n_lags, fill_value=0.0):
    values = tensor_to_numpy(values)
    if values.shape[2] == n_lags:
        return values

    out = np.full(values.shape[:2] + (n_lags,), fill_value, dtype=values.dtype)
    n_copy = min(values.shape[2], n_lags)
    out[:, :, :n_copy] = values[:, :, :n_copy]
    return out


def pad_lag_matrix(values, n_lags, fill_value=0.0):
    values = np.asarray(values)
    if values.shape[1] == n_lags:
        return values

    out = np.full((values.shape[0], n_lags), fill_value, dtype=values.dtype)
    n_copy = min(values.shape[1], n_lags)
    out[:, :n_copy] = values[:, :n_copy]
    return out


def align_lag_axes(estimated, ground_truth, fill_value=0.0):
    estimated = tensor_to_numpy(estimated)
    ground_truth = tensor_to_numpy(ground_truth)
    n_lags = max(estimated.shape[2], ground_truth.shape[2])
    return (
        pad_lag_axis(estimated, n_lags, fill_value=fill_value),
        pad_lag_axis(ground_truth, n_lags, fill_value=0.0),
    )


def make_ground_truths(window_length):
    truth = {}
    ground_truth_lags = max(window_length, 4)

    source_R0 = torch.zeros(4, 4, ground_truth_lags)
    source_R1 = torch.zeros(4, 4, ground_truth_lags)
    source_R0[0, 0, 0] = source_R1[0, 0, 0] = 0.4
    source_R0[3, 3, 0] = source_R1[3, 3, 0] = 0.5
    source_R0[1, 0, 1] = 0.7
    source_R0[2, 1, 0] = 0.6
    source_R1[1, 3, 1] = 0.7
    source_R1[2, 0, 2] = -0.6
    truth["source_change"] = {
        "R0": source_R0,
        "R1": source_R1,
        "interesting_alphas": [
            (1, 0, 1),
            (1, 3, 1),
            (2, 1, 0),
            (2, 0, 2),
        ],
    }

    lag_R0 = torch.zeros(4, 4, ground_truth_lags)
    lag_R1 = torch.zeros(4, 4, ground_truth_lags)
    lag_R0[0, 0, 0] = lag_R1[0, 0, 0] = 0.4
    lag_R0[3, 3, 0] = lag_R1[3, 3, 0] = 0.5
    lag_R0[1, 0, 0] = 0.7
    lag_R0[2, 1, 0] = 0.6
    lag_R1[1, 0, 3] = 0.7
    lag_R1[2, 1, 0] = 0.6
    truth["lag_change"] = {
        "R0": lag_R0,
        "R1": lag_R1,
        "interesting_alphas": [
            (1, 0, 0),
            (1, 0, 3),
            (2, 1, 0),
        ],
    }

    sign_R0 = torch.zeros(4, 4, ground_truth_lags)
    sign_R1 = torch.zeros(4, 4, ground_truth_lags)
    sign_R0[0, 0, 0] = sign_R1[0, 0, 0] = 0.4
    sign_R0[3, 3, 0] = sign_R1[3, 3, 0] = 0.5
    sign_R0[1, 0, 1] = 0.7
    sign_R1[1, 0, 1] = 0.7
    sign_R0[2, 1, 0] = 0.6
    sign_R1[2, 1, 0] = -0.6
    truth["sign_change"] = {
        "R0": sign_R0,
        "R1": sign_R1,
        "interesting_alphas": [
            (1, 0, 1),
            (2, 1, 0),
        ],
    }

    for item in truth.values():
        item["mask_R0"] = (item["R0"] != 0).float()
        item["mask_R1"] = (item["R1"] != 0).float()

    return truth


def test_window_regimes(regimes, window_size, train_ratio=0.6, val_ratio=0.2):
    regimes = np.asarray(regimes)
    train_end = int(train_ratio * len(regimes))
    val_end = train_end + int(val_ratio * len(regimes))
    return regimes[val_end + window_size :]


def regime_stats(results, regimes, window_size, train_ratio, val_ratio, seq_key="alpha_seq", order_idx=1):
    test_regimes = test_window_regimes(regimes, window_size, train_ratio, val_ratio)
    run_keys = [k for k in results if k.startswith("run_")]
    out = {}

    for regime in [0, 1]:
        regime_runs = []
        for run_key in run_keys:
            seq = results[run_key][seq_key][order_idx]
            regime_runs.append(seq[test_regimes == regime].mean(axis=0))

        regime_runs = np.stack(regime_runs)
        out[regime] = {
            "mean": regime_runs.mean(axis=0),
            "std": sample_std(regime_runs, axis=0),
        }

    return out


def weighted_global_truth(ground_truth_R0, ground_truth_R1, regimes, window_size, train_ratio, val_ratio):
    test_regimes = test_window_regimes(regimes, window_size, train_ratio, val_ratio)
    p0 = np.mean(test_regimes == 0)
    p1 = np.mean(test_regimes == 1)
    return p0 * ground_truth_R0 + p1 * ground_truth_R1


def task2_metrics(results_case, alpha_stats, ground_truth_R0, ground_truth_R1, names, threshold=0.04, c=1.95):
    run_keys = [k for k in results_case if k.startswith("run_")]
    mse = np.array([float(results_case[k]["test_loss"]) for k in run_keys])
    rmse = np.sqrt(mse)
    mae = np.array([float(results_case[k]["test_mae"]) for k in run_keys])

    prediction_metrics = pd.DataFrame(
        [
            {
                "mean_MSE": mse.mean(),
                "std_MSE": sample_std(mse),
                "mean_RMSE": rmse.mean(),
                "std_RMSE": sample_std(rmse),
                "mean_MAE": mae.mean(),
                "std_MAE": sample_std(mae),
            }
        ]
    )

    regime_rows = []
    link_rows = []
    lag_sign_rows = []

    for regime, ground_truth in [(0, ground_truth_R0), (1, ground_truth_R1)]:
        alpha_mean = np.flip(alpha_stats[regime]["mean"], axis=2)
        alpha_std = np.flip(alpha_stats[regime]["std"], axis=2)
        alpha_gt = tensor_to_numpy(ground_truth)
        estimated_n_lags = alpha_mean.shape[2]
        n_lags = max(estimated_n_lags, alpha_gt.shape[2])
        alpha_mean = pad_lag_axis(alpha_mean, n_lags)
        alpha_std = pad_lag_axis(alpha_std, n_lags)
        alpha_gt = pad_lag_axis(alpha_gt, n_lags)

        stable_mask = (np.abs(alpha_mean) > c * alpha_std) & (np.abs(alpha_mean) >= threshold)
        true_mask = alpha_gt != 0

        for target, source, lag in np.argwhere(stable_mask | true_mask):
            lag_available = lag < estimated_n_lags
            stable = bool(stable_mask[target, source, lag])
            true_link = bool(true_mask[target, source, lag])
            if stable and true_link:
                link_type = "true_positive"
            elif stable and not true_link:
                link_type = "false_positive"
            elif not stable and true_link:
                link_type = "missed_true_link"
            else:
                link_type = "true_negative"

            link_rows.append(
                {
                    "regime": regime,
                    "source": names[source],
                    "target": names[target],
                    "lag": lag + 1,
                    "mean_alpha": alpha_mean[target, source, lag],
                    "std_alpha": alpha_std[target, source, lag],
                    "ground_truth_alpha": alpha_gt[target, source, lag],
                    "lag_available": lag_available,
                    "stable": stable,
                    "true_link": true_link,
                    "link_type": link_type,
                }
            )

        for target, source, true_lag in np.argwhere(true_mask):
            estimated_lags = alpha_mean[target, source]
            best_lag = int(np.argmax(np.abs(estimated_lags)))
            true_lag_available = true_lag < estimated_n_lags
            estimated_alpha_true_lag = estimated_lags[true_lag] if true_lag_available else np.nan
            lag_sign_rows.append(
                {
                    "regime": regime,
                    "source": names[source],
                    "target": names[target],
                    "true_lag": true_lag + 1,
                    "estimated_best_lag": best_lag + 1,
                    "correct_lag": best_lag == true_lag,
                    "ground_truth_alpha": alpha_gt[target, source, true_lag],
                    "lag_available": true_lag_available,
                    "estimated_alpha_true_lag": estimated_alpha_true_lag,
                    "estimated_alpha_best_lag": estimated_lags[best_lag],
                    "correct_sign": (
                        bool(np.sign(estimated_alpha_true_lag) == np.sign(alpha_gt[target, source, true_lag]))
                        if true_lag_available
                        else False
                    ),
                }
            )

        fp_values = alpha_mean[stable_mask & ~true_mask]
        lag_sign_regime = [row for row in lag_sign_rows if row["regime"] == regime]
        corr = np.corrcoef(alpha_mean.flatten(), alpha_gt.flatten())[0, 1]

        regime_rows.append(
            {
                "regime": regime,
                "alpha_correlation": corr,
                "lag_accuracy": np.mean([row["correct_lag"] for row in lag_sign_regime]),
                "sign_accuracy": np.mean([row["correct_sign"] for row in lag_sign_regime]),
                "true_positives": int(np.sum(stable_mask & true_mask)),
                "false_positives": int(np.sum(stable_mask & ~true_mask)),
                "missed_true_links": int(np.sum(~stable_mask & true_mask)),
                "max_false_positive_strength": 0.0 if len(fp_values) == 0 else np.max(np.abs(fp_values)),
                "mean_false_positive_strength": 0.0 if len(fp_values) == 0 else np.mean(np.abs(fp_values)),
            }
        )

    return (
        prediction_metrics,
        pd.DataFrame(regime_rows),
        pd.DataFrame(lag_sign_rows),
        pd.DataFrame(link_rows),
    )


def temporal_train_test_split(X, y, test_size=0.3, gap=0):
    X = np.asarray(X)
    y = np.asarray(y)
    split = int((1 - test_size) * len(y))
    train_end = max(1, split - gap)
    test_start = min(len(y) - 1, split + gap)
    return X[:train_end], X[test_start:], y[:train_end], y[test_start:], train_end, test_start


def classify_regimes(results_case, regimes, args, case_dir):
    run_keys = [k for k in results_case if k.startswith("run_")]
    alpha_stack = np.stack([results_case[k]["alpha_seq"][1] for k in run_keys], axis=0)
    alpha_seq_mean = alpha_stack.mean(axis=0)

    X = alpha_seq_mean.reshape(alpha_seq_mean.shape[0], -1)
    y = test_window_regimes(regimes, args.window_length, args.train_ratio, args.val_ratio)
    X_train, X_test, y_train, y_test, train_end, test_start = temporal_train_test_split(
        X,
        y,
        test_size=args.classification_test_size,
        gap=args.classification_gap,
    )

    class_dir = case_dir / "classification"
    class_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    models = []

    majority_class = int(np.bincount(y_train.astype(int)).argmax())
    models.append(("majority_baseline", None, np.full_like(y_test, majority_class)))

    if len(np.unique(y_train)) >= 2:
        if "logistic" in args.classification_models:
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            models.append(("logistic", model, model.predict(X_test)))

        if "tree" in args.classification_models:
            model = DecisionTreeClassifier(max_depth=3, random_state=args.seed)
            model.fit(X_train, y_train)
            models.append(("tree", model, model.predict(X_test)))

        if "random_forest" in args.classification_models:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=args.seed,
                class_weight="balanced",
            )
            model.fit(X_train, y_train)
            models.append(("random_forest", model, model.predict(X_test)))

    for model_name, _, y_pred in models:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
                "train_windows": len(y_train),
                "test_windows": len(y_test),
                "train_R0": int(np.sum(y_train == 0)),
                "train_R1": int(np.sum(y_train == 1)),
                "test_R0": int(np.sum(y_test == 0)),
                "test_R1": int(np.sum(y_test == 1)),
                "gap_windows": test_start - train_end,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )

        save_confusion_matrix(cm, class_dir / f"confusion_matrix_{model_name}.pdf", model_name)

    classification_metrics = pd.DataFrame(rows)
    classification_metrics.to_csv(class_dir / "classification_metrics.csv", index=False)
    return classification_metrics


def save_confusion_matrix(cm, save_path, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["R=0", "R=1"])
    ax.set_yticklabels(["R=0", "R=1"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
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
    alpha = tensor_to_numpy(alpha)
    ground_truth_alpha = tensor_to_numpy(ground_truth_alpha)
    n_lags = max(alpha.shape[2], ground_truth_alpha.shape[2])
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
                data = np.flip(data, axis=1)
            data = pad_lag_matrix(data, n_lags)

            abs_max_value = max(np.max(np.abs(data)), 1e-12)
            im = ax.imshow(data, cmap=cmap, vmin=-abs_max_value, vmax=abs_max_value)
            data_rounded = np.round(data, 2)

            for ii in range(data_rounded.shape[0]):
                for jj in range(data_rounded.shape[1]):
                    value = data_rounded[ii, jj]
                    if value != 0:
                        text_color = "black" if abs(value) < 0.2 * abs_max_value else "white"
                        ax.text(jj, ii, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=font_size)

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


def plot_regime_heatmap_canvas(cases, names, title="alpha comparison", cmap="seismic", save_path=None):
    estimated = []
    truths = []

    for _, alpha, ground_truth in cases:
        estimated_alpha, truth_alpha = align_lag_axes(np.flip(np.asarray(alpha), axis=2), ground_truth)
        estimated.append(estimated_alpha)
        truths.append(truth_alpha)

    n_cases = len(cases)
    n_lags = max(max(est.shape[2], gt.shape[2]) for est, gt in zip(estimated, truths))
    estimated = [pad_lag_axis(est, n_lags) for est in estimated]
    truths = [pad_lag_axis(gt, n_lags) for gt in truths]
    n_series = estimated[0].shape[0]
    n_rows = 2 * n_cases
    vmax = max(max(np.max(np.abs(est)), np.max(np.abs(gt))) for est, gt in zip(estimated, truths))
    vmax = max(vmax, 1e-12)

    fig, axes = plt.subplots(n_rows, n_lags, figsize=(3.2 * n_lags, 2.6 * n_rows), constrained_layout=True)

    for case_idx, (case_name, _, _) in enumerate(cases):
        for row_offset, row_label, values in [(0, "estimated", estimated[case_idx]), (1, "ground truth", truths[case_idx])]:
            row_idx = 2 * case_idx + row_offset
            for lag_idx in range(n_lags):
                ax = axes[row_idx, lag_idx]
                matrix = values[:, :, lag_idx]
                im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")

                for target_idx in range(n_series):
                    for source_idx in range(n_series):
                        value = matrix[target_idx, source_idx]
                        if np.round(value, 2) != 0:
                            text_color = "black" if abs(value) < 0.2 * vmax else "white"
                            ax.text(source_idx, target_idx, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

                if row_idx == 0:
                    ax.set_title(f"lag {lag_idx + 1}", fontsize=12)
                if lag_idx == 0:
                    ax.set_ylabel(f"{case_name}\n{row_label}\ntarget", fontsize=10)

                ax.set_xticks(range(n_series))
                ax.set_yticks(range(n_series))
                ax.set_xticklabels(names, fontsize=9)
                ax.set_yticklabels(names, fontsize=9)
                ax.set_xlabel("source", fontsize=9)
                ax.set_xticks(np.arange(-0.5, n_series, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, n_series, 1), minor=True)
                ax.grid(which="minor", color="black", linestyle="-", linewidth=0.4)
                ax.tick_params(which="minor", bottom=False, left=False)

    fig.colorbar(im, ax=axes, shrink=0.85, label=r"$\alpha$")
    fig.suptitle(title, fontsize=14)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_series_plot(series, labels, title, save_path):
    data = tensor_to_numpy(series)
    fig, ax = plt.subplots(figsize=(12, 4))
    for series_idx, label in enumerate(labels):
        ax.plot(data[series_idx], lw=0.7, alpha=0.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_regime_plot(regimes, title, save_path):
    t = np.arange(len(regimes))
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(t, regimes, color="black", s=1)
    ax.set_yticks([0, 1])
    ax.set_xlabel("time")
    ax.set_ylabel(r"$R_t$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_time_grid(
    results_case,
    regimes,
    ground_truth_R0,
    ground_truth_R1,
    interesting_alphas,
    case_title,
    window_size,
    train_ratio,
    val_ratio,
    names,
    save_path,
):
    test_regimes = test_window_regimes(regimes, window_size, train_ratio, val_ratio)
    run_keys = [k for k in results_case if k.startswith("run_")]
    alpha_stack = np.stack([results_case[k]["alpha_seq"][1] for k in run_keys], axis=0)
    alpha_mean_t = np.flip(alpha_stack.mean(axis=0), axis=3)
    alpha_std_t = np.flip(sample_std(alpha_stack, axis=0), axis=3)

    n_plots = len(interesting_alphas)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    t = np.arange(len(test_regimes))
    for ax, (target, source, lag) in zip(axes, interesting_alphas):
        lag_available = lag < alpha_mean_t.shape[3]
        if lag_available:
            y = alpha_mean_t[:, target, source, lag]
            s = alpha_std_t[:, target, source, lag]
        else:
            y = np.full(len(t), np.nan)
            s = np.zeros(len(t))
        gt_R0 = float(ground_truth_R0[target, source, lag])
        gt_R1 = float(ground_truth_R1[target, source, lag])
        gt = np.where(test_regimes == 0, gt_R0, gt_R1)

        if lag_available:
            ax.plot(t, y, label="mean alpha")
            ax.fill_between(t, y - s, y + s, alpha=0.25, color="red", label="+/- std")
        else:
            ax.text(
                0.02,
                0.85,
                f"lag {lag + 1} is outside model window L={alpha_mean_t.shape[3]}",
                transform=ax.transAxes,
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        ax.step(t, gt, where="post", color="black", linewidth=2, linestyle="--", alpha=0.7, label="ground truth alpha")
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_ylabel("alpha")
        ax.set_title(f"{names[source]} -> {names[target]}, lag {lag + 1}")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("test window")
    fig.suptitle(case_title, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_training_bundle(case_dir, results, stats):
    with open(case_dir / "training_results.pkl", "wb") as f:
        pickle.dump({"results": results, "stats": stats}, f)


def make_criterion(loss_name, huber_delta, smooth_l1_beta):
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss(), "MSELoss"
    if loss_name == "mae":
        return nn.L1Loss(), "L1Loss"
    if loss_name == "huber":
        return nn.HuberLoss(delta=huber_delta), f"HuberLoss(delta={huber_delta})"
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss(beta=smooth_l1_beta), f"SmoothL1Loss(beta={smooth_l1_beta})"
    raise ValueError(f"Unknown loss: {loss_name}")


def train_one_case(time_series, args, device, criterion):
    train_config = {
        "verbose": False,
        "device": device,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "scheduler_patience": args.scheduler_patience,
        "early_stopping_modifier": args.early_stopping_modifier,
        "criterion": criterion,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
    }

    return collect_multiple_runs(
        n_runs=args.n_runs,
        time_series=time_series,
        window_size=args.window_length,
        temperature=args.temperature,
        order=args.order,
        config=train_config,
        seed=args.seed,
        verbose=True,
    )


def save_case_outputs(case_dir, case_name, results, stats, regimes, truth, args):
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "metrics").mkdir(exist_ok=True)
    (case_dir / "heatmaps").mkdir(exist_ok=True)
    (case_dir / "time_plots").mkdir(exist_ok=True)

    alpha_stats = regime_stats(
        results,
        regimes,
        args.window_length,
        args.train_ratio,
        args.val_ratio,
        "alpha_seq",
    )
    f_stats = regime_stats(results, regimes, args.window_length, args.train_ratio, args.val_ratio, "f_seq")
    c_stats = regime_stats(results, regimes, args.window_length, args.train_ratio, args.val_ratio, "c_seq")

    global_alpha = stats["alpha"][1]["mean"]
    global_truth = weighted_global_truth(
        truth["R0"],
        truth["R1"],
        regimes,
        args.window_length,
        args.train_ratio,
        args.val_ratio,
    )

    prediction_metrics, regime_metrics, lag_sign_metrics, link_table = task2_metrics(
        results,
        alpha_stats,
        truth["R0"],
        truth["R1"],
        NAMES,
        threshold=args.alpha_threshold,
        c=args.stability_c,
    )
    false_positives = link_table[link_table["link_type"] == "false_positive"].copy()
    classification_metrics = classify_regimes(results, regimes, args, case_dir)

    prediction_metrics.to_csv(case_dir / "metrics" / "prediction_metrics.csv", index=False)
    regime_metrics.to_csv(case_dir / "metrics" / "regime_metrics.csv", index=False)
    lag_sign_metrics.to_csv(case_dir / "metrics" / "lag_sign_metrics.csv", index=False)
    link_table.to_csv(case_dir / "metrics" / "link_table.csv", index=False)
    false_positives.to_csv(case_dir / "metrics" / "false_positives.csv", index=False)

    if not args.no_training_results:
        save_training_bundle(case_dir, results, stats)

    plot_alphas_one_canvas(global_alpha, global_truth, title=r"\alpha", save_path=case_dir / "heatmaps" / "alpha_global.pdf")
    plot_alphas_one_canvas(alpha_stats[0]["mean"], truth["R0"], save_path=case_dir / "heatmaps" / "alpha_R0.pdf")
    plot_alphas_one_canvas(alpha_stats[1]["mean"], truth["R1"], save_path=case_dir / "heatmaps" / "alpha_R1.pdf")

    plot_regime_heatmap_canvas(
        [
            ("Global", global_alpha, global_truth),
            ("R=0", alpha_stats[0]["mean"], truth["R0"]),
            ("R=1", alpha_stats[1]["mean"], truth["R1"]),
        ],
        NAMES,
        title=f"{case_name}: global vs regime alpha",
        save_path=case_dir / "heatmaps" / "alpha_global_R0_R1.pdf",
    )

    plot_alphas_one_canvas(f_stats[0]["mean"], truth["mask_R0"], title=r"f", save_path=case_dir / "heatmaps" / "f_R0.pdf")
    plot_alphas_one_canvas(f_stats[1]["mean"], truth["mask_R1"], title=r"f", save_path=case_dir / "heatmaps" / "f_R1.pdf")
    plot_alphas_one_canvas(c_stats[0]["mean"], truth["mask_R0"], title=r"C", save_path=case_dir / "heatmaps" / "C_R0.pdf")
    plot_alphas_one_canvas(c_stats[1]["mean"], truth["mask_R1"], title=r"C", save_path=case_dir / "heatmaps" / "C_R1.pdf")

    plot_alpha_time_grid(
        results,
        regimes,
        truth["R0"],
        truth["R1"],
        truth["interesting_alphas"],
        case_title=case_name,
        window_size=args.window_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        names=NAMES,
        save_path=case_dir / "time_plots" / "alpha_over_time.pdf",
    )

    return {
        "prediction_metrics": prediction_metrics,
        "regime_metrics": regime_metrics,
        "lag_sign_metrics": lag_sign_metrics,
        "link_table": link_table,
        "classification_metrics": classification_metrics,
    }


def add_metadata(df, **metadata):
    df = df.copy()
    for key, value in reversed(metadata.items()):
        df.insert(0, key, value)
    return df


def run_parameter_set(args, B_min, B_max, sigma_set, loss_name, device):
    group_dir = args.output_dir / b_token(B_min, B_max)
    sigma_dir = group_dir / sigma_token(sigma_set)
    loss_dir = sigma_dir / f"loss_{loss_name}"
    group_dir.mkdir(parents=True, exist_ok=True)
    sigma_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    sigma_name = sigma_set["name"]
    sigma_X1 = sigma_set["sigma_X1"]
    sigma_X2 = sigma_set["sigma_X2"]
    sigma_X3 = sigma_set["sigma_X3"]
    sigma_X4 = sigma_set["sigma_X4"]

    criterion, criterion_name = make_criterion(loss_name, args.huber_delta, args.smooth_l1_beta)
    truth = make_ground_truths(args.window_length)

    cases = time_series_regime_change(
        mean=0,
        ts_length=args.ts_length,
        sigma_X1=sigma_X1,
        sigma_X2=sigma_X2,
        sigma_X3=sigma_X3,
        sigma_X4=sigma_X4,
        B_min=B_min,
        B_max=B_max,
        burn_in=args.burn_in,
        seed=args.seed,
    )

    np.savez_compressed(
        sigma_dir / "generated_series.npz",
        regimes=cases["source_change"][1],
        source_change=tensor_to_numpy(cases["source_change"][0]),
        lag_change=tensor_to_numpy(cases["lag_change"][0]),
        sign_change=tensor_to_numpy(cases["sign_change"][0]),
    )

    time_dir = sigma_dir / "time_series"
    time_dir.mkdir(exist_ok=True)
    save_regime_plot(cases["source_change"][1], f"Regime variable B=({B_min}, {B_max})", time_dir / "regime_variable.pdf")
    for case_name, (time_series, _) in cases.items():
        save_series_plot(time_series, NAMES, case_name, time_dir / f"{case_name}.pdf")

    save_json(
        {
            "B_min": B_min,
            "B_max": B_max,
            "sigma_set": sigma_name,
            "loss": loss_name,
            "criterion": criterion_name,
            "n_runs": args.n_runs,
            "T": args.ts_length,
            "L": args.window_length,
            "burn_in": args.burn_in,
            "seed": args.seed,
            "sigmas": {
                "sigma_X1": sigma_X1,
                "sigma_X2": sigma_X2,
                "sigma_X3": sigma_X3,
                "sigma_X4": sigma_X4,
            },
            "model": {
                "temperature": args.temperature,
                "order": args.order,
                "learning_rate": args.learning_rate,
                "scheduler_patience": args.scheduler_patience,
                "early_stopping_modifier": args.early_stopping_modifier,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "test_ratio": 1 - args.train_ratio - args.val_ratio,
                "device": str(device),
            },
            "alpha_threshold": args.alpha_threshold,
            "stability_c": args.stability_c,
            "classification_test_size": args.classification_test_size,
            "classification_gap": args.classification_gap,
        },
        loss_dir / "config.json",
    )

    all_outputs = {
        "case_summary": [],
        "regime_metrics": [],
        "lag_sign_metrics": [],
        "link_tables": [],
        "classification_metrics": [],
    }

    for case_name, (time_series, regimes) in cases.items():
        print(f"  case: {case_name}")
        results = train_one_case(time_series, args, device, criterion)
        stats = calculate_multiple_run_statistics(results)

        outputs = save_case_outputs(loss_dir / case_name, case_name, results, stats, regimes, truth[case_name], args)

        metadata = {
            "B_min": B_min,
            "B_max": B_max,
            "sigma_set": sigma_name,
            "sigma_X1": sigma_X1,
            "sigma_X2": sigma_X2,
            "sigma_X3": sigma_X3,
            "sigma_X4": sigma_X4,
            "loss": loss_name,
            "case": case_name,
        }

        prediction = outputs["prediction_metrics"].iloc[0].to_dict()
        regime_summary = outputs["regime_metrics"].mean(numeric_only=True).to_dict()
        class_summary = outputs["classification_metrics"]
        logistic_acc = class_summary.loc[class_summary["model"] == "logistic", "accuracy"]
        tree_acc = class_summary.loc[class_summary["model"] == "tree", "accuracy"]

        all_outputs["case_summary"].append(
            {
                **metadata,
                **prediction,
                "mean_alpha_correlation": regime_summary.get("alpha_correlation"),
                "mean_lag_accuracy": regime_summary.get("lag_accuracy"),
                "mean_sign_accuracy": regime_summary.get("sign_accuracy"),
                "false_positives": int(outputs["regime_metrics"]["false_positives"].sum()),
                "missed_true_links": int(outputs["regime_metrics"]["missed_true_links"].sum()),
                "logistic_accuracy": np.nan if logistic_acc.empty else float(logistic_acc.iloc[0]),
                "tree_accuracy": np.nan if tree_acc.empty else float(tree_acc.iloc[0]),
            }
        )

        for key in ["regime_metrics", "lag_sign_metrics", "link_table", "classification_metrics"]:
            summary_key = "link_tables" if key == "link_table" else key
            all_outputs[summary_key].extend(add_metadata(outputs[key], **metadata).to_dict("records"))

        del results, stats, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    pd.DataFrame(all_outputs["case_summary"]).to_csv(loss_dir / "case_summary.csv", index=False)
    return all_outputs


def write_global_summaries(summary_dir, rows):
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows["case_summary"]).to_csv(summary_dir / "all_case_summary.csv", index=False)
    pd.DataFrame(rows["regime_metrics"]).to_csv(summary_dir / "all_regime_metrics.csv", index=False)
    pd.DataFrame(rows["lag_sign_metrics"]).to_csv(summary_dir / "all_lag_sign_metrics.csv", index=False)
    pd.DataFrame(rows["link_tables"]).to_csv(summary_dir / "all_link_tables.csv", index=False)
    pd.DataFrame(rows["classification_metrics"]).to_csv(summary_dir / "all_classification_metrics.csv", index=False)


def parse_b_values(value):
    out = []
    for part in value.split(","):
        if ":" in part:
            left, right = part.split(":", 1)
        elif "-" in part:
            left, right = part.split("-", 1)
        else:
            raise argparse.ArgumentTypeError("B values must look like 25:75,50:150")
        out.append([int(left), int(right)])
    return out


def parse_string_list(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_config_defaults(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "output_dir" in config:
        config["output_dir"] = Path(config["output_dir"])

    return config


def get_sigma_sets(args):
    sigma_sets = getattr(args, "sigma_sets", None)
    selected_names = getattr(args, "sigma_set_names", None)

    if sigma_sets is None:
        sigma_sets = [
            {
                "name": "custom",
                "sigma_X1": args.sigma_X1,
                "sigma_X2": args.sigma_X2,
                "sigma_X3": args.sigma_X3,
                "sigma_X4": args.sigma_X4,
            }
        ]

    out = []
    required = ["sigma_X1", "sigma_X2", "sigma_X3", "sigma_X4"]
    for idx, sigma_set in enumerate(sigma_sets, start=1):
        item = dict(sigma_set)
        item.setdefault("name", f"set_{idx}")
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"Sigma set '{item['name']}' is missing: {missing}")
        out.append(
            {
                "name": item["name"],
                "sigma_X1": float(item["sigma_X1"]),
                "sigma_X2": float(item["sigma_X2"]),
                "sigma_X3": float(item["sigma_X3"]),
                "sigma_X4": float(item["sigma_X4"]),
            }
        )

    if selected_names is not None:
        selected_names = set(selected_names)
        out = [sigma_set for sigma_set in out if sigma_set["name"] in selected_names]
        if not out:
            raise ValueError(f"No sigma sets matched: {sorted(selected_names)}")

    return out


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Run DCIts regime-change synthetic Task 2 pipeline.",
        parents=[config_parser],
    )

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--combo-limit", type=int, default=None)
    parser.add_argument("--B-values", type=parse_b_values, default=[[25, 75], [50, 150], [100, 300], [50, 2000]])
    parser.add_argument("--losses", type=parse_string_list, default=["mse"])
    parser.add_argument(
        "--no-training-results",
        action="store_true",
        help="Do not save training_results.pkl bundles with learned alpha/f/C sequences.",
    )

    parser.add_argument("--ts-length", type=int, default=20000)
    parser.add_argument("--window-length", type=int, default=5)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sigma-X1", type=float, default=0.05)
    parser.add_argument("--sigma-X2", type=float, default=0.10)
    parser.add_argument("--sigma-X3", type=float, default=0.15)
    parser.add_argument("--sigma-X4", type=float, default=0.20)
    parser.add_argument("--sigma-set-names", type=parse_string_list, default=None)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--early-stopping-modifier", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--order", type=parse_int_list, default=[1, 1])
    parser.add_argument("--alpha-threshold", type=float, default=0.04)
    parser.add_argument("--stability-c", type=float, default=1.95)

    parser.add_argument("--classification-test-size", type=float, default=0.3)
    parser.add_argument("--classification-gap", type=int, default=5)
    parser.add_argument("--classification-models", type=parse_string_list, default=["logistic", "tree"])

    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    if config_args.config is not None:
        if not config_args.config.exists():
            parser.error(f"Config file does not exist: {config_args.config}")
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
    sigma_sets = get_sigma_sets(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = args.output_dir / "summary_tables"

    save_json(
        {
            "task": "regime_change",
            "config_file": str(args.config) if args.config is not None else None,
            "B_values": args.B_values,
            "losses": args.losses,
            "sigma_sets": sigma_sets,
            "n_runs": args.n_runs,
            "T": args.ts_length,
            "L": args.window_length,
            "burn_in": args.burn_in,
            "seed": args.seed,
            "alpha_threshold": args.alpha_threshold,
            "stability_c": args.stability_c,
            "device": str(device),
            "output_format": "pdf",
        },
        args.output_dir / "run_config.json",
    )

    jobs = [
        (B_min, B_max, sigma_set, loss_name)
        for B_min, B_max in args.B_values
        for sigma_set in sigma_sets
        for loss_name in args.losses
    ]
    if args.combo_limit is not None:
        jobs = jobs[: args.combo_limit]

    all_rows = {
        "case_summary": [],
        "regime_metrics": [],
        "lag_sign_metrics": [],
        "link_tables": [],
        "classification_metrics": [],
    }

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Total parameter jobs: {len(jobs)}")

    for job_idx, (B_min, B_max, sigma_set, loss_name) in enumerate(jobs, start=1):
        print(f"Job {job_idx}/{len(jobs)}: B=({B_min}, {B_max}), sigma={sigma_set['name']}, loss={loss_name}")
        outputs = run_parameter_set(args, B_min, B_max, sigma_set, loss_name, device)
        for key in all_rows:
            all_rows[key].extend(outputs[key])
        write_global_summaries(summary_dir, all_rows)

    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
