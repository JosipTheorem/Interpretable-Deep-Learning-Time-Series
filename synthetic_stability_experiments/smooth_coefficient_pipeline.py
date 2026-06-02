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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.api import VAR

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import calculate_multiple_run_statistics, collect_multiple_runs


DEFAULT_OUTPUT_DIR = Path("results/smooth_coefficient")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("smooth_coefficient_config.json")
NAMES = ["X1", "X2", "X3", "X4"]
MAIN_TARGET = 1
MAIN_SOURCE = 0
MAIN_LAG = 1
FIXED_TARGET = 2
FIXED_SOURCE = 1
FIXED_LAG = 0


def sample_std(values, axis=None):
    values = np.asarray(values)
    n = values.size if axis is None else values.shape[axis]
    return np.std(values, axis=axis, ddof=1 if n > 1 else 0)


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def safe_name(value):
    return str(value).replace("-", "m").replace(".", "p").replace(" ", "_")


def float_token(value, digits=2):
    return f"{float(value):.{digits}f}".replace("-", "m").replace(".", "p")


def sigma_token(sigma_set):
    return f"sigma_{safe_name(sigma_set['name'])}"


def coefficient_curve(kind, ts_length, **params):
    t = np.arange(ts_length)

    if kind == "sinusoidal":
        return params["a0"] + params["a1"] * np.sin(2 * np.pi * t / params["P"] + params["phi"])

    if kind == "monotonic_drift":
        return params["a_min"] + (params["a_max"] - params["a_min"]) * t / (ts_length - 1)

    if kind == "gaussian_pulse":
        return params["a0"] + params["a1"] * np.exp(-((t - params["t0"]) ** 2) / (2 * params["s"] ** 2))

    if kind == "zero_crossing_sine":
        return params["a1"] * np.sin(2 * np.pi * t / params["P"])

    raise ValueError(f"Unknown coefficient curve: {kind}")


def time_series_smooth_coefficient(
    mean,
    ts_length,
    coefficient_config,
    sigma_X1=0.05,
    sigma_X2=0.10,
    sigma_X3=0.15,
    sigma_X4=0.20,
    burn_in=1000,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    a_t = coefficient_curve(ts_length=ts_length, **coefficient_config)
    a_full = np.concatenate([np.full(burn_in, a_t[0]), a_t])
    total_length = ts_length + burn_in

    time_series = torch.zeros(4, total_length)
    time_series[:, 0:4] = torch.tensor(np.random.normal(0, 1, (4, 4)), dtype=torch.float32)

    for t in range(4, total_length):
        eta_X1 = np.random.normal(mean, sigma_X1)
        eta_X2 = np.random.normal(mean, sigma_X2)
        eta_X3 = np.random.normal(mean, sigma_X3)
        eta_X4 = np.random.normal(mean, sigma_X4)

        time_series[0, t] = 0.5 * time_series[0, t - 1] + eta_X1
        time_series[1, t] = 0.4 * time_series[1, t - 1] + a_full[t] * time_series[0, t - 2] + eta_X2
        time_series[2, t] = 0.5 * time_series[2, t - 1] + 0.6 * time_series[1, t - 1] + eta_X3
        time_series[3, t] = 0.3 * time_series[3, t - 1] + eta_X4

    return time_series[:, burn_in:], a_t


def test_window_times(ts_length, window_size, train_ratio=0.6, val_ratio=0.2):
    train_end = int(train_ratio * ts_length)
    val_end = train_end + int(val_ratio * ts_length)
    return np.arange(val_end + window_size, ts_length)


def make_ground_truth_alpha(a_t, window_length):
    gt = torch.zeros(4, 4, window_length)
    gt[0, 0, 0] = 0.5
    gt[1, 1, 0] = 0.4
    gt[2, 2, 0] = 0.5
    gt[3, 3, 0] = 0.3
    gt[MAIN_TARGET, MAIN_SOURCE, MAIN_LAG] = float(np.mean(a_t))
    gt[FIXED_TARGET, FIXED_SOURCE, FIXED_LAG] = 0.6
    return gt


def true_mask_from_ground_truth(ground_truth_alpha):
    true_mask = tensor_to_numpy(ground_truth_alpha) != 0
    true_mask[MAIN_TARGET, MAIN_SOURCE, MAIN_LAG] = True
    return true_mask


def alpha_over_runs(results, order_idx=1):
    run_keys = [k for k in results if k.startswith("run_")]
    alpha_stack = np.stack([results[k]["alpha_seq"][order_idx] for k in run_keys], axis=0)
    alpha_mean_t = np.flip(alpha_stack.mean(axis=0), axis=3)
    alpha_std_t = np.flip(sample_std(alpha_stack, axis=0), axis=3)
    return alpha_mean_t, alpha_std_t


def safe_corr(x, y):
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def mean_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return values.mean(), sample_std(values)


def make_case_id(case_type, params):
    name = params.get("name")
    if case_type == "sinusoidal":
        core = f"a1_{float_token(params['a1'])}_P_{int(params['P'])}_phi_{float_token(params['phi'])}"
    elif case_type == "monotonic_drift":
        core = f"amin_{float_token(params['a_min'])}_amax_{float_token(params['a_max'])}"
    elif case_type == "gaussian_pulse":
        core = f"a0_{float_token(params['a0'])}_a1_{float_token(params['a1'])}_t0_{float_token(params['t0_frac'])}_s_{float_token(params['s_frac'])}"
    elif case_type == "zero_crossing_sine":
        core = f"a1_{float_token(params['a1'])}_P_{int(params['P'])}"
    else:
        raise ValueError(f"Unknown case type: {case_type}")
    return f"{safe_name(name)}_{core}" if name else core


def build_coefficient_config(case_type, params, ts_length):
    if case_type == "sinusoidal":
        return {
            "kind": "sinusoidal",
            "a0": float(params["a0"]),
            "a1": float(params["a1"]),
            "P": int(params["P"]),
            "phi": float(params["phi"]),
        }
    if case_type == "monotonic_drift":
        return {
            "kind": "monotonic_drift",
            "a_min": float(params["a_min"]),
            "a_max": float(params["a_max"]),
        }
    if case_type == "gaussian_pulse":
        return {
            "kind": "gaussian_pulse",
            "a0": float(params["a0"]),
            "a1": float(params["a1"]),
            "t0": int(float(params["t0_frac"]) * ts_length),
            "s": int(float(params["s_frac"]) * ts_length),
        }
    if case_type == "zero_crossing_sine":
        return {
            "kind": "zero_crossing_sine",
            "a1": float(params["a1"]),
            "P": int(params["P"]),
        }
    raise ValueError(f"Unknown case type: {case_type}")


def get_coefficient_jobs(args):
    case_map = {
        "sinusoidal": args.sinusoidal_cases,
        "monotonic_drift": args.monotonic_drift_cases,
        "gaussian_pulse": args.gaussian_pulse_cases,
        "zero_crossing_sine": args.zero_crossing_sine_cases,
    }

    jobs = []
    selected = set(args.case_types)
    for case_type, cases in case_map.items():
        if case_type not in selected:
            continue
        for params in cases:
            params = dict(params)
            jobs.append(
                {
                    "case_type": case_type,
                    "case_id": make_case_id(case_type, params),
                    "params": params,
                    "coefficient_config": build_coefficient_config(case_type, params, args.ts_length),
                }
            )
    return jobs


def task3_metrics(results_case, stats_case, ground_truth_alpha, alpha_mean_t, alpha_std_t, a_t_test, case_name, threshold=0.04, c=1.95):
    run_keys = [k for k in results_case if k.startswith("run_")]
    mse = np.array([float(results_case[k]["test_loss"]) for k in run_keys])
    rmse = np.sqrt(mse)
    mae = np.array([float(results_case[k].get("test_mae", np.nan)) for k in run_keys])

    mean_mse, std_mse = mean_std(mse)
    mean_rmse, std_rmse = mean_std(rmse)
    mean_mae, std_mae = mean_std(mae)

    prediction_metrics = pd.DataFrame(
        [
            {
                "case": case_name,
                "mean_MSE": mean_mse,
                "std_MSE": std_mse,
                "mean_RMSE": mean_rmse,
                "std_RMSE": std_rmse,
                "mean_MAE": mean_mae,
                "std_MAE": std_mae,
            }
        ]
    )

    alpha_mean = np.flip(stats_case["alpha"][1]["mean"], axis=2)
    alpha_std = np.flip(stats_case["alpha"][1]["std"], axis=2)
    alpha_gt = tensor_to_numpy(ground_truth_alpha).copy()
    true_mask = true_mask_from_ground_truth(ground_truth_alpha)
    stable_mask = (np.abs(alpha_mean) > c * alpha_std) & (np.abs(alpha_mean) >= threshold)

    link_rows = []
    for target, source, lag in np.argwhere(stable_mask | true_mask):
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
                "case": case_name,
                "source": NAMES[source],
                "target": NAMES[target],
                "lag": lag + 1,
                "mean_alpha": alpha_mean[target, source, lag],
                "std_alpha": alpha_std[target, source, lag],
                "ground_truth_alpha": alpha_gt[target, source, lag],
                "stable": stable,
                "true_link": true_link,
                "link_type": link_type,
            }
        )

    lag_sign_rows = []
    for target, source, true_lag in np.argwhere(true_mask):
        estimated_lags = alpha_mean[target, source]
        best_lag = int(np.argmax(np.abs(estimated_lags)))
        gt_value = alpha_gt[target, source, true_lag]
        correct_sign = np.nan if abs(gt_value) <= 1e-12 else np.sign(estimated_lags[true_lag]) == np.sign(gt_value)

        lag_sign_rows.append(
            {
                "case": case_name,
                "source": NAMES[source],
                "target": NAMES[target],
                "true_lag": true_lag + 1,
                "estimated_best_lag": best_lag + 1,
                "correct_lag": best_lag == true_lag,
                "ground_truth_alpha": gt_value,
                "estimated_alpha_true_lag": estimated_lags[true_lag],
                "estimated_alpha_best_lag": estimated_lags[best_lag],
                "correct_sign": correct_sign,
            }
        )

    link_table = pd.DataFrame(link_rows)
    lag_sign_table = pd.DataFrame(lag_sign_rows)

    false_positives = link_table[link_table["link_type"] == "false_positive"].copy()
    if len(false_positives) > 0:
        false_positives["abs_mean_alpha"] = false_positives["mean_alpha"].abs()
        false_positives = false_positives.sort_values("abs_mean_alpha", ascending=False)

    fp_values = alpha_mean[stable_mask & ~true_mask]
    mfp_t = np.abs(alpha_mean_t[:, ~true_mask]).sum(axis=1)

    y = alpha_mean_t[:, MAIN_TARGET, MAIN_SOURCE, MAIN_LAG]
    s = alpha_std_t[:, MAIN_TARGET, MAIN_SOURCE, MAIN_LAG]
    n = min(len(y), len(a_t_test))
    y = y[:n]
    s = s[:n]
    a_true = np.asarray(a_t_test)[:n]

    tracking_rmse = np.sqrt(np.mean((y - a_true) ** 2))
    tracking_corr = safe_corr(y, a_true)
    if np.std(a_true) > 1e-12 and np.std(y) > 1e-12:
        slope, intercept = np.polyfit(y, a_true, 1)
        calibrated = slope * y + intercept
        calibrated_rmse = np.sqrt(np.mean((calibrated - a_true) ** 2))
    else:
        calibrated_rmse = np.nan

    sign_values = pd.Series(lag_sign_table["correct_sign"]).dropna()
    summary_metrics = pd.DataFrame(
        [
            {
                "case": case_name,
                "alpha_correlation_global_mean": safe_corr(alpha_mean, alpha_gt),
                "main_link_tracking_corr": tracking_corr,
                "main_link_RMSE": tracking_rmse,
                "main_link_calibrated_RMSE": calibrated_rmse,
                "lag_accuracy": lag_sign_table["correct_lag"].mean(),
                "sign_accuracy": sign_values.mean() if len(sign_values) > 0 else np.nan,
                "true_positives": int(np.sum(stable_mask & true_mask)),
                "false_positives": int(np.sum(stable_mask & ~true_mask)),
                "missed_true_links": int(np.sum(~stable_mask & true_mask)),
                "max_false_positive_strength": 0.0 if len(fp_values) == 0 else np.max(np.abs(fp_values)),
                "mean_false_positive_strength": 0.0 if len(fp_values) == 0 else np.mean(np.abs(fp_values)),
                "mean_M_FP_t": mfp_t.mean(),
                "max_M_FP_t": mfp_t.max(),
                "mean_alpha_std_all": alpha_std.mean(),
                "mean_alpha_std_true_links": alpha_std[true_mask].mean(),
                "max_alpha_std_true_links": alpha_std[true_mask].max(),
                "mean_main_link_std_t": s.mean(),
                "max_main_link_std_t": s.max(),
            }
        ]
    )

    return prediction_metrics, summary_metrics, lag_sign_table, link_table, false_positives, mfp_t


def run_global_var(time_series, a_t_test, args):
    data = pd.DataFrame(tensor_to_numpy(time_series).T, columns=NAMES)
    L = args.window_length
    train_end = int((args.train_ratio + args.val_ratio) * args.ts_length)
    test_start = train_end + L
    fit = VAR(data.iloc[:train_end]).fit(L)

    pred = []
    for t in range(test_start, len(data)):
        pred.append(fit.forecast(data.iloc[t - L : t].values, steps=1)[0])

    pred = np.asarray(pred)
    true = data.iloc[test_start:].values
    coef = float(fit.coefs[MAIN_LAG, MAIN_TARGET, MAIN_SOURCE])

    metrics = pd.DataFrame(
        [
            {
                "model": "global_var",
                "W": np.nan,
                "MSE": mean_squared_error(true, pred),
                "MAE": mean_absolute_error(true, pred),
                "coef_mean": coef,
                "coef_std": 0.0,
                "coef_corr_true_a": np.nan,
                "coef_rmse_true_a": np.sqrt(np.mean((coef - np.asarray(a_t_test)) ** 2)),
                "true_a_mean": np.mean(a_t_test),
            }
        ]
    )
    return fit, pred, true, metrics


def run_sliding_var(time_series, a_t_test, args, W):
    data = pd.DataFrame(tensor_to_numpy(time_series).T, columns=NAMES)
    L = args.window_length
    train_end = int(args.train_ratio * args.ts_length)
    val_end = train_end + int(args.val_ratio * args.ts_length)
    test_start = val_end + L

    pred = []
    coef = []
    for t in range(test_start, len(data)):
        start = max(0, t - W)
        local_data = data.iloc[start:t]
        if len(local_data) <= L:
            continue
        fit = VAR(local_data).fit(L)
        pred.append(fit.forecast(data.iloc[t - L : t].values, steps=1)[0])
        coef.append(fit.coefs[MAIN_LAG, MAIN_TARGET, MAIN_SOURCE])

    pred = np.asarray(pred)
    coef = np.asarray(coef)
    true = data.iloc[test_start : test_start + len(pred)].values
    a_true = np.asarray(a_t_test)[: len(coef)]

    metrics = pd.DataFrame(
        [
            {
                "model": "sliding_var",
                "W": W,
                "MSE": mean_squared_error(true, pred),
                "MAE": mean_absolute_error(true, pred),
                "coef_mean": coef.mean(),
                "coef_std": sample_std(coef),
                "coef_corr_true_a": safe_corr(coef, a_true),
                "coef_rmse_true_a": np.sqrt(np.mean((coef - a_true) ** 2)),
                "true_a_mean": np.mean(a_true),
            }
        ]
    )
    return pred, true, coef, metrics


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


def save_series_plot(series, save_path, title):
    data = tensor_to_numpy(series)
    fig, ax = plt.subplots(figsize=(12, 4))
    for idx, label in enumerate(NAMES):
        ax.plot(data[idx], lw=0.7, alpha=0.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_single_series_plot(series, label, save_path, title):
    data = tensor_to_numpy(series)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(data, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_alphas_one_canvas(alpha, ground_truth_alpha, cmap="seismic", figsize=(6, 5), font_size=14, title=r"\alpha", space=0.1, cbar_font=11, save_path=None):
    ground_truth_alpha = tensor_to_numpy(ground_truth_alpha)
    n_targets = alpha.shape[0]
    fig, axes = plt.subplots(n_targets, 2, figsize=(figsize[0] * 2, figsize[1] * n_targets), gridspec_kw={"wspace": space, "hspace": space})

    if n_targets == 1:
        axes = np.array([axes])

    for i in range(n_targets):
        for idx, (data, title_prefix, needs_flip) in enumerate(
            zip([alpha[i], ground_truth_alpha[i]], [rf"${title}_{{{i+1},j,l}}$", rf"Ground truth ${title}_{{{i+1},j,l}}$"], [True, False])
        ):
            ax = axes[i, idx]
            if needs_flip:
                data = np.flip(data, axis=1)

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


def plot_smooth_metric_lag_comparison(smooth_cases, case_order, metric="alpha", alpha_key=1, cmap="seismic", save_path=None):
    estimated_by_case = []
    truth_by_case = []
    case_titles = []

    for case_name in case_order:
        case = smooth_cases[case_name]
        if metric == "alpha":
            estimated = case["stats"]["alpha"][alpha_key]["mean"]
            truth = tensor_to_numpy(case["ground_truth_alpha"])
            colorbar_label = r"$\alpha$"
        elif metric == "focuser":
            estimated = case["stats"]["f"][alpha_key]["mean"]
            truth = tensor_to_numpy(case["alpha_mask"])
            colorbar_label = "Focuser"
        elif metric == "coefficient":
            estimated = case["stats"]["c"][alpha_key]["mean"]
            truth = tensor_to_numpy(case["alpha_mask"])
            colorbar_label = "Coefficient"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        estimated_by_case.append(np.flip(estimated, axis=2))
        truth_by_case.append(truth)
        case_titles.append(case["title"])

    n_cases = len(case_order)
    n_lags = estimated_by_case[0].shape[2]
    n_series = estimated_by_case[0].shape[0]
    n_rows = 2 * n_cases
    vmax = max(max(np.max(np.abs(est)), np.max(np.abs(truth))) for est, truth in zip(estimated_by_case, truth_by_case))
    vmax = max(vmax, 1e-12)

    fig, axes = plt.subplots(n_rows, n_lags, figsize=(3.2 * n_lags, 2.6 * n_rows), constrained_layout=True)
    if n_rows == 2:
        axes = np.asarray(axes)

    for case_idx, (case_title, estimated, truth) in enumerate(zip(case_titles, estimated_by_case, truth_by_case)):
        for row_offset, row_label in [(0, "estimated"), (1, "ground truth")]:
            row_idx = 2 * case_idx + row_offset
            data = estimated if row_offset == 0 else truth
            for lag_idx in range(n_lags):
                ax = axes[row_idx, lag_idx]
                matrix = data[:, :, lag_idx]
                im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
                data_rounded = np.round(matrix, 2)
                for target_idx in range(n_series):
                    for source_idx in range(n_series):
                        value = data_rounded[target_idx, source_idx]
                        if value != 0:
                            text_color = "black" if abs(value) < 0.2 * vmax else "white"
                            ax.text(source_idx, target_idx, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

                if row_idx == 0:
                    ax.set_title(f"lag {lag_idx + 1}", fontsize=12)
                if lag_idx == 0:
                    ax.set_ylabel(f"{case_title}\n{row_label}\ntarget", fontsize=10)

                ax.set_xticks(range(n_series))
                ax.set_yticks(range(n_series))
                ax.set_xticklabels(NAMES, fontsize=9)
                ax.set_yticklabels(NAMES, fontsize=9)
                ax.set_xlabel("source", fontsize=9)
                ax.set_xticks(np.arange(-0.5, n_series, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, n_series, 1), minor=True)
                ax.grid(which="minor", color="black", linestyle="-", linewidth=0.4)
                ax.tick_params(which="minor", bottom=False, left=False)

    fig.colorbar(im, ax=axes, shrink=0.85, label=colorbar_label)
    fig.suptitle(f"{metric}: estimated vs ground truth", fontsize=14)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_alpha_vs_true(alpha_mean_t, alpha_std_t, a_true, target, source, lag, title, save_path):
    y = alpha_mean_t[:, target, source, lag]
    s = alpha_std_t[:, target, source, lag]
    n = min(len(y), len(a_true))
    y = y[:n]
    s = s[:n]
    a_true = np.asarray(a_true)[:n]
    t = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, y, label="mean alpha")
    ax.fill_between(t, y - s, y + s, alpha=0.25, color="red", label="+/- std")
    ax.plot(t, a_true, color="black", linewidth=2, linestyle="--", alpha=0.7, label="true")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("test window")
    ax.set_ylabel("coefficient")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_false_positive_mass(mfp_t, title, save_path):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(mfp_t)
    ax.set_xlabel("test window")
    ax.set_ylabel("M_FP(t)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_var_comparison(a_t_test, global_coef, sliding_outputs, title, save_path):
    fig, ax = plt.subplots(figsize=(14, 4))
    for W, coef in sliding_outputs.items():
        n = min(len(coef), len(a_t_test))
        ax.plot(np.arange(n), coef[:n], label=f"sliding VAR, W={W}")
    ax.axhline(global_coef, color="red", linestyle="--", label="global VAR")
    ax.plot(a_t_test, color="black", linestyle="--", alpha=0.7, linewidth=2, label="true a(t)")
    ax.set_xlabel("test window")
    ax.set_ylabel("coefficient")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_training_bundle(case_dir, results, stats):
    with open(case_dir / "training_results.pkl", "wb") as f:
        pickle.dump({"results": results, "stats": stats}, f)


def add_metadata(df, **metadata):
    df = df.copy()
    for key, value in reversed(metadata.items()):
        df.insert(0, key, value)
    return df


def save_case_outputs(case_dir, case_label, time_series, a_t, a_t_test, ground_truth_alpha, results, stats, args):
    case_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["metrics", "time_series", "heatmaps", "coefficient_tracking", "var"]:
        (case_dir / subdir).mkdir(exist_ok=True)

    alpha_mean_t, alpha_std_t = alpha_over_runs(results)
    alpha_mask = torch.tensor(true_mask_from_ground_truth(ground_truth_alpha).astype(float), dtype=torch.float32)

    prediction_metrics, summary_metrics, lag_sign_table, link_table, false_positives, mfp_t = task3_metrics(
        results,
        stats,
        ground_truth_alpha,
        alpha_mean_t,
        alpha_std_t,
        a_t_test,
        case_label,
        threshold=args.alpha_threshold,
        c=args.stability_c,
    )

    prediction_metrics.to_csv(case_dir / "metrics" / "prediction_metrics.csv", index=False)
    summary_metrics.to_csv(case_dir / "metrics" / "summary_metrics.csv", index=False)
    lag_sign_table.to_csv(case_dir / "metrics" / "lag_sign_table.csv", index=False)
    link_table.to_csv(case_dir / "metrics" / "link_table.csv", index=False)
    false_positives.to_csv(case_dir / "metrics" / "false_positives.csv", index=False)

    if not args.no_training_results:
        save_training_bundle(case_dir, results, stats)

    save_series_plot(time_series, case_dir / "time_series" / "all_X.pdf", f"{case_label}: generated time series")
    for idx, name in enumerate(NAMES):
        save_single_series_plot(time_series[idx], name, case_dir / "time_series" / f"{name}.pdf", f"{case_label}: {name}")
    save_single_series_plot(a_t, "a(t)", case_dir / "time_series" / "true_a_t.pdf", f"{case_label}: true a(t)")

    plot_alphas_one_canvas(stats["alpha"][1]["mean"], ground_truth_alpha, title=r"\alpha", save_path=case_dir / "heatmaps" / "alpha_global.pdf")
    plot_alphas_one_canvas(stats["f"][1]["mean"], alpha_mask, title=r"f", save_path=case_dir / "heatmaps" / "f_global.pdf")
    plot_alphas_one_canvas(stats["c"][1]["mean"], alpha_mask, title=r"C", save_path=case_dir / "heatmaps" / "C_global.pdf")

    smooth_case = {
        "case": {
            "title": case_label,
            "stats": stats,
            "ground_truth_alpha": ground_truth_alpha,
            "alpha_mask": alpha_mask,
        }
    }
    plot_smooth_metric_lag_comparison(smooth_case, ["case"], "alpha", save_path=case_dir / "heatmaps" / "alpha_pipeline_canvas.pdf")
    plot_smooth_metric_lag_comparison(smooth_case, ["case"], "focuser", save_path=case_dir / "heatmaps" / "focuser_pipeline_canvas.pdf")
    plot_smooth_metric_lag_comparison(smooth_case, ["case"], "coefficient", save_path=case_dir / "heatmaps" / "coefficient_pipeline_canvas.pdf")

    plot_alpha_vs_true(
        alpha_mean_t,
        alpha_std_t,
        a_t_test,
        MAIN_TARGET,
        MAIN_SOURCE,
        MAIN_LAG,
        f"{case_label}: X1 -> X2, lag 2",
        case_dir / "coefficient_tracking" / "dcits_alpha_vs_true.pdf",
    )
    plot_alpha_vs_true(
        alpha_mean_t,
        alpha_std_t,
        np.full_like(a_t_test, 0.6),
        FIXED_TARGET,
        FIXED_SOURCE,
        FIXED_LAG,
        f"{case_label}: fixed X2 -> X3, lag 1",
        case_dir / "coefficient_tracking" / "fixed_alpha_vs_true.pdf",
    )
    plot_false_positive_mass(mfp_t, f"{case_label}: false positive mass", case_dir / "coefficient_tracking" / "false_positive_mass.pdf")

    global_fit, global_pred, global_true, global_metrics = run_global_var(time_series, a_t_test, args)
    with open(case_dir / "var" / "global_var.pkl", "wb") as f:
        pickle.dump(global_fit, f)

    var_rows = [global_metrics]
    sliding_outputs = {}
    for W in args.var_windows:
        sliding_pred, sliding_true, sliding_coef, sliding_metrics = run_sliding_var(time_series, a_t_test, args, W)
        np.savez_compressed(
            case_dir / "var" / f"sliding_var_W_{W}.npz",
            pred=sliding_pred,
            true=sliding_true,
            coef=sliding_coef,
        )
        sliding_outputs[W] = sliding_coef
        var_rows.append(sliding_metrics)

    var_metrics = pd.concat(var_rows, ignore_index=True)
    var_metrics.to_csv(case_dir / "metrics" / "var_metrics.csv", index=False)
    plot_var_comparison(
        a_t_test,
        float(global_metrics.loc[0, "coef_mean"]),
        sliding_outputs,
        f"{case_label}: VAR comparison",
        case_dir / "coefficient_tracking" / "var_comparison.pdf",
    )

    return {
        "prediction_metrics": prediction_metrics,
        "summary_metrics": summary_metrics,
        "lag_sign_table": lag_sign_table,
        "link_table": link_table,
        "false_positives": false_positives,
        "var_metrics": var_metrics,
    }


def run_parameter_set(args, coefficient_job, sigma_set, loss_name, device):
    case_type = coefficient_job["case_type"]
    case_id = coefficient_job["case_id"]
    coefficient_config = coefficient_job["coefficient_config"]
    params = coefficient_job["params"]

    case_root = args.output_dir / case_type / case_id
    sigma_dir = case_root / sigma_token(sigma_set)
    loss_dir = sigma_dir / f"loss_{loss_name}"
    loss_dir.mkdir(parents=True, exist_ok=True)

    sigma_name = sigma_set["name"]
    sigma_X1 = sigma_set["sigma_X1"]
    sigma_X2 = sigma_set["sigma_X2"]
    sigma_X3 = sigma_set["sigma_X3"]
    sigma_X4 = sigma_set["sigma_X4"]

    criterion, criterion_name = make_criterion(loss_name, args.huber_delta, args.smooth_l1_beta)

    time_series, a_t = time_series_smooth_coefficient(
        mean=0,
        ts_length=args.ts_length,
        coefficient_config=coefficient_config,
        sigma_X1=sigma_X1,
        sigma_X2=sigma_X2,
        sigma_X3=sigma_X3,
        sigma_X4=sigma_X4,
        burn_in=args.burn_in,
        seed=args.seed,
    )
    t_test = test_window_times(args.ts_length, args.window_length, args.train_ratio, args.val_ratio)
    a_t_test = a_t[t_test]
    ground_truth_alpha = make_ground_truth_alpha(a_t_test, args.window_length)

    np.savez_compressed(
        sigma_dir / "generated_series.npz",
        time_series=tensor_to_numpy(time_series),
        a_t=a_t,
        a_t_test=a_t_test,
        t_test=t_test,
    )

    save_json(
        {
            "case_type": case_type,
            "case_id": case_id,
            "case_parameters": params,
            "coefficient_config": coefficient_config,
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
            "var_windows": args.var_windows,
        },
        loss_dir / "config.json",
    )

    print(f"  training: {case_type}/{case_id}, sigma={sigma_name}, loss={loss_name}")
    results = train_one_case(time_series, args, device, criterion)
    stats = calculate_multiple_run_statistics(results)
    outputs = save_case_outputs(loss_dir, case_id, time_series, a_t, a_t_test, ground_truth_alpha, results, stats, args)

    metadata = {
        "case_type": case_type,
        "case_id": case_id,
        "sigma_set": sigma_name,
        "sigma_X1": sigma_X1,
        "sigma_X2": sigma_X2,
        "sigma_X3": sigma_X3,
        "sigma_X4": sigma_X4,
        "loss": loss_name,
    }
    metadata.update({f"param_{key}": value for key, value in params.items() if key != "name"})

    case_summary = {
        **metadata,
        **outputs["prediction_metrics"].iloc[0].to_dict(),
        **outputs["summary_metrics"].iloc[0].to_dict(),
    }
    global_var = outputs["var_metrics"][outputs["var_metrics"]["model"] == "global_var"].iloc[0]
    case_summary.update(
        {
            "global_VAR_MSE": global_var["MSE"],
            "global_VAR_MAE": global_var["MAE"],
            "global_VAR_coef": global_var["coef_mean"],
        }
    )
    for _, row in outputs["var_metrics"][outputs["var_metrics"]["model"] == "sliding_var"].iterrows():
        W = int(row["W"])
        case_summary[f"sliding_VAR_W_{W}_MSE"] = row["MSE"]
        case_summary[f"sliding_VAR_W_{W}_MAE"] = row["MAE"]
        case_summary[f"sliding_VAR_W_{W}_coef_corr"] = row["coef_corr_true_a"]
        case_summary[f"sliding_VAR_W_{W}_coef_rmse"] = row["coef_rmse_true_a"]

    all_outputs = {
        "case_summary": [case_summary],
        "prediction_metrics": add_metadata(outputs["prediction_metrics"], **metadata).to_dict("records"),
        "summary_metrics": add_metadata(outputs["summary_metrics"], **metadata).to_dict("records"),
        "lag_sign_tables": add_metadata(outputs["lag_sign_table"], **metadata).to_dict("records"),
        "link_tables": add_metadata(outputs["link_table"], **metadata).to_dict("records"),
        "false_positives": add_metadata(outputs["false_positives"], **metadata).to_dict("records"),
        "var_metrics": add_metadata(outputs["var_metrics"], **metadata).to_dict("records"),
    }

    del results, stats, outputs, time_series
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return all_outputs


def write_global_summaries(summary_dir, rows):
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows["case_summary"]).to_csv(summary_dir / "all_case_summary.csv", index=False)
    pd.DataFrame(rows["prediction_metrics"]).to_csv(summary_dir / "all_prediction_metrics.csv", index=False)
    pd.DataFrame(rows["summary_metrics"]).to_csv(summary_dir / "all_summary_metrics.csv", index=False)
    pd.DataFrame(rows["lag_sign_tables"]).to_csv(summary_dir / "all_lag_sign_tables.csv", index=False)
    pd.DataFrame(rows["link_tables"]).to_csv(summary_dir / "all_link_tables.csv", index=False)
    pd.DataFrame(rows["false_positives"]).to_csv(summary_dir / "all_false_positives.csv", index=False)
    pd.DataFrame(rows["var_metrics"]).to_csv(summary_dir / "all_var_metrics.csv", index=False)


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
        description="Run DCIts smooth time-varying coefficient synthetic Task 3 pipeline.",
        parents=[config_parser],
    )

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--combo-limit", type=int, default=None)

    parser.add_argument("--case-types", type=parse_string_list, default=["sinusoidal", "monotonic_drift", "gaussian_pulse", "zero_crossing_sine"])
    parser.add_argument("--losses", type=parse_string_list, default=["mse"])
    parser.add_argument("--var-windows", type=parse_int_list, default=[20, 100])
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

    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.set_defaults(
        sinusoidal_cases=[
            {"a0": 0.50, "a1": 0.25, "P": 500, "phi": 0.0, "name": "baseline"},
            {"a0": 0.50, "a1": 0.10, "P": 500, "phi": 0.0, "name": "low_amplitude"},
            {"a0": 0.50, "a1": 0.40, "P": 500, "phi": 0.0, "name": "high_amplitude"},
            {"a0": 0.50, "a1": 0.25, "P": 100, "phi": 0.0, "name": "fast_period"},
            {"a0": 0.50, "a1": 0.25, "P": 2000, "phi": 0.0, "name": "slow_period"},
            {"a0": 0.50, "a1": 0.25, "P": 500, "phi": 1.5707963267948966, "name": "phase_pi_over_2"},
        ],
        monotonic_drift_cases=[{"a_min": 0.10, "a_max": 0.90, "name": "default"}],
        gaussian_pulse_cases=[
            {"a0": 0.10, "a1": 0.75, "t0_frac": 0.60, "s_frac": 0.30, "name": "wide_mid"},
            {"a0": 0.10, "a1": 0.75, "t0_frac": 0.50, "s_frac": 0.15, "name": "train_peak"},
            {"a0": 0.10, "a1": 0.75, "t0_frac": 0.80, "s_frac": 0.15, "name": "test_border_peak"},
        ],
        zero_crossing_sine_cases=[{"a1": 0.60, "P": 500, "name": "default"}],
    )

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
    coefficient_jobs = get_coefficient_jobs(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = args.output_dir / "summary_tables"

    jobs = [
        (coefficient_job, sigma_set, loss_name)
        for coefficient_job in coefficient_jobs
        for sigma_set in sigma_sets
        for loss_name in args.losses
    ]
    if args.combo_limit is not None:
        jobs = jobs[: args.combo_limit]

    save_json(
        {
            "task": "smooth_coefficient",
            "config_file": str(args.config) if args.config is not None else None,
            "case_types": args.case_types,
            "coefficient_jobs": coefficient_jobs,
            "sigma_sets": sigma_sets,
            "losses": args.losses,
            "var_windows": args.var_windows,
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

    all_rows = {
        "case_summary": [],
        "prediction_metrics": [],
        "summary_metrics": [],
        "lag_sign_tables": [],
        "link_tables": [],
        "false_positives": [],
        "var_metrics": [],
    }

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Total parameter jobs: {len(jobs)}")

    for job_idx, (coefficient_job, sigma_set, loss_name) in enumerate(jobs, start=1):
        print(
            f"Job {job_idx}/{len(jobs)}: "
            f"{coefficient_job['case_type']}/{coefficient_job['case_id']}, "
            f"sigma={sigma_set['name']}, loss={loss_name}"
        )
        outputs = run_parameter_set(args, coefficient_job, sigma_set, loss_name, device)
        for key in all_rows:
            all_rows[key].extend(outputs[key])
        write_global_summaries(summary_dir, all_rows)

    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
