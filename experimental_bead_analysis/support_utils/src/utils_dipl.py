import time
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.dcits import DCITS
from src.utils import *  # Reuse plotting/helpers from the original module.


def summarize_interpretability(focus_dict: Dict[int, np.ndarray], coefficient_dict: Dict[int, np.ndarray]) -> Dict:
    """Summarize focus/coefficient outputs into alpha statistics."""
    summary = {
        "focus": focus_dict,
        "coefficients": coefficient_dict,
        "alpha_mean": {},
        "alpha_std": {},
        "alpha_bias": None,
        "alpha_bias_std": None,
        "f_means": {},
        "c_means": {},
    }

    for order_idx in focus_dict.keys():
        alpha_values = focus_dict[order_idx] * coefficient_dict[order_idx]

        if order_idx == 0:
            summary["alpha_bias"] = alpha_values.mean(0)
            summary["alpha_bias_std"] = alpha_values.std(0)
        else:
            summary["alpha_mean"][order_idx] = alpha_values.mean(0)
            summary["alpha_std"][order_idx] = alpha_values.std(0)
            summary["f_means"][order_idx] = focus_dict[order_idx].mean(0)
            summary["c_means"][order_idx] = coefficient_dict[order_idx].mean(0)

    return summary


def _evaluate_loader(model, loader, criterion, device):
    """Evaluate one split and collect interpretability outputs."""
    split_focus = {i: [] for i in range(len(model.order))}
    split_coefficients = {i: [] for i in range(len(model.order))}
    split_loss = 0.0
    split_predictions = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device).unsqueeze(1)
            targets = targets.to(device)

            outputs, focus_weights, coefficients = model(inputs)
            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                raise ValueError("Loss became NaN during split evaluation.")

            split_loss += loss.item() * inputs.size(0)
            split_predictions.append(outputs.detach().cpu())

            if model.bias:
                split_focus[0].append(focus_weights[0].cpu().numpy())
                split_coefficients[0].append(coefficients[0].cpu().numpy())

            for order_idx, enabled in enumerate(model.order[1:], start=1):
                if enabled == 1:
                    split_focus[order_idx].append(focus_weights[order_idx].cpu().numpy())
                    split_coefficients[order_idx].append(coefficients[order_idx].cpu().numpy())

    split_loss /= len(loader.dataset)

    focus_dict = {i: np.concatenate(split_focus[i], axis=0) for i in split_focus if split_focus[i]}
    coefficient_dict = {
        i: np.concatenate(split_coefficients[i], axis=0)
        for i in split_coefficients
        if split_coefficients[i]
    }
    predictions = torch.cat(split_predictions, dim=0)

    return split_loss, focus_dict, coefficient_dict, predictions


def train_and_evaluate(
    time_series,
    window_size,
    temperature,
    order=[1, 1],
    config=None,
):
    """
    Train DCIts and return split-aware interpretability summaries.

    Returns:
        tuple:
            test_loss,
            train_losses,
            val_losses,
            split_results,
            debug_info,
            model
    """
    default_config = {
        "seed": 1000,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "epochs": 100,
        "device": None,
        "scheduler_patience": 5,
        "verbose": False,
        "debug_mode": False,
        "memory_callback": None,
        "criterion": None,
        "early_stopping_modifier": 4,
        "min_epochs": 10,
        "min_learning_rate": 1e-6,
    }

    if config is not None:
        default_config.update(config)
    config = default_config

    if config["device"] is None:
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["criterion"] is None:
        config["criterion"] = nn.MSELoss()

    debug_info = {
        "epoch_details": [],
        "early_stopping_trigger": None,
        "lr_changes": [],
        "batch_losses": [] if config["debug_mode"] else None,
        "gradient_norms": [] if config["debug_mode"] else None,
    }

    train_series, val_series, test_series = split_time_series(
        time_series,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        window_size=window_size,
    )

    train_windowed_dataset = create_windowed_dataset(train_series, window_size)
    val_windowed_dataset = create_windowed_dataset(val_series, window_size)
    test_windowed_dataset = create_windowed_dataset(test_series, window_size)

    train_inputs = train_windowed_dataset[:, :, :-1].float()
    train_targets = train_windowed_dataset[:, :, -1].float()

    val_inputs = val_windowed_dataset[:, :, :-1].float()
    val_targets = val_windowed_dataset[:, :, -1].float()

    test_inputs = test_windowed_dataset[:, :, :-1].float()
    test_targets = test_windowed_dataset[:, :, -1].float()

    train_gen = torch.Generator().manual_seed(config["seed"])
    val_gen = torch.Generator().manual_seed(config["seed"])
    test_gen = torch.Generator().manual_seed(config["seed"])

    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=train_gen,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        generator=val_gen,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        generator=test_gen,
        num_workers=0,
    )

    if len(train_loader.dataset) == 0:
        raise ValueError("Training dataset is empty. Cannot proceed with training.")
    if len(val_loader.dataset) == 0:
        raise ValueError("Validation dataset is empty. Cannot proceed with training.")
    if len(test_loader.dataset) == 0:
        raise ValueError("Test dataset is empty. Cannot proceed with evaluation.")

    model = DCITS(
        no_of_timeseries=time_series.shape[0],
        window_length=window_size,
        temperature=temperature,
        order=order,
    )
    model = model.to(config["device"])

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config["scheduler_patience"],
        min_lr=config["min_learning_rate"],
    )

    early_stopping_patience = int(config["scheduler_patience"] * config["early_stopping_modifier"])
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    def compute_gradient_norm():
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        batch_losses_epoch = [] if config["debug_mode"] else None
        gradient_norms_epoch = [] if config["debug_mode"] else None

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if config["memory_callback"] is not None:
                config["memory_callback"]()

            inputs = inputs.to(config["device"]).unsqueeze(1)
            targets = targets.to(config["device"])

            optimizer.zero_grad()
            outputs, _, _ = model(inputs)
            loss = config["criterion"](outputs, targets)
            if torch.isnan(loss):
                raise ValueError(f"Loss is NaN at epoch {epoch + 1}, batch {batch_idx + 1}.")
            loss.backward()

            if config["debug_mode"]:
                grad_norm = compute_gradient_norm()
                gradient_norms_epoch.append(grad_norm)
                debug_info["gradient_norms"].append(grad_norm)
                batch_losses_epoch.append(loss.item())

            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(config["device"]).unsqueeze(1)
                targets = targets.to(config["device"])
                outputs, _, _ = model(inputs)
                loss = config["criterion"](outputs, targets)
                if torch.isnan(loss):
                    raise ValueError(f"Validation loss is NaN at epoch {epoch + 1}.")
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        if old_lr != new_lr:
            debug_info["lr_changes"].append((epoch + 1, old_lr, new_lr))

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "learning_rate": new_lr,
        }
        if config["debug_mode"]:
            epoch_info.update(
                {
                    "batch_losses": batch_losses_epoch,
                    "gradient_norms": gradient_norms_epoch,
                }
            )
        debug_info["epoch_details"].append(epoch_info)

        if config["verbose"]:
            print(
                f"Epoch {epoch + 1}/{config['epochs']}, "
                f"Train Loss: {epoch_loss:.6e}, "
                f"Val Loss: {val_loss:.6e}, "
                f"LR: {new_lr:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if (epoch + 1) >= config["min_epochs"] and epochs_no_improve >= early_stopping_patience:
                debug_info["early_stopping_trigger"] = epoch + 1
                if config["verbose"]:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    eval_loaders = {
        "train": DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0),
        "val": DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0),
        "test": DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0),
    }
    split_tensors = {
        "train": {"inputs": train_inputs, "targets": train_targets},
        "val": {"inputs": val_inputs, "targets": val_targets},
        "test": {"inputs": test_inputs, "targets": test_targets},
    }

    split_results = {}
    for split_name, loader in eval_loaders.items():
        split_loss, focus_dict, coefficient_dict, predictions = _evaluate_loader(
            model=model,
            loader=loader,
            criterion=config["criterion"],
            device=config["device"],
        )
        split_summary = summarize_interpretability(focus_dict, coefficient_dict)
        split_summary.update(
            {
                "loss": split_loss,
                "inputs": split_tensors[split_name]["inputs"],
                "targets": split_tensors[split_name]["targets"],
                "predictions": predictions,
            }
        )
        split_results[split_name] = split_summary

    test_loss = split_results["test"]["loss"]

    if config["verbose"]:
        print(f"Final Test Loss: {test_loss:.6f}")

    return test_loss, train_losses, val_losses, split_results, debug_info, model


def create_supervised_windowed_dataset(input_series, target_series, window_size):
    """
    Create windows where an input window predicts a separately defined target series.

    If input_series contains X[0], ..., X[T-2] and target_series contains
    |X[1]-X[0]|, ..., |X[T-1]-X[T-2]|, then window X[0:5] predicts |X[5]-X[4]|.
    """
    if input_series.shape != target_series.shape:
        raise ValueError("input_series and target_series must have the same shape.")

    input_series = torch.as_tensor(input_series, dtype=torch.float32)
    target_series = torch.as_tensor(target_series, dtype=torch.float32)

    _, series_length = input_series.shape
    num_windows = series_length - window_size + 1
    if num_windows <= 0:
        raise ValueError("Not enough data to create supervised windows.")

    inputs = []
    targets = []
    for start in range(num_windows):
        inputs.append(input_series[:, start:start + window_size].unsqueeze(0))
        targets.append(target_series[:, start + window_size - 1].unsqueeze(0))

    return torch.cat(inputs, dim=0), torch.cat(targets, dim=0)


def train_and_evaluate_supervised(
    input_series,
    target_series,
    window_size,
    temperature,
    order=[1, 1],
    config=None,
):
    """
    Train DCIts with separate input and target time series.

    This keeps the DCIts architecture unchanged, so input_series and target_series
    must have the same number of series. Example use:
        input_series = raw_xy[:, :-1]
        target_series = abs(diff(raw_xy))
    """
    default_config = {
        "seed": 1000,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "epochs": 100,
        "device": None,
        "scheduler_patience": 5,
        "verbose": False,
        "debug_mode": False,
        "memory_callback": None,
        "criterion": None,
        "early_stopping_modifier": 4,
        "min_epochs": 10,
        "min_learning_rate": 1e-6,
    }

    if config is not None:
        default_config.update(config)
    config = default_config

    if config["device"] is None:
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["criterion"] is None:
        config["criterion"] = nn.MSELoss()

    input_series = torch.as_tensor(input_series, dtype=torch.float32)
    target_series = torch.as_tensor(target_series, dtype=torch.float32)

    if input_series.shape != target_series.shape:
        raise ValueError("input_series and target_series must have the same shape.")

    debug_info = {
        "epoch_details": [],
        "early_stopping_trigger": None,
        "lr_changes": [],
        "batch_losses": [] if config["debug_mode"] else None,
        "gradient_norms": [] if config["debug_mode"] else None,
    }

    train_input_series, val_input_series, test_input_series = split_time_series(
        input_series,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        window_size=window_size,
    )
    train_target_series, val_target_series, test_target_series = split_time_series(
        target_series,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        window_size=window_size,
    )

    train_inputs, train_targets = create_supervised_windowed_dataset(
        train_input_series,
        train_target_series,
        window_size,
    )
    val_inputs, val_targets = create_supervised_windowed_dataset(
        val_input_series,
        val_target_series,
        window_size,
    )
    test_inputs, test_targets = create_supervised_windowed_dataset(
        test_input_series,
        test_target_series,
        window_size,
    )

    train_inputs = train_inputs.float()
    train_targets = train_targets.float()
    val_inputs = val_inputs.float()
    val_targets = val_targets.float()
    test_inputs = test_inputs.float()
    test_targets = test_targets.float()

    train_gen = torch.Generator().manual_seed(config["seed"])
    val_gen = torch.Generator().manual_seed(config["seed"])
    test_gen = torch.Generator().manual_seed(config["seed"])

    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=train_gen,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        generator=val_gen,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        generator=test_gen,
        num_workers=0,
    )

    if len(train_loader.dataset) == 0:
        raise ValueError("Training dataset is empty. Cannot proceed with training.")
    if len(val_loader.dataset) == 0:
        raise ValueError("Validation dataset is empty. Cannot proceed with training.")
    if len(test_loader.dataset) == 0:
        raise ValueError("Test dataset is empty. Cannot proceed with evaluation.")

    model = DCITS(
        no_of_timeseries=input_series.shape[0],
        window_length=window_size,
        temperature=temperature,
        order=order,
    )
    model = model.to(config["device"])

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config["scheduler_patience"],
        min_lr=config["min_learning_rate"],
    )

    early_stopping_patience = int(config["scheduler_patience"] * config["early_stopping_modifier"])
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    def compute_gradient_norm():
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        batch_losses_epoch = [] if config["debug_mode"] else None
        gradient_norms_epoch = [] if config["debug_mode"] else None

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if config["memory_callback"] is not None:
                config["memory_callback"]()

            inputs = inputs.to(config["device"]).unsqueeze(1)
            targets = targets.to(config["device"])

            optimizer.zero_grad()
            outputs, _, _ = model(inputs)
            loss = config["criterion"](outputs, targets)
            if torch.isnan(loss):
                raise ValueError(f"Loss is NaN at epoch {epoch + 1}, batch {batch_idx + 1}.")
            loss.backward()

            if config["debug_mode"]:
                grad_norm = compute_gradient_norm()
                gradient_norms_epoch.append(grad_norm)
                debug_info["gradient_norms"].append(grad_norm)
                batch_losses_epoch.append(loss.item())

            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(config["device"]).unsqueeze(1)
                targets = targets.to(config["device"])
                outputs, _, _ = model(inputs)
                loss = config["criterion"](outputs, targets)
                if torch.isnan(loss):
                    raise ValueError(f"Validation loss is NaN at epoch {epoch + 1}.")
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        if old_lr != new_lr:
            debug_info["lr_changes"].append((epoch + 1, old_lr, new_lr))

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "learning_rate": new_lr,
        }
        if config["debug_mode"]:
            epoch_info.update(
                {
                    "batch_losses": batch_losses_epoch,
                    "gradient_norms": gradient_norms_epoch,
                }
            )
        debug_info["epoch_details"].append(epoch_info)

        if config["verbose"]:
            print(
                f"Epoch {epoch + 1}/{config['epochs']}, "
                f"Train Loss: {epoch_loss:.6e}, "
                f"Val Loss: {val_loss:.6e}, "
                f"LR: {new_lr:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if (epoch + 1) >= config["min_epochs"] and epochs_no_improve >= early_stopping_patience:
                debug_info["early_stopping_trigger"] = epoch + 1
                if config["verbose"]:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    eval_loaders = {
        "train": DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0),
        "val": DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0),
        "test": DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0),
    }
    split_tensors = {
        "train": {"inputs": train_inputs, "targets": train_targets},
        "val": {"inputs": val_inputs, "targets": val_targets},
        "test": {"inputs": test_inputs, "targets": test_targets},
    }

    split_results = {}
    for split_name, loader in eval_loaders.items():
        split_loss, focus_dict, coefficient_dict, predictions = _evaluate_loader(
            model=model,
            loader=loader,
            criterion=config["criterion"],
            device=config["device"],
        )
        split_summary = summarize_interpretability(focus_dict, coefficient_dict)
        split_summary.update(
            {
                "loss": split_loss,
                "inputs": split_tensors[split_name]["inputs"],
                "targets": split_tensors[split_name]["targets"],
                "predictions": predictions,
            }
        )
        split_results[split_name] = split_summary

    test_loss = split_results["test"]["loss"]

    if config["verbose"]:
        print(f"Final Test Loss: {test_loss:.6f}")

    return test_loss, train_losses, val_losses, split_results, debug_info, model


def collect_multiple_runs_supervised(
    n_runs: int,
    input_series: torch.Tensor,
    target_series: torch.Tensor,
    window_size: int,
    temperature: float,
    order: List[int] = None,
    config=None,
    seed: int = 1000,
    verbose: bool = True,
) -> Dict:
    """Run multiple supervised trainings with separate input and target series."""
    if order is None:
        order = [1, 1]

    if config is None:
        train_config = {
            "verbose": True,
            "device": None,
            "learning_rate": 1e-3,
            "scheduler_patience": 5,
            "early_stopping_modifier": 2,
            "criterion": nn.MSELoss(),
        }
    else:
        train_config = dict(config)

    print("Training Configuration:")
    for key, value in train_config.items():
        print(f"  {key}: {value}")

    results = {}
    run_durations = []

    for run in range(n_runs):
        if verbose:
            print(f"Starting Run {run + 1}/{n_runs}")

        start_time = time.time()
        current_seed = seed + run
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)

        run_config = dict(train_config)
        run_config["seed"] = current_seed

        test_loss, train_losses, val_losses, split_results, debug_info, model = train_and_evaluate_supervised(
            input_series=input_series,
            target_series=target_series,
            window_size=window_size,
            temperature=temperature,
            order=order,
            config=run_config,
        )

        elapsed_time = time.time() - start_time
        run_durations.append(elapsed_time)

        results[f"run_{run + 1}"] = {
            "seed": current_seed,
            "test_loss": test_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "split_results": split_results,
            "alpha": split_results["test"]["alpha_mean"],
            "alpha_std": split_results["test"]["alpha_std"],
            "alpha_bias": split_results["test"]["alpha_bias"],
            "alpha_bias_std": split_results["test"]["alpha_bias_std"],
            "f_means": split_results["test"]["f_means"],
            "c_means": split_results["test"]["c_means"],
            "debug_info": debug_info,
            "duration_seconds": elapsed_time,
        }

        if verbose:
            print(f"Run {run + 1} completed. Test Loss: {test_loss:.6e}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_losses = [results[f"run_{i + 1}"]["test_loss"] for i in range(n_runs)]
    split_loss_summary = {}
    for split_name in ["train", "val", "test"]:
        losses = [results[f"run_{i + 1}"]["split_results"][split_name]["loss"] for i in range(n_runs)]
        split_loss_summary[split_name] = {
            "mean_loss": np.mean(losses),
            "std_loss": np.std(losses),
            "min_loss": np.min(losses),
            "max_loss": np.max(losses),
        }

    results["summary"] = {
        "mean_test_loss": np.mean(test_losses),
        "std_test_loss": np.std(test_losses),
        "min_test_loss": np.min(test_losses),
        "max_test_loss": np.max(test_losses),
        "best_run": f"run_{np.argmin(test_losses) + 1}",
        "split_losses": split_loss_summary,
        "mean_duration_seconds": np.mean(run_durations),
    }

    return results


def collect_multiple_runs(
    n_runs: int,
    time_series: torch.Tensor,
    window_size: int,
    temperature: float,
    order: List[int] = None,
    config=None,
    seed: int = 1000,
    verbose: bool = True,
) -> Dict:
    """Run multiple trainings and keep split-aware alpha summaries for each run."""
    if order is None:
        order = [1, 1]

    if config is None:
        train_config = {
            "verbose": True,
            "device": None,
            "learning_rate": 1e-3,
            "scheduler_patience": 5,
            "early_stopping_modifier": 2,
            "criterion": nn.MSELoss(),
        }
    else:
        train_config = dict(config)

    print("Training Configuration:")
    for key, value in train_config.items():
        print(f"  {key}: {value}")

    results = {}
    run_durations = []

    for run in range(n_runs):
        if verbose:
            print(f"Starting Run {run + 1}/{n_runs}")

        start_time = time.time()
        current_seed = seed + run
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)

        run_config = dict(train_config)
        run_config["seed"] = current_seed

        test_loss, train_losses, val_losses, split_results, debug_info, model = train_and_evaluate(
            time_series,
            window_size=window_size,
            temperature=temperature,
            order=order,
            config=run_config,
        )

        elapsed_time = time.time() - start_time
        run_durations.append(elapsed_time)

        results[f"run_{run + 1}"] = {
            "seed": current_seed,
            "test_loss": test_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "split_results": split_results,
            "alpha": split_results["test"]["alpha_mean"],
            "alpha_std": split_results["test"]["alpha_std"],
            "alpha_bias": split_results["test"]["alpha_bias"],
            "alpha_bias_std": split_results["test"]["alpha_bias_std"],
            "f_means": split_results["test"]["f_means"],
            "c_means": split_results["test"]["c_means"],
            "debug_info": debug_info,
            "duration_seconds": elapsed_time,
        }

        if verbose:
            print(f"Run {run + 1} completed. Test Loss: {test_loss:.6e}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_losses = [results[f"run_{i + 1}"]["test_loss"] for i in range(n_runs)]
    split_loss_summary = {}
    for split_name in ["train", "val", "test"]:
        losses = [results[f"run_{i + 1}"]["split_results"][split_name]["loss"] for i in range(n_runs)]
        split_loss_summary[split_name] = {
            "mean_loss": np.mean(losses),
            "std_loss": np.std(losses),
            "min_loss": np.min(losses),
            "max_loss": np.max(losses),
        }

    results["summary"] = {
        "mean_test_loss": np.mean(test_losses),
        "std_test_loss": np.std(test_losses),
        "min_test_loss": np.min(test_losses),
        "max_test_loss": np.max(test_losses),
        "best_run": f"run_{np.argmin(test_losses) + 1}",
        "split_losses": split_loss_summary,
        "mean_duration_seconds": np.mean(run_durations),
    }

    return results


def calculate_multiple_run_statistics(results: Dict, split: str = "test") -> Dict[str, Dict[str, np.ndarray]]:
    """
    Calculate mean/std statistics across runs for a chosen split.

    Args:
        results: Output of collect_multiple_runs().
        split: One of 'train', 'val', or 'test'.
    """
    run_keys = [key for key in results.keys() if key.startswith("run_")]
    if not run_keys:
        raise ValueError("No run data found in results.")
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of 'train', 'val', or 'test'.")

    split_reference = results[run_keys[0]]["split_results"][split]

    stats = {
        "alpha": {},
        "alpha_bias": {"mean": None, "std": None},
        "beta": {},
        "f": {},
        "c": {},
        "loss": {
            "mean": np.mean([results[run_key]["split_results"][split]["loss"] for run_key in run_keys]),
            "std": np.std([results[run_key]["split_results"][split]["loss"] for run_key in run_keys]),
        },
    }

    alpha_keys = split_reference["alpha_mean"].keys()
    for key in alpha_keys:
        alphas = np.stack(
            [results[run_key]["split_results"][split]["alpha_mean"][key] for run_key in run_keys],
            axis=0,
        )
        focus_means = np.stack(
            [results[run_key]["split_results"][split]["f_means"][key] for run_key in run_keys],
            axis=0,
        )
        coefficient_means = np.stack(
            [results[run_key]["split_results"][split]["c_means"][key] for run_key in run_keys],
            axis=0,
        )

        betas = np.empty((len(run_keys),) + alphas.shape[1:-1])
        for run_idx in range(len(run_keys)):
            beta_tilde = np.abs(alphas[run_idx]).sum(axis=-1)
            betas[run_idx] = beta_tilde / beta_tilde.sum(axis=1, keepdims=True)

        stats["alpha"][key] = {"mean": np.mean(alphas, axis=0), "std": np.std(alphas, axis=0)}
        stats["beta"][key] = {"mean": np.mean(betas, axis=0), "std": np.std(betas, axis=0)}
        stats["f"][key] = {"mean": np.mean(focus_means, axis=0), "std": np.std(focus_means, axis=0)}
        stats["c"][key] = {
            "mean": np.mean(coefficient_means, axis=0),
            "std": np.std(coefficient_means, axis=0),
        }

    alpha_bias_values = [
        results[run_key]["split_results"][split]["alpha_bias"]
        for run_key in run_keys
        if results[run_key]["split_results"][split]["alpha_bias"] is not None
    ]
    if alpha_bias_values:
        alpha_bias_stack = np.stack(alpha_bias_values, axis=0)
        stats["alpha_bias"] = {
            "mean": np.mean(alpha_bias_stack, axis=0),
            "std": np.std(alpha_bias_stack, axis=0),
        }

    return stats
