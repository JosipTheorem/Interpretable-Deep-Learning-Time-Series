## Cell 39

```python
# Supervised event test: raw_xy / rel_xy / dxy -> E(t)

cluster_idx = CLUSTER_INDICES_TO_RUN[0]
cluster = clusters[cluster_idx]

x_selected = cluster["x_selected"]
y_selected = cluster["y_selected"]
selected_ids = cluster["selected_particle_ids"]

time_series_raw, frames_raw = build_dcits_input(x_selected, y_selected, frames, mode="raw_xy")
time_series_rel, frames_rel = build_dcits_input(x_selected, y_selected, frames, mode="rel_xy")
time_series_dxy, frames_dxy = build_dcits_input(x_selected, y_selected, frames, mode="dxy")

delta_raw = torch.abs(time_series_raw[:, 1:] - time_series_raw[:, :-1])

EVENT_QUANTILE = 0.95
event_thresholds = torch.quantile(delta_raw, EVENT_QUANTILE, dim=1, keepdim=True)
target_series_event = (delta_raw >= event_thresholds).float()

print("Event fraction:", target_series_event.mean().item())

event_supervised_variants = {
    "raw_xy_to_event": {
        "input_series": time_series_raw[:, :-1],
        "target_series": target_series_event,
    },
    "rel_xy_to_event": {
        "input_series": time_series_rel[:, :-1],
        "target_series": target_series_event,
    },
    "dxy_to_event": {
        "input_series": time_series_dxy[:, :-1],
        "target_series": target_series_event[:, 1:],
    },
}

n_positive = target_series_event.sum()
n_negative = target_series_event.numel() - n_positive
pos_weight = n_negative / n_positive
print('##############',pos_weight)

event_train_config = {
    "verbose": False,
    "device": device,
    "seed": seed,
    "learning_rate": 1e-3,
    "scheduler_patience": 5,
    "early_stopping_modifier": 2,
    "criterion": nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)),
    "criterion": nn.BCEWithLogitsLoss(),
    "epochs": EPOCHS,
    "batch_size": 64,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
}

al
```

```text
Event fraction: 0.05016722530126572
############## tensor(18.9333)
================================================================================
Training supervised event variant: raw_xy_to_event
Training Configuration:
  verbose: False
  device: cuda
  seed: 42
  learning_rate: 0.001
  scheduler_patience: 5
  early_stopping_modifier: 2
  criterion: BCEWithLogitsLoss()
  epochs: 100
  batch_size: 64
  train_ratio: 0.8
  val_ratio: 0.1
raw_xy_to_event: mean test loss = 0.274563 +/- 0.215745
================================================================================
Training supervised event variant: rel_xy_to_event
Training Configuration:
  verbose: False
  device: cuda
  seed: 42
  learning_rate: 0.001
  scheduler_patience: 5
  early_stopping_modifier: 2
  criterion: BCEWithLogitsLoss()
  epochs: 100
  batch_size: 64
  train_ratio: 0.8
  val_ratio: 0.1
rel_xy_to_event: mean test loss = 0.240571 +/- 0.064490
================================================================================
Training supervised event variant: dxy_to_event
Training Configuration:
  verbose: False
  device: cuda
  seed: 42
  learning_rate: 0.001
  scheduler_patience: 5
  early_stopping_modifier: 2
  criterion: BCEWithLogitsLoss()
  epochs: 100
  batch_size: 64
  train_ratio: 0.8
  val_ratio: 0.1
dxy_to_event: mean test loss = 0.162142 +/- 0.002626
```

## Cell 40

```python
label_map_event = {
    "raw_xy_to_event": [f"x_{pid}" for pid in selected_ids] + [f"y_{pid}" for pid in selected_ids],
    "rel_xy_to_event": [f"xrel_{pid}" for pid in selected_ids] + [f"yrel_{pid}" for pid in selected_ids],
    "dxy_to_event": [f"dx_{pid}" for pid in selected_ids] + [f"dy_{pid}" for pid in selected_ids],
}

show_average_metric_heatmaps(
    all_results=all_results_event,
    label_map=label_map_event,
    cluster_idx=cluster_idx,
    variant_names=list(event_supervised_variants.keys()),
    split_name="test",
    metrics=("Focuser", "alpha", "alpha * Q"),
    order_idx=1,
)
```

```text
<Figure size 2000x1020 with 18 Axes>
```

```text
<Figure size 2000x1020 with 18 Axes>
```

```text
<Figure size 2000x1020 with 18 Axes>
```

## Cell 41

```python
EVENT_PROB_THRESHOLD = 0.5

for variant_name, bundle in all_results_event.items():
    result = bundle["result"]

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for run_key in result["run_keys"]:
        split_result = result["runs"][run_key]["split_results"]["test"]

        logits = split_result["predictions"]
        y_true = split_result["targets"]

        probs = torch.sigmoid(logits)
        y_pred = (probs >= EVENT_PROB_THRESHOLD).float()

        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        tn = ((y_pred == 0) & (y_true == 0)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    print("=" * 80)
    print(variant_name)
    print(f"accuracy:  {np.mean(accuracies):.3f} +/- {np.std(accuracies):.3f}")
    print(f"precision: {np.mean(precisions):.3f} +/- {np.std(precisions):.3f}")
    print(f"recall:    {np.mean(recalls):.3f} +/- {np.std(recalls):.3f}")
    print(f"F1:        {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
```

```text
================================================================================
raw_xy_to_event
accuracy:  0.964 +/- 0.000
precision: 0.000 +/- 0.000
recall:    0.000 +/- 0.000
F1:        0.000 +/- 0.000
================================================================================
rel_xy_to_event
accuracy:  0.930 +/- 0.039
precision: 0.028 +/- 0.028
recall:    0.034 +/- 0.033
F1:        0.025 +/- 0.021
================================================================================
dxy_to_event
accuracy:  0.964 +/- 0.000
precision: 0.000 +/- 0.000
recall:    0.000 +/- 0.000
F1:        0.000 +/- 0.000
```

## Cell 42

```python
TOP_FRACTION = 0.05

for variant_name, bundle in all_results_event.items():
    result = bundle["result"]

    top_hit_rates = []

    for run_key in result["run_keys"]:
        split_result = result["runs"][run_key]["split_results"]["test"]

        logits = split_result["predictions"]
        y_true = split_result["targets"]

        probs = torch.sigmoid(logits).flatten()
        y_true_flat = y_true.flatten()

        n_top = int(TOP_FRACTION * len(probs))
        top_idx = torch.topk(probs, n_top).indices

        hit_rate = y_true_flat[top_idx].mean().item()
        top_hit_rates.append(hit_rate)

    print("=" * 80)
    print(variant_name)
    print(f"top {100 * TOP_FRACTION:.1f}% hit rate: {np.mean(top_hit_rates):.3f} +/- {np.std(top_hit_rates):.3f}")
    
#random guessing gives about 0.05
```

```text
================================================================================
raw_xy_to_event
top 5.0% hit rate: 0.011 +/- 0.009
================================================================================
rel_xy_to_event
top 5.0% hit rate: 0.032 +/- 0.020
================================================================================
dxy_to_event
top 5.0% hit rate: 0.025 +/- 0.014
```

## Cell 43

```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

EVENT_PROB_THRESHOLD = 0.1

for variant_name, bundle in all_results_event.items():
    result = bundle["result"]

    y_true_all = []
    y_pred_all = []

    for run_key in result["run_keys"]:
        split_result = result["runs"][run_key]["split_results"]["test"]

        logits = split_result["predictions"]
        y_true = split_result["targets"]

        probs = torch.sigmoid(logits)
        y_pred = (probs >= EVENT_PROB_THRESHOLD).int()

        y_true_all.append(y_true.cpu().numpy().reshape(-1))
        y_pred_all.append(y_pred.cpu().numpy().reshape(-1))

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["no event", "event"],
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{variant_name}: confusion matrix")
    plt.show()
```

```text
<Figure size 640x480 with 2 Axes>
```

```text
<Figure size 640x480 with 2 Axes>
```

```text
<Figure size 640x480 with 2 Axes>
```

## Cell 44

```python
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score,
)

for variant_name, bundle in all_results_event.items():
    result = bundle["result"]

    y_true_all = []
    y_score_all = []

    for run_key in result["run_keys"]:
        split_result = result["runs"][run_key]["split_results"]["test"]

        logits = split_result["predictions"]
        y_true = split_result["targets"]

        probs = torch.sigmoid(logits)

        y_true_all.append(y_true.cpu().numpy().reshape(-1))
        y_score_all.append(probs.cpu().numpy().reshape(-1))

    y_true_all = np.concatenate(y_true_all)
    y_score_all = np.concatenate(y_score_all)

    roc_auc = roc_auc_score(y_true_all, y_score_all)
    avg_precision = average_precision_score(y_true_all, y_score_all)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    RocCurveDisplay.from_predictions(
        y_true_all,
        y_score_all,
        ax=axes[0],
        name=variant_name,
    )
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title(f"ROC AUC = {roc_auc:.3f}")

    PrecisionRecallDisplay.from_predictions(
        y_true_all,
        y_score_all,
        ax=axes[1],
        name=variant_name,
    )
    axes[1].axhline(y_true_all.mean(), color="black", linestyle="--", lw=1)
    axes[1].set_title(f"Average precision = {avg_precision:.3f}")

    fig.suptitle(variant_name)
    plt.tight_layout()
    plt.show()

    print("=" * 80)
    print(variant_name)
    print(f"event fraction: {y_true_all.mean():.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Average precision: {avg_precision:.3f}")
```

```text
<Figure size 1100x400 with 2 Axes>
```

```text
================================================================================
raw_xy_to_event
event fraction: 0.036
ROC AUC: 0.388
Average precision: 0.028
```

```text
<Figure size 1100x400 with 2 Axes>
```

```text
================================================================================
rel_xy_to_event
event fraction: 0.036
ROC AUC: 0.529
Average precision: 0.040
```

```text
<Figure size 1100x400 with 2 Axes>
```

```text
================================================================================
dxy_to_event
event fraction: 0.036
ROC AUC: 0.435
Average precision: 0.036
```
