import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parent
SUPPORT_ROOT = EXPERIMENT_ROOT / "support_utils"

DCITS_ROOT_CANDIDATES = [
    REPO_ROOT.parent / "DCIts",
    Path.cwd() / "DCIts",
    Path.cwd().parent / "DCIts",
]

DCITS_ROOT = next(
    (path for path in DCITS_ROOT_CANDIDATES if (path / "src" / "dcits.py").exists()),
    None,
)

if DCITS_ROOT is None:
    raise FileNotFoundError(
        "Could not find a sibling DCIts clone. Expected a layout like: "
        "workspace/DCIts and workspace/Interpretable-Deep-Learning-Time-Series."
    )

for path in [SUPPORT_ROOT, DCITS_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.util_echo import (
    discover_echo_folders,
    load_particles_full_frame,
    make_echo_result_dirs,
    plot_acf_for_clusters,
    plot_all_clusters,
    plot_cluster_time_series,
    plot_granger_y_pairs,
    plot_individual_clusters,
    run_adf_tests,
    run_granger_y_tests,
    save_cluster_members,
    select_clusters_constrained_kmeans,
    train_clusters_for_echo,
)


DEFAULT_DATA_ROOT = EXPERIMENT_ROOT / "sample_data"
DEFAULT_RESULTS_ROOT = EXPERIMENT_ROOT / "artifacts" / "pipeline_results"

N_SERIES_TO_MODEL = 10
WINDOW_SIZE = 5
ORDER = [1, 1]
TEMPERATURE = 1.0
EPOCHS = 100
N_RUNS = 5
FIGURE_DPI = 200
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    else:
        print("Using CPU")

    return device


def run_one_echo(job, args, device, n_series_to_model=None):
    echo_dir = job["echo_dir"]
    amplitude_result_name = job["amplitude_result_name"]
    echo_name = job["echo_name"]
    if n_series_to_model is None:
        n_series_to_model = args.n_series_to_model

    print("=" * 100)
    print(f"Amplitude: {job['amplitude_name']}")
    print(f"Echo: {echo_name}")
    print(f"Input: {echo_dir}")
    print(f"Particles per cluster target: {n_series_to_model}")

    result_dirs = make_echo_result_dirs(
        results_root=args.results_root,
        amplitude_result_name=amplitude_result_name,
        echo_name=echo_name,
    )

    frames, particle_ids, x_full, y_full = load_particles_full_frame(echo_dir)

    clusters = select_clusters_constrained_kmeans(
        particle_ids=particle_ids,
        x_matrix=x_full,
        y_matrix=y_full,
        n_particles=n_series_to_model,
        frame_index=0,
        random_state=args.seed,
    )

    clusters_to_use = clusters
    if args.cluster_limit is not None:
        clusters_to_use = clusters[: args.cluster_limit]

    metadata = {
        "amplitude": amplitude_result_name,
        "echo": echo_name,
        "input_dir": str(echo_dir),
        "n_full_frame_particles": len(particle_ids),
        "n_clusters": len(clusters),
        "n_clusters_analyzed": len(clusters_to_use),
        "n_series_to_model": n_series_to_model,
    }

    pd.DataFrame([metadata]).to_csv(result_dirs["root"] / "echo_metadata.csv", index=False)
    save_cluster_members(
        clusters=clusters_to_use,
        save_path=result_dirs["root"] / "cluster_members.csv",
        metadata={
            "amplitude": amplitude_result_name,
            "echo": echo_name,
            "input_dir": str(echo_dir),
        },
    )

    print("Full-frame particles:", len(particle_ids))
    print("Number of clusters:", len(clusters))

    if not args.skip_cluster_plots:
        print("Saving cluster overview plots")
        plot_all_clusters(
            x_full=x_full,
            y_full=y_full,
            clusters=clusters_to_use,
            save_path=result_dirs["root"] / "clusters.png",
            figure_dpi=args.figure_dpi,
        )
        plot_individual_clusters(
            x_full=x_full,
            y_full=y_full,
            particle_ids=particle_ids,
            clusters=clusters_to_use,
            output_dir=result_dirs["individual_clusters"],
            figure_dpi=args.figure_dpi,
        )
        plot_cluster_time_series(
            frames=frames,
            clusters=clusters_to_use,
            output_dir=result_dirs["individual_cluster_time_series"],
            figure_dpi=args.figure_dpi,
            event_percentile=args.event_percentile,
        )

    if not args.skip_adf:
        print("Running ADF tests")
        adf_summary = run_adf_tests(clusters=clusters_to_use, frames=frames)
        adf_summary.update(metadata)
        pd.DataFrame([adf_summary]).to_csv(result_dirs["root"] / "adf_summary.csv", index=False)

    if not args.skip_acf_pacf:
        print("Saving ACF plots")
        plot_acf_for_clusters(
            clusters=clusters_to_use,
            frames=frames,
            output_dir=result_dirs["acf"],
            figure_dpi=args.figure_dpi,
            lags=10,
            include_shuffled=not args.no_shuffled_controls,
            shuffle_seed=args.shuffle_control_seed,
        )

    granger_results = []
    if not args.skip_granger:
        print("Running Granger y-pair tests")
        granger_results = run_granger_y_tests(
            clusters=clusters_to_use,
            lag=args.granger_lag,
            alpha=args.granger_alpha,
        )
        granger_df = pd.DataFrame(granger_results)
        granger_df.to_csv(result_dirs["root"] / "granger_y_pairs.csv", index=False)
        plot_granger_y_pairs(
            x_full=x_full,
            y_full=y_full,
            clusters=clusters_to_use,
            granger_results=granger_results,
            save_path=result_dirs["root"] / "granger_y_pairs_highlighted.png",
            figure_dpi=args.figure_dpi,
        )
        print("Mutual y-Granger pairs:", len(granger_results))

    if not args.skip_training:
        print("Training DCIts models")
        training_metadata = dict(metadata)
        training_metadata.update(
            {
                "n_runs": args.n_runs,
                "epochs": args.epochs,
                "window_size": args.window_size,
                "temperature": args.temperature,
                "include_shuffled_controls": not args.no_shuffled_controls,
                "shuffle_control_seed": args.shuffle_control_seed,
                "event_percentile": args.event_percentile,
                "event_context_radius": args.event_context_radius,
                "quiet_exclusion_radius": args.quiet_exclusion_radius,
            }
        )

        train_clusters_for_echo(
            clusters=clusters_to_use,
            frames=frames,
            result_dirs=result_dirs,
            device=device,
            seed=args.seed,
            n_runs=args.n_runs,
            window_size=args.window_size,
            temperature=args.temperature,
            order=ORDER,
            epochs=args.epochs,
            figure_dpi=args.figure_dpi,
            cluster_limit=None,
            alpha_heatmap_threshold=args.alpha_heatmap_threshold,
            metadata=training_metadata,
            include_shuffled=not args.no_shuffled_controls,
            shuffle_seed=args.shuffle_control_seed,
            event_percentile=args.event_percentile,
            event_context_radius=args.event_context_radius,
            quiet_exclusion_radius=args.quiet_exclusion_radius,
            requested_bead_ids=args.requested_bead_ids,
        )

    print(f"Finished {amplitude_result_name}/{echo_name}")


def validate_args(args):
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
    if args.window_size < 2:
        raise ValueError("--window-size must be at least 2 because the baseline uses the previous timestep.")
    if args.n_series_to_model < 1:
        raise ValueError("--n-series-to-model must be at least 1.")
    if args.alternate_n_series_to_model is not None and args.alternate_n_series_to_model < 1:
        raise ValueError("--alternate-n-series-to-model must be at least 1 when provided.")
    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.echo_limit is not None and args.echo_limit < 1:
        raise ValueError("--echo-limit must be at least 1 when provided.")
    if args.cluster_limit is not None and args.cluster_limit < 1:
        raise ValueError("--cluster-limit must be at least 1 when provided.")
    if not 0 < args.event_percentile < 100:
        raise ValueError("--event-percentile must be between 0 and 100.")

def run_all_echoes(args):
    device = get_device(args.seed)
    echo_jobs = discover_echo_folders(args.data_root)

    if args.start_at is not None:
        start_idx = None
        for job_idx, job in enumerate(echo_jobs):
            job_name = f"{job['amplitude_result_name']}/{job['echo_name']}"
            if job_name == args.start_at:
                start_idx = job_idx
                break

        if start_idx is None:
            available_jobs = [f"{job['amplitude_result_name']}/{job['echo_name']}" for job in echo_jobs]
            raise ValueError(
                f"Could not find --start-at {args.start_at!r}. "
                f"Available jobs include: {available_jobs}"
            )

        echo_jobs = echo_jobs[start_idx:]

    if args.echo_limit is not None:
        echo_jobs = echo_jobs[: args.echo_limit]

    jobs_df = pd.DataFrame(
        [
            {
                "amplitude": job["amplitude_result_name"],
                "echo": job["echo_name"],
                "input_dir": str(job["echo_dir"]),
            }
            for job in echo_jobs
        ]
    )
    args.results_root.mkdir(parents=True, exist_ok=True)
    jobs_df.to_csv(args.results_root / "echo_jobs.csv", index=False)

    print(f"Found {len(echo_jobs)} echo folders to analyze")

    for job_idx, job in enumerate(echo_jobs, start=1):
        print(f"Job {job_idx}/{len(echo_jobs)}")
        n_series_to_model = args.n_series_to_model

        if args.alternate_n_series_to_model is not None:
            echo_number = int(job["echo_name"].replace("echo", ""))
            use_alternate = (
                args.alternate_n_series_on == "even" and echo_number % 2 == 0
            ) or (
                args.alternate_n_series_on == "odd" and echo_number % 2 == 1
            )

            if use_alternate:
                n_series_to_model = args.alternate_n_series_to_model

        run_one_echo(
            job,
            args=args,
            device=device,
            n_series_to_model=n_series_to_model,
        )

    print("All echo analyses finished")


def parse_args():
    parser = argparse.ArgumentParser(description="Run DCIts echo analysis for all amplitude/echo folders.")

    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--n-series-to-model", type=int, default=N_SERIES_TO_MODEL)
    parser.add_argument("--alternate-n-series-to-model", type=int, default=None)
    parser.add_argument("--alternate-n-series-on", choices=["even", "odd"], default="even")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--figure-dpi", type=int, default=FIGURE_DPI)

    parser.add_argument("--echo-limit", type=int, default=None)
    parser.add_argument("--cluster-limit", type=int, default=None)
    parser.add_argument("--start-at", type=str, default=None)

    parser.add_argument("--granger-lag", type=int, default=3)
    parser.add_argument("--granger-alpha", type=float, default=0.05)
    parser.add_argument("--alpha-heatmap-threshold", type=float, default=0.1)
    parser.add_argument("--event-percentile", type=float, default=99.6)
    parser.add_argument("--event-context-radius", type=int, default=2)
    parser.add_argument("--quiet-exclusion-radius", type=int, default=2)
    parser.add_argument("--shuffle-control-seed", type=int, default=2026)
    parser.add_argument("--requested-bead-ids", type=int, nargs="*", default=[109])

    parser.add_argument("--skip-cluster-plots", action="store_true")
    parser.add_argument("--skip-adf", dest="skip_adf", action="store_true")
    parser.add_argument("--run-adf", dest="skip_adf", action="store_false")
    parser.add_argument("--skip-acf-pacf", action="store_true")
    parser.add_argument("--skip-granger", dest="skip_granger", action="store_true")
    parser.add_argument("--run-granger", dest="skip_granger", action="store_false")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--no-shuffled-controls", action="store_true")
    parser.set_defaults(skip_adf=True, skip_granger=True)

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    validate_args(cli_args)
    run_all_echoes(cli_args)
