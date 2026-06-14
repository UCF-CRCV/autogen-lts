#!/usr/bin/env python3
"""Joint trace-generation + decision-transformer training loop for AssistantBench.

Trace generation (running the agbench scenario) and decision-transformer
training stay fully decoupled by design; this script just drives them in
alternation. Each epoch:

  1. Generate verification traces by running the agbench scenario with the
     latest decision-transformer checkpoint. The checkpoint path is passed via
     the ``DECISION_TRANSFORMER_WEIGHTS`` env var, which the scenario template
     already reads (and resolves relative to the benchmark dir / repo mount).
  2. Move the run output (``./Results``) to ``./Results_epoch{N}``.
  3. Train the decision transformer for ``--reuse-iterations`` passes over the
     accumulated traces. The new checkpoint feeds the next epoch's generation.

The model/optimizer are kept in memory across epochs so optimizer state
persists; checkpoints are still written to disk each reuse iteration so the
generation subprocess can load them.

Typical usage (run from the AssistantBench benchmark directory):

    python joint_train.py \\
        Tasks/assistant_bench_v1.0_dev__VerifiedMemoryParallelAgents.jsonl \\
        --epochs 5 --reuse-iterations 10 -r 4

Defaults: 5 epochs, 10 reuse iterations.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time

import torch

try:
    import wandb
except Exception:  # wandb is optional
    wandb = None  # type: ignore[assignment]

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH_DIR)

from decision_transformer import DecisionTransformer  # noqa: E402
from train_decision import train_epoch  # noqa: E402

DEFAULT_SCENARIO = "Tasks/assistant_bench_v1.0_dev__VerifiedMemoryParallelAgents.jsonl"


def scenario_name_from_tasks(tasks_path: str) -> str:
    name = os.path.basename(tasks_path)
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    return name


def latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Return the newest ``decision_llm_epoch_{e}_{r}.pt`` by (epoch, reuse_iter)."""
    best, best_key = None, (-1, -1)
    for p in glob.glob(os.path.join(checkpoint_dir, "decision_llm_epoch_*_*.pt")):
        m = re.search(r"decision_llm_epoch_(\d+)_(\d+)\.pt$", os.path.basename(p))
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if key > best_key:
            best_key, best = key, p
    return best


def run_generation(args: argparse.Namespace, weights_rel: str | None) -> None:
    env = os.environ.copy()
    if weights_rel:
        env["DECISION_TRANSFORMER_WEIGHTS"] = weights_rel
        print(f"  generation checkpoint: {weights_rel}", flush=True)
    else:
        env.pop("DECISION_TRANSFORMER_WEIGHTS", None)
        print("  generation checkpoint: <none> (random init)", flush=True)

    cmd = [sys.executable, "-m", "agbench", "run", args.scenario, "-r", str(args.repeat)]
    if args.parallel and args.parallel > 1:
        cmd += ["-p", str(args.parallel)]
    if args.subsample:
        cmd += ["-s", args.subsample]
    if args.native:
        cmd += ["--native"]
    if args.docker_image:
        cmd += ["-d", args.docker_image]
    if args.config:
        cmd += ["-c", args.config]
    if args.env:
        cmd += ["-e", args.env]
    cmd += args.agbench_args

    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=BENCH_DIR, env=env)


def generate_epoch(args: argparse.Namespace, epoch: int, weights_rel: str | None) -> str:
    results_epoch_dir = f"{args.results_prefix}{epoch}"
    if os.path.isdir(results_epoch_dir):
        print(f"[epoch {epoch}] {results_epoch_dir} exists; skipping generation.", flush=True)
        return results_epoch_dir

    # Move any stale ./Results aside so agbench starts from a clean slate.
    if os.path.isdir("Results"):
        stale = f"Results.stale.{int(time.time())}"
        print(f"[epoch {epoch}] moving stale ./Results -> {stale}", flush=True)
        shutil.move("Results", stale)

    print(f"[epoch {epoch}] generating traces ...", flush=True)
    run_generation(args, weights_rel)

    if not os.path.isdir("Results"):
        raise RuntimeError(f"[epoch {epoch}] agbench run produced no ./Results directory")
    shutil.move("Results", results_epoch_dir)
    print(f"[epoch {epoch}] traces -> {results_epoch_dir}", flush=True)
    return results_epoch_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default=DEFAULT_SCENARIO,
        help=f"Tasks JSONL to run for trace generation (default: {DEFAULT_SCENARIO}).",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of generate+train rounds (default: 5).")
    parser.add_argument(
        "--reuse-iterations",
        type=int,
        default=10,
        help="Training passes over the trace pool per epoch (default: 10).",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Epoch index to start from, for resuming (default: 0).",
    )
    parser.add_argument(
        "--scenario-name",
        default=None,
        help="Override the results/scenario folder name (default: derived from the tasks filename).",
    )
    parser.add_argument(
        "--results-prefix",
        default="Results_epoch",
        help="Prefix for per-epoch results folders (default: Results_epoch).",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Where checkpoints are read/written.")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional checkpoint to warm-start the model and seed epoch-start generation.",
    )
    parser.add_argument(
        "--no-accumulate",
        action="store_true",
        help="Train only on the epoch just generated instead of all epochs so far.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")

    # agbench run passthrough
    parser.add_argument(
        "-r",
        "--repeat",
        type=int,
        default=4,
        help="Repetitions per task for trace generation (default: 4; >=2 needed for advantages).",
    )
    parser.add_argument("-p", "--parallel", type=int, default=1, help="Parallel agbench processes (default: 1).")
    parser.add_argument("-s", "--subsample", default=None, help="agbench --subsample value.")
    parser.add_argument("-c", "--config", default=None, help="agbench --config file.")
    parser.add_argument("-e", "--env", default=None, help="agbench --env file.")
    parser.add_argument("-d", "--docker-image", default=None, help="agbench --docker-image.")
    parser.add_argument("--native", action="store_true", help="Run agbench with --native.")
    parser.add_argument(
        "agbench_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded verbatim to `agbench run` (place after `--`).",
    )

    args = parser.parse_args(argv)
    if args.agbench_args and args.agbench_args[0] == "--":
        args.agbench_args = args.agbench_args[1:]
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.chdir(BENCH_DIR)

    scenario_name = args.scenario_name or scenario_name_from_tasks(args.scenario)
    use_wandb = (not args.no_wandb) and (wandb is not None)
    if not use_wandb and not args.no_wandb:
        print("wandb not available; continuing without logging.", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecisionTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.init_checkpoint and os.path.isfile(args.init_checkpoint):
        print(f"Warm-starting model from {args.init_checkpoint}", flush=True)
        model.load(args.init_checkpoint)

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n========== EPOCH {epoch} / {args.epochs - 1} ==========", flush=True)

        # Checkpoint that drives this epoch's generation.
        if epoch == args.start_epoch and args.init_checkpoint:
            weights_rel: str | None = args.init_checkpoint
        else:
            weights_rel = latest_checkpoint(args.checkpoint_dir)

        generate_epoch(args, epoch, weights_rel)

        results_epochs = [epoch] if args.no_accumulate else list(range(0, epoch + 1))
        print(
            f"[epoch {epoch}] training on epochs {results_epochs} "
            f"for {args.reuse_iterations} reuse iterations ...",
            flush=True,
        )

        # One wandb run per epoch.
        if use_wandb:
            wandb.init(
                project="agbench-decision",
                name=f"{scenario_name}-epoch{epoch}",
                group=scenario_name,
                reinit=True,
                config={
                    "scenario_name": scenario_name,
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "reuse_iterations": args.reuse_iterations,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "repeat": args.repeat,
                    "accumulate": not args.no_accumulate,
                    "results_epochs": results_epochs,
                },
            )

        ckpt, _ = train_epoch(
            epoch,
            scenario_name,
            reuse_iterations=args.reuse_iterations,
            results_epochs=results_epochs,
            results_prefix=args.results_prefix,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_wandb=use_wandb,
            model=model,
            optimizer=optimizer,
            device=device,
            global_step=0,
        )

        if use_wandb:
            wandb.finish()
        print(f"[epoch {epoch}] latest checkpoint -> {ckpt}", flush=True)

    print("\nJoint training complete.", flush=True)


if __name__ == "__main__":
    main()
