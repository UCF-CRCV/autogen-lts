from glob import glob
from itertools import groupby
from typing import List, Tuple
import os
import json
import re
import numpy as np
import random
import sys
from tqdm import tqdm
import torch
import wandb


from decision_transformer import DecisionTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Scripts"))
from assistantbench_evaluator import question_scorer


def order_trace_states(trace_states: list[str]) -> dict[str, list[list[str]]]:
    def get_ts(path: str) -> int:
        name = os.path.basename(path)
        m = re.search(r"-ts(\d+)\.pt$", name)
        if not m:
            return -1
        return int(m.group(1))

    def get_problem_id(path: str) -> str:
        return path.split("/")[-4]

    def get_repetition(path: str) -> str:
        return path.split("/")[-3]

    def get_epoch(path: str) -> int:
        # Example path segment: Results_epoch1
        parts = path.split("/")
        for part in parts:
            m = re.match(r"Results_epoch(\d+)$", part)
            if m:
                return int(m.group(1))
        return -1
    
    trace_states = [p for p in trace_states if get_ts(p) >= 0]
    # Sort once by (problem_id, epoch, repetition, timestamp) so groupby can build the structure.
    trace_states = sorted(
        trace_states,
        key=lambda p: (get_problem_id(p), get_epoch(p), get_repetition(p), get_ts(p)),
    )

    ordered: dict[str, list[list[str]]] = {}
    for problem_id, group in groupby(trace_states, key=get_problem_id):
        reps: list[list[str]] = []
        # Keep epoch boundaries so Runs from different Results_epochX folders
        # don't get merged into the same repetition bucket.
        for _epoch_rep, rep_group in groupby(group, key=lambda p: (get_epoch(p), get_repetition(p))):
            reps.append(list(rep_group))
        ordered[problem_id] = reps
    return ordered


def find_answer_files(trace_paths: List[List[str]]) -> Tuple[List[str], List[str], List[str], List[str]]:
    # Get only one per repetition.
    trace_paths = [trace[0] for trace in trace_paths]
    expected_answers = [f[:f.index("traces/")] + "expected_answer.txt" for f in trace_paths]
    output_answers = [f[:f.index("traces/")] + "first_vs_aggregated_metrics.json" for f in trace_paths]
    shared_memory_files = [f[:f.index("traces/")] + ".shared_memory.json" for f in trace_paths]
    prompt_files = [f[:f.index("traces/")] + "prompt.txt" for f in trace_paths]
    return expected_answers, output_answers, shared_memory_files, prompt_files


def extract_final_answer_text(output: str) -> str:
    cleaned = re.sub(r"[*_`]", "", output)
    m = re.search(r"FINAL (?:AGGREGATED )?ANSWER:\s*(.+)", cleaned)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    return cleaned.strip()


def score_answer(output: str, expected_answer: str) -> float:
    return float(question_scorer(extract_final_answer_text(output), expected_answer))



def load_memory_usage_vector(memory_events_file: str, trace_data: list[dict], decisions: list[int]) -> list[int]:
    usage = [0] * len(decisions)
    if not os.path.isfile(memory_events_file):
        return usage

    store_by_key: dict[str, tuple[int, int]] = {}
    try:
        with open(memory_events_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if event.get("event") != "memory_store":
                    continue
                key = event.get("key")
                team_idx = event.get("team_idx")
                step_num = event.get("step_num")
                if isinstance(key, str) and isinstance(team_idx, int) and isinstance(step_num, int):
                    store_by_key[key] = (team_idx, step_num)
    except Exception:
        return usage

    trace_index_by_team_step: dict[tuple[int, int], int] = {}
    for i, trace_state in enumerate(trace_data):
        team_idx = trace_state.get("team_idx")
        step_num = trace_state.get("step_num")
        if isinstance(team_idx, int) and isinstance(step_num, int):
            trace_index_by_team_step.setdefault((team_idx, step_num), i)

    try:
        with open(memory_events_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if event.get("event") != "memory_use":
                    continue
                key = event.get("key")
                if not isinstance(key, str):
                    continue
                stored = store_by_key.get(key)
                if stored is None:
                    continue
                idx = trace_index_by_team_step.get(stored)
                if idx is None:
                    continue
                if decisions[idx] == 0:
                    usage[idx] += 1
    except Exception:
        return usage

    return usage


def collect_gradient_inputs(
    model: DecisionTransformer,
    scenario_name: str,
    results_epochs,
    results_prefix: str = "Results_epoch",
):
    """Build advantage-weighted gradient inputs from the traces in one or more
    ``{results_prefix}{N}`` folders.

    Returns ``(all_gradient_inputs, trace_count, total_reward)``.
    """
    all_trace_states = []
    for epoch_num in results_epochs:
        all_trace_states.extend(glob(f'./{results_prefix}{epoch_num}/{scenario_name}/*/*/traces/*.pt'))
    print('trace states count:', len(all_trace_states))
    all_trace_states = order_trace_states(all_trace_states)

    all_gradient_inputs = []
    trace_count, total_reward = 0, 0
    for i_record, (problem_id, question_traces) in enumerate(all_trace_states.items()):
        if len(question_traces) < 2:
            continue
        expected_answers, output_answers, shared_memory_files, prompt_files = find_answer_files(question_traces)

        question_rewards, bad_indices = [], []
        for i_trace, (expected_answer, output_answer, shared_memory_file, prompt_file) in enumerate(zip(expected_answers, output_answers, shared_memory_files, prompt_files)):
            try:
                with open(expected_answer, "r") as f:
                    expected_answer = f.read().strip()
                with open(output_answer, "r") as f:
                    output_answer = json.load(f)
                with open(shared_memory_file, "r") as f:
                    shared_memory = json.load(f)
                with open(prompt_file, "r") as f:
                    query_text = f.read().strip()
            except Exception as e:
                bad_indices.append(i_trace)
                continue

            reward = score_answer(output_answer["aggregated"]["final_answer"], expected_answer) + score_answer(
                output_answer["first_team"]["final_answer"], expected_answer
            )
            question_rewards.append(float(reward))

        # Drop bad indices from question_traces
        if len(bad_indices) > 0:
            question_traces = [trace for i, trace in enumerate(question_traces) if i not in bad_indices]
            if len(question_traces) == 0:
                continue
            expected_answers = [expected_answers[i] for i in range(len(expected_answers)) if i not in bad_indices]
            output_answers = [output_answers[i] for i in range(len(output_answers)) if i not in bad_indices]
            shared_memory_files = [shared_memory_files[i] for i in range(len(shared_memory_files)) if i not in bad_indices]
            prompt_files = [prompt_files[i] for i in range(len(prompt_files)) if i not in bad_indices]
            # continue

        # Calculate advantage for each trace
        # If only one trace, no normalization needed
        advantages = None
        if len(question_rewards) == 1:
            advantages = np.array([0])
        else:
            baseline = np.mean(question_rewards)
            advantages = np.array(question_rewards) - baseline
            # Normalize advantages
            std = np.std(advantages)
            if std > 1e-8:
                advantages = advantages / std

        print(i_record, advantages)

        # check if all advantages are smaller than 1e-8
        trace_count += 1

        # If all advantages are ~0, there's no learning signal for this question.
        if all(abs(advantage) < 1e-8 for advantage in advantages):
            continue
        total_reward += 1

        # Create gradient inputs for each trace
        for i_trace in range(len(question_traces)):
            trace_data = [model.load_trace_state(trace) for trace in question_traces[i_trace]]

            decisions = [trace_state["decision"] for trace_state in trace_data]
            if decisions[0] in ["YES", "NO"]:
                decisions = [0 if decision == "YES" else 1 for decision in decisions]
            decision_logits = [trace_state["valid_logits"] for trace_state in trace_data]
            decision_embeddings = [trace_state["verification_embeddings"] for trace_state in trace_data]
            memory_bank_embeddings = [trace_state["memory_bank"] for trace_state in trace_data]
            base = question_traces[i_trace][0]
            memory_events_file = base[: base.index("traces/")] + "memory_events.jsonl"
            memory_usage = load_memory_usage_vector(memory_events_file, trace_data, decisions)

            gradient_input = {
                "query_text": query_text,
                "advantage": advantages[i_trace],
                "decisions": decisions,
                "decision_logits": decision_logits,
                "decision_embeddings": decision_embeddings,
                "memory_bank_embeddings": memory_bank_embeddings,
                "memory_usage": memory_usage,
                "trace_length": len(decisions),
                "batch_record_idx": i_record,
                "trace_idx_in_record": i_trace,
            }

            all_gradient_inputs.append(gradient_input)

    return all_gradient_inputs, trace_count, total_reward


def train_epoch(
    epoch: int,
    scenario_name: str,
    *,
    reuse_iterations: int = 10,
    results_epochs=None,
    results_prefix: str = "Results_epoch",
    checkpoint_in: str | None = None,
    checkpoint_dir: str = "checkpoints",
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    use_wandb: bool = True,
    model: DecisionTransformer | None = None,
    optimizer=None,
    device=None,
    global_step: int = 0,
):
    """Train the decision transformer for a single data-collection epoch.

    Reuses an in-memory ``model``/``optimizer`` when provided so a joint loop can
    keep optimizer state across epochs; otherwise creates them and optionally
    loads ``checkpoint_in``. When ``use_wandb`` is True, assumes the caller has
    already initialized the wandb run. Returns ``(last_checkpoint_path, global_step)``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = DecisionTransformer().to(device)
        if checkpoint_in is not None:
            print("loading model from:", checkpoint_in)
            model.load(checkpoint_in)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if results_epochs is None:
        results_epochs = [epoch]

    all_gradient_inputs, trace_count, total_reward = collect_gradient_inputs(
        model, scenario_name, results_epochs, results_prefix
    )

    if trace_count > 0:
        print("Average epoch reward:", total_reward / trace_count)
        print("trace count:", trace_count)
    else:
        print("Average epoch reward: N/A (no traces)")

    # Train the model
    model.train()

    last_checkpoint = None
    for reuse_iter in tqdm(range(reuse_iterations), desc="Training Reused Epochs"):
        # Shuffle gradient inputs for this iteration
        random_indices = list(range(len(all_gradient_inputs)))
        random.shuffle(random_indices)

        # Process in batches with gradient accumulation
        num_batches = (len(random_indices) + batch_size - 1) // batch_size
        print("num_batches:", num_batches)

        for batch_idx in range(num_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(random_indices))
            batch_indices = random_indices[start_idx:end_idx]

            # Zero gradients at the start of each batch
            optimizer.zero_grad()

            batch_loss = []
            reward_base_advantage = []
            reward_usage_bonus_per_step = []
            reward_combined_step_advantage_mean = []
            reward_combined_step_advantage_mean_nonzero = []
            reward_nonzero_step_fraction = []
            step_losses = []
            sparsity_losses = []
            policy_losses = []

            # Process each trace in the batch
            for idx in batch_indices:
                gradient_input = all_gradient_inputs[idx]
                # Calculate loss for this trace (which handles its own backpropagation)
                loss, metrics = model.run_decision_prediction_grad(gradient_input)
                batch_loss.append(loss)
                reward_base_advantage.append(metrics["reward/base_advantage"])
                reward_usage_bonus_per_step.append(metrics["reward/usage_bonus_per_step"])
                reward_combined_step_advantage_mean.append(metrics["reward/combined_step_advantage_mean"])
                reward_combined_step_advantage_mean_nonzero.append(metrics["reward/combined_step_advantage_mean_nonzero"])
                reward_nonzero_step_fraction.append(metrics["reward/nonzero_step_fraction"])
                step_losses.extend(metrics["step_losses"])
                sparsity_losses.extend(metrics["sparsity_losses"])
                policy_losses.extend(metrics["policy_losses"])

            # Update parameters after processing the entire batch
            optimizer.step()

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": float(np.mean(batch_loss)) if batch_loss else 0.0,
                        "reward/base_advantage": float(np.mean(reward_base_advantage)) if reward_base_advantage else 0.0,
                        "reward/usage_bonus_per_step": float(np.mean(reward_usage_bonus_per_step)) if reward_usage_bonus_per_step else 0.0,
                        "reward/combined_step_advantage_mean": float(np.mean(reward_combined_step_advantage_mean)) if reward_combined_step_advantage_mean else 0.0,
                        "reward/combined_step_advantage_mean_nonzero": float(np.mean(reward_combined_step_advantage_mean_nonzero)) if reward_combined_step_advantage_mean_nonzero else 0.0,
                        "reward/nonzero_step_fraction": float(np.mean(reward_nonzero_step_fraction)) if reward_nonzero_step_fraction else 0.0,
                        "step_losses": float(np.mean(step_losses)) if step_losses else 0.0,
                        "sparsity_losses": float(np.mean(sparsity_losses)) if sparsity_losses else 0.0,
                        "policy_losses": float(np.mean(policy_losses)) if policy_losses else 0.0,
                        "train/epoch": epoch,
                        "train/reuse_iter": reuse_iter,
                        "train/batch_idx": batch_idx,
                    },
                    step=global_step,
                )
            global_step += 1

        # save model
        os.makedirs(checkpoint_dir, exist_ok=True)
        last_checkpoint = os.path.join(checkpoint_dir, f"decision_llm_epoch_{epoch}_{reuse_iter}.pt")
        model.save(last_checkpoint)

    print("Done training")
    return last_checkpoint, global_step


def find_latest_full_model(scenario_name: str, results_prefix: str = "Results_epoch") -> str | None:
    """Find the newest ``full_model.pt`` produced by past runs (by epoch index)."""
    model_candidates = glob(f"./{results_prefix}*/{scenario_name}/full_model.pt")
    latest_model_path = None
    latest_epoch = -1
    for p in model_candidates:
        m = re.search(rf"{results_prefix}(\d+)", p)
        if m:
            epoch_num = int(m.group(1))
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_model_path = p
    return latest_model_path


if __name__ == "__main__":

    # Standalone single-epoch training (unchanged default behavior).
    reuse_iterations = 10
    batch_size = 16
    learning_rate = 1e-4
    epoch = 1
    use_wandb = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecisionTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scenario_name = "assistant_bench_v1.0_dev__VerifiedMemoryParallelAgents"

    if use_wandb:
        wandb.init(
            project="agbench-decision",
            config={
                "scenario_name": scenario_name,
                "reuse_iterations": reuse_iterations,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epoch": epoch,
            },
        )

    latest_model_path = find_latest_full_model(scenario_name)
    if latest_model_path is not None:
        print("loading model from:", latest_model_path)
        model.load(latest_model_path)

    train_epoch(
        epoch,
        scenario_name,
        reuse_iterations=reuse_iterations,
        results_epochs=[0],
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_wandb=use_wandb,
        model=model,
        optimizer=optimizer,
        device=device,
    )

    if use_wandb:
        wandb.finish()
