import json
import os
import re
import sys

import pandas as pd
from agbench.tabulate_cmd import default_tabulate

sys.path.insert(0, os.path.dirname(__file__))

from assistantbench_evaluator import question_scorer

EXCLUDE_DIR_NAMES = ["__pycache__"]


def _extract_final_answer_text(output: str) -> str:
    """Extract bare answer text from FINAL ANSWER or FINAL AGGREGATED ANSWER lines."""
    cleaned = re.sub(r"[*_`]", "", output)
    matches = list(
        re.finditer(r"FINAL (?:AGGREGATED )?ANSWER:\s*(.+)", cleaned, re.IGNORECASE)
    )
    if not matches:
        return cleaned.strip()

    answer = matches[-1].group(1).strip().splitlines()[0].strip()
    answer = re.sub(r"[.!?]+$", "", answer).strip()
    return answer


def _safe_read_expected_answer(instance_dir: str) -> str | None:
    expected_answer_file = os.path.join(instance_dir, "expected_answer.txt")
    if not os.path.isfile(expected_answer_file):
        return None
    try:
        with open(expected_answer_file, "rt", encoding="utf-8") as fh:
            expected = fh.read().strip()
    except Exception:
        return None
    return expected or None


def _read_metrics(instance_dir: str) -> dict | None:
    metrics_path = os.path.join(instance_dir, "first_vs_aggregated_metrics.json")
    if not os.path.isfile(metrics_path):
        return None
    try:
        with open(metrics_path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _score_answer_text(final_answer: str, expected_answer: str) -> bool:
    prediction = _extract_final_answer_text(final_answer)
    try:
        return float(question_scorer(prediction, expected_answer)) == 1.0
    except Exception:
        return False


def _score_from_metrics(
    instance_dir: str,
    *,
    rescore: bool,
    use_first_team: bool = False,
) -> bool | None:
    data = _read_metrics(instance_dir)
    if data is None:
        return None

    section = (data.get("first_team", {}) or {}) if use_first_team else (data.get("aggregated", {}) or {})

    if rescore:
        expected_answer = _safe_read_expected_answer(instance_dir)
        if expected_answer is None:
            return None
        final_answer = section.get("final_answer")
        if not isinstance(final_answer, str):
            return None
        try:
            return _score_answer_text(final_answer, expected_answer)
        except Exception:
            return None

    correct = section.get("correct")
    return correct if isinstance(correct, bool) else None


def _score_from_console(instance_dir: str) -> bool | None:
    expected_answer = _safe_read_expected_answer(instance_dir)
    if expected_answer is None:
        return None

    console_log_file = os.path.join(instance_dir, "console_log.txt")
    if not os.path.isfile(console_log_file):
        return None

    with open(console_log_file, "rt", encoding="utf-8") as fh:
        console_log = fh.read()

    cleaned = re.sub(r"[*_`]", "", console_log)
    if not re.search(r"FINAL (?:AGGREGATED )?ANSWER:\s*\S", cleaned, re.IGNORECASE):
        return None

    try:
        return _score_answer_text(_extract_final_answer_text(console_log), expected_answer)
    except Exception:
        return None


def _score_aggregated_answer_text(instance_dir: str) -> bool | None:
    data = _read_metrics(instance_dir)
    if data is None:
        return None
    expected_answer = _safe_read_expected_answer(instance_dir)
    if expected_answer is None:
        return None
    final_answer = (data.get("aggregated") or {}).get("final_answer")
    if not isinstance(final_answer, str):
        return None
    return _score_answer_text(final_answer, expected_answer)


def make_scorer(*, rescore: bool = False):
    """Prefer aggregated answer text scored with the AssistantBench evaluator."""

    def scorer(instance_dir: str) -> bool | None:
        if rescore:
            result = _score_from_metrics(instance_dir, rescore=True, use_first_team=False)
            if result is not None:
                return result
            return _score_from_console(instance_dir)

        result = _score_aggregated_answer_text(instance_dir)
        if result is not None:
            return result
        return _score_from_console(instance_dir)

    return scorer


def _parse_tabulate_options(rest: list[str]) -> tuple[int, bool, list[str]]:
    verbosity = 0
    rescore = False
    filtered: list[str] = []
    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg in ("-v", "--verbose"):
            verbosity += 1
            i += 1
            continue
        if arg.startswith("-") and len(arg) > 1 and all(ch == "v" for ch in arg[1:]):
            verbosity += len(arg) - 1
            i += 1
            continue
        if arg == "--rescore":
            rescore = True
            i += 1
            continue
        filtered.append(arg)
        i += 1
    return verbosity, rescore, filtered


def _resolve_runlogs(filtered_rest: list[str]) -> str | None:
    for arg in filtered_rest:
        if not arg.startswith("-"):
            return arg
    return None


def _build_exclude_dir_names(runlogs: str | None) -> list[str]:
    exclude_dir_names = list(EXCLUDE_DIR_NAMES)
    if runlogs is None or not os.path.isdir(runlogs):
        return exclude_dir_names

    for task_id in os.listdir(runlogs):
        if task_id in exclude_dir_names:
            continue
        task_path = os.path.join(runlogs, task_id)
        if not os.path.isdir(task_path):
            continue
        has_trial = any(name.isdigit() for name in os.listdir(task_path))
        if not has_trial:
            exclude_dir_names.append(task_id)
    return exclude_dir_names


_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_SCENARIO_DIR = os.path.realpath(os.path.join(_SCRIPT_DIR, os.path.pardir))
_AB_REPO_DIR = os.path.join(_SCENARIO_DIR, "Downloads", "AssistantBench")


def _load_task_id_to_difficulty() -> dict[str, str]:
    task_id_to_difficulty: dict[str, str] = {}
    for filename in ("assistant_bench_v1.0_dev.jsonl", "assistant_bench_v1.0_test.jsonl"):
        data_path = os.path.join(_AB_REPO_DIR, filename)
        if not os.path.isfile(data_path):
            continue
        with open(data_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_id = row.get("id")
                difficulty = row.get("difficulty")
                if not isinstance(task_id, str) or difficulty is None:
                    continue
                task_id_to_difficulty.setdefault(task_id, str(difficulty))
    return task_id_to_difficulty


def _tabulate_accuracy_by_difficulty(runlogs: str, *, rescore: bool) -> None:
    if not os.path.isdir(runlogs):
        sys.stderr.write(f"\n[verbosity] '{runlogs}' is not a directory; skipping difficulty breakdown.\n\n")
        return

    task_id_to_difficulty = _load_task_id_to_difficulty()
    if not task_id_to_difficulty:
        sys.stderr.write(
            "\n[verbosity] Could not find AssistantBench task files under "
            f"'{_AB_REPO_DIR}'. Run init_tasks.py first.\n\n"
        )
        return

    score = make_scorer(rescore=rescore)
    rows: list[dict] = []
    for task_id in sorted(
        os.listdir(runlogs),
        key=lambda s: os.path.getmtime(os.path.join(runlogs, s)),
    ):
        if task_id in EXCLUDE_DIR_NAMES:
            continue

        difficulty = task_id_to_difficulty.get(task_id)
        if difficulty is None:
            continue

        task_path = os.path.join(runlogs, task_id)
        if not os.path.isdir(task_path):
            continue

        for instance in sorted(os.listdir(task_path), key=lambda s: os.path.getmtime(os.path.join(task_path, s))):
            if not instance.isdigit():
                continue
            instance_dir = os.path.join(task_path, instance)
            result = score(instance_dir)
            if result is None:
                continue
            rows.append(
                {
                    "Difficulty": difficulty,
                    "Trial": int(instance),
                    "Correct": int(result),
                    "Total": 1,
                }
            )

    if not rows:
        sys.stderr.write(f"\n[verbosity] No scorable instances found under '{runlogs}'.\n\n")
        return

    df = pd.DataFrame(rows).groupby(["Difficulty", "Trial"], as_index=False).sum(numeric_only=True)
    df["Accuracy"] = df["Correct"] / df["Total"]

    print("\nAccuracy by Difficulty (aggregated answers)\n")
    print(df.sort_values(["Difficulty", "Trial"]).to_string(index=False))


def _tabulate_first_vs_aggregated(runlogs: str, *, rescore: bool) -> None:
    if not os.path.isdir(runlogs):
        sys.stderr.write(f"\n[verbosity] '{runlogs}' is not a directory; skipping first-team breakdown.\n\n")
        return

    rows = []
    for task_id in sorted(
        os.listdir(runlogs),
        key=lambda s: os.path.getmtime(os.path.join(runlogs, s)),
    ):
        if task_id in EXCLUDE_DIR_NAMES:
            continue

        task_path = os.path.join(runlogs, task_id)
        if not os.path.isdir(task_path):
            continue

        for instance in sorted(os.listdir(task_path), key=lambda s: os.path.getmtime(os.path.join(task_path, s))):
            if not instance.isdigit():
                continue
            instance_dir = os.path.join(task_path, instance)
            data = _read_metrics(instance_dir)
            if data is None:
                continue

            first = data.get("first_team", {}) or {}
            agg = data.get("aggregated", {}) or {}

            if rescore:
                first_correct = _score_from_metrics(
                    instance_dir, rescore=True, use_first_team=True
                )
                agg_correct = _score_from_metrics(
                    instance_dir, rescore=True, use_first_team=False
                )
            else:
                expected_answer = _safe_read_expected_answer(instance_dir)
                first_correct = None
                agg_correct = None
                if expected_answer is not None:
                    first_answer = first.get("final_answer")
                    if isinstance(first_answer, str):
                        first_correct = _score_answer_text(first_answer, expected_answer)
                    agg_answer = agg.get("final_answer")
                    if isinstance(agg_answer, str):
                        agg_correct = _score_answer_text(agg_answer, expected_answer)

            rows.append(
                {
                    "Task Id": task_id,
                    "Instance": int(instance),
                    "First Team Id": first.get("team_idx"),
                    "First Correct": first_correct,
                    "First Runtime (s)": first.get("runtime_seconds"),
                    "Aggregated Correct": agg_correct,
                    "Aggregated Runtime (s)": agg.get("runtime_seconds"),
                }
            )

    if not rows:
        sys.stderr.write(
            "\n[verbosity] No first_vs_aggregated_metrics.json files found. "
            "Run a parallel/memory template first.\n\n"
        )
        return

    df = pd.DataFrame(rows).sort_values(["Task Id", "Instance"])

    print("\nFirst Team vs Aggregated (per instance)\n")
    print(df.to_string(index=False))

    summary = {}
    for col in ["First Correct", "Aggregated Correct"]:
        if col in df.columns:
            summary[f"Mean {col}"] = df[col].astype(float).mean(skipna=True)
    for col in ["First Runtime (s)", "Aggregated Runtime (s)"]:
        if col in df.columns:
            summary[f"Mean {col}"] = df[col].astype(float).mean(skipna=True)

    if summary:
        print("\nFirst Team vs Aggregated Summary\n")
        print(pd.DataFrame([summary]).to_string(index=False))


def main(args):
    invocation_cmd, *rest = args
    verbosity, rescore, filtered_rest = _parse_tabulate_options(rest)
    runlogs = _resolve_runlogs(filtered_rest)
    exclude_dir_names = _build_exclude_dir_names(runlogs)

    default_tabulate(
        [invocation_cmd] + filtered_rest,
        scorer=make_scorer(rescore=rescore),
        exclude_dir_names=exclude_dir_names,
    )

    if verbosity == 0:
        return

    if runlogs is None:
        sys.stderr.write(
            f"\n[verbosity] Could not determine runlogs directory from arguments: {filtered_rest}\n\n"
        )
        return

    if verbosity >= 1:
        _tabulate_accuracy_by_difficulty(runlogs, rescore=rescore)
    if verbosity >= 2:
        _tabulate_first_vs_aggregated(runlogs, rescore=rescore)


if __name__ == "__main__" and __package__ is None:
    main(sys.argv)
