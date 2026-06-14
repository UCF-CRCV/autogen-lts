# Download the GAIA dataset and generate AgBench task JSONLs under `../Tasks/`.

import argparse
import json
import os
import re
import sys

from huggingface_hub import snapshot_download

SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
TEMPLATES_DIR = os.path.join(SCENARIO_DIR, "Templates")
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")
DOWNLOADS_DIR = os.path.join(SCENARIO_DIR, "Downloads")
REPO_DIR = os.path.join(DOWNLOADS_DIR, "GAIA")


def download_gaia():
    """Download the GAIA benchmark from Hugging Face."""

    if not os.path.isdir(DOWNLOADS_DIR):
        os.mkdir(DOWNLOADS_DIR)

    snapshot_download(
        repo_id="gaia-benchmark/GAIA",
        repo_type="dataset",
        local_dir=REPO_DIR,
        local_dir_use_symlinks=True,
    )


def create_jsonl(name, tasks, files_dir, template):
    """Creates a JSONL scenario file with a given name, and template path."""

    if not os.path.isdir(TASKS_DIR):
        os.mkdir(TASKS_DIR)

    with open(os.path.join(TASKS_DIR, name + ".jsonl"), "wt") as fh:
        for task in tasks:
            print(f"Converting: [{name}] {task['task_id']}")

            # Figure out what files we need to copy
            template_cp_list = [template]
            if len(task["file_name"].strip()) > 0:
                template_cp_list.append(
                    [
                        os.path.join(files_dir, task["file_name"].strip()),
                        task["file_name"].strip(),
                        #os.path.join("coding", task["file_name"].strip()),
                    ]
                )

            record = {
                "id": task["task_id"],
                "template": template_cp_list,
                "substitutions": {
                    "scenario.py": {
                        "__FILE_NAME__": task["file_name"],
                    },
                    "expected_answer.txt": {"__EXPECTED_ANSWER__": task["Final answer"]},
                    "prompt.txt": {"__PROMPT__": task["Question"]},
                },
            }

            fh.write(json.dumps(record).strip() + "\n")


###############################################################################
def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=argv[0])
    parser.add_argument(
        "--split",
        choices=["levels", "all"],
        default="levels",
        help="Generate per-level files (levels) or a single file with all levels combined (all).",
    )
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    gaia_validation_files = os.path.join(REPO_DIR, "2023", "validation")
    gaia_test_files = os.path.join(REPO_DIR, "2023", "test")

    if not os.path.isdir(gaia_validation_files) or not os.path.isdir(gaia_test_files):
        download_gaia()

    if not os.path.isdir(gaia_validation_files) or not os.path.isdir(gaia_test_files):
        sys.exit(f"Error: '{REPO_DIR}' does not appear to be a copy of the GAIA repository.")

    # Load the GAIA data
    gaia_validation_tasks = [[], [], []] if args.split == "levels" else [[]]
    with open(os.path.join(gaia_validation_files, "metadata.jsonl")) as fh:
        for line in fh:
            data = json.loads(line)
            if args.split == "levels":
                gaia_validation_tasks[data["Level"] - 1].append(data)
            else:
                gaia_validation_tasks[0].append(data)

    gaia_test_tasks = [[], [], []] if args.split == "levels" else [[]]
    with open(os.path.join(gaia_test_files, "metadata.jsonl")) as fh:
        for line in fh:
            data = json.loads(line)

            # A welcome message -- not a real task
            if data["task_id"] == "0-0-0-0-0":
                continue

            if args.split == "levels":
                gaia_test_tasks[data["Level"] - 1].append(data)
            else:
                gaia_test_tasks[0].append(data)

    # list all directories in the Templates directory
    # and populate a dictionary with the name and path
    templates = {}
    for entry in os.scandir(TEMPLATES_DIR):
        if entry.is_dir():
            templates[re.sub(r"\s", "", entry.name)] = entry.path

    # Add coding directories if needed (these are usually empty and left out of the repo)
    #for template in templates.values():
    #    code_dir_path = os.path.join(template, "coding")
    #    if not os.path.isdir(code_dir_path):
    #        os.mkdir(code_dir_path)

    # Create the various combinations of [models] x [templates]
    for t in templates.items():
        if args.split == "levels":
            create_jsonl(
                f"gaia_validation_level_1__{t[0]}",
                gaia_validation_tasks[0],
                gaia_validation_files,
                t[1],
            )
            create_jsonl(
                f"gaia_validation_level_2__{t[0]}",
                gaia_validation_tasks[1],
                gaia_validation_files,
                t[1],
            )
            create_jsonl(
                f"gaia_validation_level_3__{t[0]}",
                gaia_validation_tasks[2],
                gaia_validation_files,
                t[1],
            )
            create_jsonl(
                f"gaia_test_level_1__{t[0]}",
                gaia_test_tasks[0],
                gaia_test_files,
                t[1],
            )
            create_jsonl(
                f"gaia_test_level_2__{t[0]}",
                gaia_test_tasks[1],
                gaia_test_files,
                t[1],
            )
            create_jsonl(
                f"gaia_test_level_3__{t[0]}",
                gaia_test_tasks[2],
                gaia_test_files,
                t[1],
            )
        else:
            create_jsonl(
                f"gaia_validation_all__{t[0]}",
                gaia_validation_tasks[0],
                gaia_validation_files,
                t[1],
            )
            create_jsonl(
                f"gaia_test_all__{t[0]}",
                gaia_test_tasks[0],
                gaia_test_files,
                t[1],
            )

    return 0

if __name__ == "__main__" and __package__ is None:
    raise SystemExit(main(sys.argv))
