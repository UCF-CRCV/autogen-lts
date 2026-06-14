import asyncio
import os
import json
import time
import yaml
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_core.models import ChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_agentchat.agents import CodeExecutorAgent

# Suppress warnings about the requests.Session() not being closed
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)


async def intercept_messages_for_tracing(stream, team_idx: int):
    """Wrap the MagenticOne team stream and emit step_message events for each message.

    Events are written to the JSONL file pointed to by MEMORY_EVENTS_FILE in
    the same schema used by the parallel-memory templates.
    """
    events_file = os.environ.get("MEMORY_EVENTS_FILE")
    step_num = 0

    async for message in stream:
        source = getattr(message, "source", "unknown")
        if hasattr(message, "to_model_text"):
            content = message.to_model_text()
        elif hasattr(message, "content"):
            content = str(getattr(message, "content"))
        else:
            content = str(message)

        if events_file:
            try:
                os.makedirs(os.path.dirname(events_file), exist_ok=True)
                event: Dict[str, Any] = {
                    "event": "step_message",
                    "timestamp": datetime.utcnow().isoformat(),
                    "team_idx": team_idx,
                    "step_num": step_num,
                    "source": source,
                    "message_type": type(message).__name__,
                    "content": content,
                }
                with open(events_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            except Exception:
                # Tracing must never break the main run.
                pass

        step_num += 1
        yield message


async def main() -> None:

    # Track scenario runtime for logging
    scenario_start_time = time.time()

    # Load model configuration and create the model client.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    orchestrator_client = ChatCompletionClient.load_component(config["orchestrator_client"])
    coder_client = ChatCompletionClient.load_component(config["coder_client"])
    web_surfer_client = ChatCompletionClient.load_component(config["web_surfer_client"])
    file_surfer_client = ChatCompletionClient.load_component(config["file_surfer_client"])
    
    # Read the prompt
    prompt = ""
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read().strip()
    filename = "__FILE_NAME__".strip()

    # Prepare step-level tracing file (shared schema with other templates)
    events_file_path = os.path.abspath("memory_events.jsonl")
    os.environ["MEMORY_EVENTS_FILE"] = events_file_path

    # Set up the team
    coder = MagenticOneCoderAgent(
        "Assistant",
        model_client = coder_client,
    )

    executor = CodeExecutorAgent("ComputerTerminal", code_executor=LocalCommandLineCodeExecutor(work_dir=os.getcwd()))

    file_surfer = FileSurfer(
        name="FileSurfer",
        model_client = file_surfer_client,
    )
                
    web_surfer = MultimodalWebSurfer(
        name="WebSurfer",
        model_client = web_surfer_client,
        downloads_folder=os.getcwd(),
        debug_dir="logs",
        to_save_screenshots=True,
    )

    team = MagenticOneGroupChat(
        [coder, executor, file_surfer, web_surfer],
        model_client=orchestrator_client,
        max_turns=30,
        final_answer_prompt= f""",
We have completed the following task:

{prompt}

The above messages contain the conversation that took place to complete the task.
Read the above conversation and output a FINAL ANSWER to the question.
To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
""".strip()
    )

    # Prepare the prompt
    filename_prompt = ""
    if len(filename) > 0:
        filename_prompt = f"The question is about a file, document or image, which can be accessed by the filename '{filename}' in the current working directory."
    task = f"{prompt}\n\n{filename_prompt}"

    # Run the task with tracing
    stream = team.run_stream(task=task.strip())
    traced_stream = intercept_messages_for_tracing(stream, team_idx=0)
    task_result = await Console(traced_stream)

    # Extract final answer from the task_result
    final_answer: Optional[str] = None
    try:
        if task_result.messages:
            last_msg = task_result.messages[-1]
            if hasattr(last_msg, "content"):
                final_answer = str(last_msg.content)
            elif hasattr(last_msg, "to_model_text"):
                final_answer = last_msg.to_model_text()
    except Exception:
        final_answer = None

    overall_runtime = time.time() - scenario_start_time

    first_correct: Optional[bool] = None
    aggregated_correct: Optional[bool] = None

    # Persist comparison metrics for consistency with multi-team templates.
    comparison_metrics: Dict[str, Any] = {
        "first_team": {
            "team_idx": 0,
            "runtime_seconds": overall_runtime,
            "final_answer": final_answer,
            "correct": first_correct,
        },
        "aggregated": {
            "runtime_seconds": overall_runtime,
            "final_answer": final_answer,
            "correct": aggregated_correct,
        },
    }
    try:
        with open("first_vs_aggregated_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(comparison_metrics, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Failed to write first_vs_aggregated_metrics.json: {e}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
