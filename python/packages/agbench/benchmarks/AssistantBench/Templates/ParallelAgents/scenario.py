import asyncio
import os
import re
import logging
import yaml
import warnings
import contextvars
import builtins
import shutil
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from collections import deque
from autogen_agentchat import TRACE_LOGGER_NAME as AGENTCHAT_TRACE_LOGGER_NAME, EVENT_LOGGER_NAME as AGENTCHAT_EVENT_LOGGER_NAME
from autogen_core import EVENT_LOGGER_NAME as CORE_EVENT_LOGGER_NAME
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    UserMessage,
)
from autogen_core.logging import LLMCallEvent
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import (
    HandoffMessage,
    MultiModalMessage,
    StopMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_core import CancellationToken
from autogen_ext.models.openai._model_info import _MODEL_TOKEN_LIMITS, resolve_model
from autogen_agentchat.utils import content_to_str

# Suppress warnings about the requests.Session() not being closed
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

core_event_logger = logging.getLogger(CORE_EVENT_LOGGER_NAME)
agentchat_event_logger = logging.getLogger(AGENTCHAT_EVENT_LOGGER_NAME)
agentchat_trace_logger = logging.getLogger(AGENTCHAT_TRACE_LOGGER_NAME)


# Create a context variable to hold the current team's log file and the current team id.
current_log_file = contextvars.ContextVar("current_log_file", default=None)
current_team_id = contextvars.ContextVar("current_team_id", default=None)

# Save the original print function and event_logger.info method.
original_print = builtins.print
original_agentchat_event_logger_info = agentchat_event_logger.info
original_core_event_logger_info = core_event_logger.info

class LogHandler(logging.FileHandler):
    def __init__(self, filename: str = "log.jsonl", print_message: bool = True) -> None:
        super().__init__(filename, mode="w")
        self.print_message = print_message

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = datetime.fromtimestamp(record.created).isoformat()
            if AGENTCHAT_EVENT_LOGGER_NAME in record.name:
                original_msg = record.msg
                record.msg = json.dumps(
                    {
                        "timestamp": ts,
                        "source": record.msg.source,
                        "message": content_to_str(record.msg.content),
                        "type": record.msg.type,
                    }
                )
                super().emit(record)
                record.msg = original_msg
            elif CORE_EVENT_LOGGER_NAME in record.name:
                if isinstance(record.msg, LLMCallEvent):
                    original_msg = record.msg
                    record.msg = json.dumps(
                        {
                            "timestamp": ts,
                            "prompt_tokens": record.msg.kwargs["prompt_tokens"],
                            "completion_tokens": record.msg.kwargs["completion_tokens"],
                            "type": "LLMCallEvent",
                        }
                    )
                    super().emit(record)
                    record.msg = original_msg
        except Exception:
            print("error in logHandler.emit", flush=True)
            self.handleError(record)

def tee_print(*args, **kwargs):
    # Get the current log file from the context.
    log_file = current_log_file.get()
    # Call the original print (goes to the console).
    original_print(*args, **kwargs)
    # Also write to the log file if one is set.
    if log_file is not None:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        message = sep.join(map(str, args)) + end
        log_file.write(message)
        log_file.flush()

def team_specific_agentchat_event_logger_info(msg, *args, **kwargs):
    team_id = current_team_id.get()
    if team_id is not None:
        # Get a logger with a team-specific name.
        team_logger = logging.getLogger(f"{AGENTCHAT_EVENT_LOGGER_NAME}.team{team_id}")
        team_logger.info(msg, *args, **kwargs)
    else:
        original_agentchat_event_logger_info(msg, *args, **kwargs)

def team_specific_core_event_logger_info(msg, *args, **kwargs):
    team_id = current_team_id.get()
    if team_id is not None:
        # Get a logger with a team-specific name.
        team_logger = logging.getLogger(f"{CORE_EVENT_LOGGER_NAME}.team{team_id}")
        team_logger.info(msg, *args, **kwargs)
    else:
        original_core_event_logger_info(msg, *args, **kwargs)

# Monkey-patch the built-in print and event_logger.info methods with our team-specific versions.
builtins.print = tee_print
agentchat_event_logger.info = team_specific_agentchat_event_logger_info
core_event_logger.info = team_specific_core_event_logger_info


async def intercept_messages_for_tracing(stream, team_idx: int):
    """Wrap a team stream and emit step_message events for each message.

    Events are written to the JSONL file pointed to by MEMORY_EVENTS_FILE in
    the same schema used by the parallel-memory templates, so downstream
    training code can consume traces from ParallelAgents as well.
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


async def run_team(team: MagenticOneGroupChat, team_idx: int, task: str, cancellation_token: CancellationToken, logfile):
    token_logfile = current_log_file.set(logfile)
    token_team_id = current_team_id.set(team_idx)
    try:
        stream = team.run_stream(
            task=task.strip(),
            cancellation_token=cancellation_token
        )
        traced_stream = intercept_messages_for_tracing(stream, team_idx=team_idx)
        task_result = await Console(traced_stream)
        return team_idx, task_result
    finally:
        current_log_file.reset(token_logfile)
        current_team_id.reset(token_team_id)
        logfile.close()

async def aggregate_final_answer(task: str, client: ChatCompletionClient, team_results, source: str = "Aggregator", cancellation_token: Optional[CancellationToken] = None) -> str:
        """
        team_results: {"team_key": TaskResult}
        team_completion_order: The order in which the teams completed their tasks
        """

        if len(team_results) == 1:
            final_answer = list(team_results.values())[0].messages[-1].content
            aggregator_logger.info(
                f"{source} (Response):\n{final_answer}"
            )
            return final_answer

        assert len(team_results) > 1

        aggregator_messages_to_send = {team_id: deque() for team_id in team_results.keys()} # {team_id: context}

        team_ids = list(team_results.keys())
        current_round = 0
        while (
            not all(len(team_result.messages) == 0 for team_result in team_results.values())
            and ((not resolve_model(client._create_args["model"]) in _MODEL_TOKEN_LIMITS) or client.remaining_tokens([m for messages in aggregator_messages_to_send.values() for m in messages])
            > 2000)
        ):
            team_idx = team_ids[current_round % len(team_ids)]
            if len(team_results[team_idx].messages) > 0:
                m = team_results[team_idx].messages[-1]
                if isinstance(m, ToolCallRequestEvent | ToolCallExecutionEvent):
                    # Ignore tool call messages.
                    pass
                elif isinstance(m, StopMessage | HandoffMessage):
                    aggregator_messages_to_send[team_idx].appendleft(UserMessage(content=m.to_model_text(), source=m.source))
                elif m.source == "MagenticOneOrchestrator":
                    assert isinstance(m, TextMessage | ToolCallSummaryMessage)
                    aggregator_messages_to_send[team_idx].appendleft(AssistantMessage(content=m.to_model_text(), source=m.source))
                else:
                    assert isinstance(m, (TextMessage, MultiModalMessage, ToolCallSummaryMessage))
                    aggregator_messages_to_send[team_idx].appendleft(UserMessage(content=m.to_model_text(), source=m.source))
                team_results[team_idx].messages.pop()
            current_round += 1

        # Log the messages to send
        payload = ""
        for team_idx, messages in aggregator_messages_to_send.items():
            payload += f"\n{'*'*75} \n" f"Team #: {team_idx}" f"\n{'*'*75} \n"
            for message in messages:
                payload += f"\n{'-'*75} \n" f"{message.source}:\n" f"\n{message.content}\n"
            payload += f"\n{'-'*75} \n" f"Team #{team_idx} stop reason:\n" f"\n{team_results[team_idx].stop_reason}\n"
        payload += f"\n{'*'*75} \n"
        aggregator_logger.info(f"{source} (Aggregator Messages):\n{payload}")

        context: List[LLMMessage] = []

        # Add the preamble
        context.append(
            UserMessage(
                content=f"Earlier you were asked the following:\n\n{task}\n\nYour team then worked diligently to address that request. You have been provided with a collection of transcripts and stop reasons from {len(team_results)} different teams to the question. Your task is to carefully evaluate the correctness of each team's response by analyzing their respective transcripts and stop reasons. After considering all perspectives, provide a FINAL ANSWER to the question. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.",
                source=source,
            )
        )

        for team_idx, aggregator_messages in aggregator_messages_to_send.items():
            context.append(
                UserMessage(
                    content=f"Transcript from Team #{team_idx}:",
                    source=source,
                )
            )
            for message in aggregator_messages:
                context.append(message)
            context.append(
                UserMessage(
                    content=f"Stop reason from Team #{team_idx}:",
                    source=source,
                )
            )
            context.append(
                UserMessage(
                    content=team_results[team_idx].stop_reason if team_results[team_idx].stop_reason else "No stop reason provided.",
                    source=source,
                )
            )

        # ask for the final answer
        context.append(
            UserMessage(
                content=f"""
    Let's think step-by-step. Carefully review the conversation above, critically evaluate the correctness of each team's response, and then output a FINAL ANSWER to the question. The question is repeated here for convenience:

    {task}

    To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
    Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
    ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
    If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and don't include units such as $ or percent signs unless specified otherwise.
    If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
    If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
    """.strip(),
                source=source,
            )
        )

        response = await client.create(context, cancellation_token=cancellation_token)
        assert isinstance(response.content, str)

        final_answer = re.sub(r"FINAL ANSWER:", "[FINAL ANSWER]:", response.content)
        aggregator_logger.info(
            f"{source} (Response):\n{final_answer}"
        )

        return re.sub(r"FINAL ANSWER:", "FINAL AGGREGATED ANSWER:", response.content)


async def main(num_teams: int, num_answers: int) -> None:

    # Record start time for tracking first completing team and overall runtime
    start_time = time.time()

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

    # Prepare the prompt
    filename_prompt = ""
    if len(filename) > 0:
        filename_prompt = f"The question is about a file, document or image, which can be accessed by the filename '{filename}' in the current working directory."
    task = f"{prompt}\n\n{filename_prompt}"

    # Reset logs directory (remove all files in it)
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)

    # Prepare step-level tracing file (shared schema with other templates)
    events_file_path = os.path.abspath("memory_events.jsonl")
    os.environ["MEMORY_EVENTS_FILE"] = events_file_path

    teams = []
    async_tasks = []
    tokens = []

    # For logging first team metrics in-memory as well as to file
    first_team_saved = False
    first_team_id: Optional[int] = None
    first_team_answer: Optional[str] = None
    first_team_runtime: Optional[float] = None

    for team_idx in range(num_teams):
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
            debug_dir=logs_dir,
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
        teams.append(team)
        cancellation_token = CancellationToken()
        tokens.append(cancellation_token)
        logfile = open(f"console_log_{team_idx}.txt", "w")
        team_agentchat_logger = logging.getLogger(f"{AGENTCHAT_EVENT_LOGGER_NAME}.team{team_idx}")
        team_core_logger = logging.getLogger(f"{CORE_EVENT_LOGGER_NAME}.team{team_idx}")
        team_log_handler = LogHandler(f"log_{team_idx}.jsonl", print_message=False)
        team_agentchat_logger.addHandler(team_log_handler)
        team_core_logger.addHandler(team_log_handler)
        async_task = asyncio.create_task(
            run_team(team, team_idx, task, cancellation_token, logfile)
        )
        async_tasks.append(async_task)

    # Wait until at least num_answers tasks have completed.
    team_results = {}
    for future in asyncio.as_completed(async_tasks):
        try:
            team_id, result = await future
            team_results[team_id] = result
            
            # Save the answer and runtime of the first completing team
            if not first_team_saved:
                end_time = time.time()
                runtime = end_time - start_time
                
                # Extract the answer from the team result (matching pattern from aggregate_final_answer)
                answer = ""
                if result.messages:
                    last_message = result.messages[-1]
                    if hasattr(last_message, 'content'):
                        answer = str(last_message.content)
                    elif hasattr(last_message, 'to_model_text'):
                        answer = last_message.to_model_text()
                
                # Save to file and in-memory
                first_team_id = team_id
                first_team_answer = answer
                first_team_runtime = runtime

                # Save to file
                first_team_data = {
                    "team_id": team_id,
                    "answer": answer,
                    "runtime_seconds": runtime,
                    "stop_reason": result.stop_reason if hasattr(result, 'stop_reason') else None
                }
                
                with open("first_team_completion.json", "w") as f:
                    json.dump(first_team_data, f, indent=2)
                
                print(f"First completing team (Team {team_id}) saved:")
                print(f"  Runtime: {runtime:.2f} seconds")
                print(f"  Answer: {answer[:200]}..." if len(answer) > 200 else f"  Answer: {answer}")
                
                first_team_saved = True
        except Exception as e:
            # Optionally log exception.
            print(f"Task raised an exception: {e}")
        if len(team_results) >= num_answers:
            break

    # Cancel any pending teams.
    for task, token in zip(async_tasks, tokens):
        if not task.done():
            token.cancel()
    # Await all tasks to handle cancellation gracefully.
    await asyncio.gather(*async_tasks, return_exceptions=True)

    print("len(team_results):", len(team_results))
    final_answer = await aggregate_final_answer(prompt, orchestrator_client, team_results)
    print(final_answer)

    # Compute overall runtime
    overall_runtime = time.time() - start_time

    first_correct: Optional[bool] = None
    aggregated_correct: Optional[bool] = None

    # Persist comparison metrics for easy tabulation later.
    comparison_metrics: Dict[str, Any] = {
        "first_team": {
            "team_idx": first_team_id,
            "runtime_seconds": first_team_runtime,
            "final_answer": first_team_answer,
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
    num_teams = 3
    num_answers = 3

    agentchat_trace_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("trace.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    agentchat_trace_logger.addHandler(file_handler)

    core_event_logger.setLevel(logging.DEBUG)
    agentchat_event_logger.setLevel(logging.DEBUG)
    log_handler = LogHandler()
    core_event_logger.addHandler(log_handler)
    agentchat_event_logger.addHandler(log_handler)

    # Create another logger for the aggregator
    aggregator_logger = logging.getLogger("aggregator")
    aggregator_logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("aggregator_log.txt", mode="w")
    fh.setLevel(logging.DEBUG)
    aggregator_logger.addHandler(fh)


    asyncio.run(main(num_teams, num_answers))
