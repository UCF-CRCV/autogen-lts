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
import atexit
import time
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any
from collections import deque
from autogen_agentchat import TRACE_LOGGER_NAME as AGENTCHAT_TRACE_LOGGER_NAME, EVENT_LOGGER_NAME as AGENTCHAT_EVENT_LOGGER_NAME
from autogen_agentchat.base import TaskResult
from autogen_core import EVENT_LOGGER_NAME as CORE_EVENT_LOGGER_NAME, CancellationToken
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.teams import MagenticMemoryGroupChat
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
from autogen_ext.models.openai._model_info import _MODEL_TOKEN_LIMITS, resolve_model
from autogen_agentchat.utils import content_to_str

import torch


def _load_decision_transformer():
    """Load DecisionTransformer in both local-dev and agbench task containers."""
    # Prefer a local import if available (e.g., running from the AssistantBench folder).
    try:
        from decision_transformer import DecisionTransformer  # type: ignore
        return DecisionTransformer
    except Exception:
        # In agbench task containers, repo code is typically available under /autogen_python.
        benchmark_path = "/autogen_python/packages/agbench/benchmarks/AssistantBench"
        if benchmark_path not in sys.path:
            sys.path.append(benchmark_path)
        from decision_transformer import DecisionTransformer  # type: ignore
        return DecisionTransformer

# Suppress warnings about the requests.Session() not being closed
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

core_event_logger = logging.getLogger(CORE_EVENT_LOGGER_NAME)
agentchat_event_logger = logging.getLogger(AGENTCHAT_EVENT_LOGGER_NAME)
agentchat_trace_logger = logging.getLogger(AGENTCHAT_TRACE_LOGGER_NAME)
aggregator_logger = logging.getLogger("aggregator")


# Module-level functions that read from a JSON file
# This works across subprocess boundaries since files are shared
# These functions will be serialized and available in executor subprocesses
def get_shared_memory_keys():
    """Get all keys from shared memory. Returns a list of strings.
    
    This function can be called from Python code executed in the executor.
    Example: keys = get_shared_memory_keys()
    
    Note: The shared memory is stored in a JSON file (.shared_memory.json)
    The path is set via SHARED_MEMORY_FILE environment variable.
    """
    import json
    import os
    
    # First try environment variable (set by main process)
    memory_file = os.environ.get('SHARED_MEMORY_FILE')
    if memory_file and os.path.exists(memory_file):
        with open(memory_file, 'r') as f:
            data = json.load(f)
            return list(data.keys())
    return []


# def get_shared_memory_value(key: str):
#     """Get a value from shared memory by key. Returns the value or None.
    
#     Args:
#         key: The key to look up (e.g., "team0-step1-MagenticMemoryOrchestrator")
    
#     Returns:
#         The value associated with the key, or None if not found.
    
#     Example: value = get_shared_memory_value("team0-step1-MagenticMemoryOrchestrator")
    
#     Note: The shared memory is stored in a JSON file (.shared_memory.json)
#     The path is set via SHARED_MEMORY_FILE environment variable.
#     """
#     import json
#     import os
    
#     # First try environment variable (set by main process)
#     memory_file = os.environ.get('SHARED_MEMORY_FILE')
#     if memory_file and os.path.exists(memory_file):
#         with open(memory_file, 'r') as f:
#             data = json.load(f)
#             return data.get(key)
    
#     return None


def get_shared_memory_value(key: str):
    """Get a value from shared memory by key. Returns the value or None.
    
    Args:
        key: The key to look up (e.g., "team0-step1-MagenticMemoryOrchestrator")
    
    Returns:
        The value associated with the key, or None if not found.
    
    Example: value = get_shared_memory_value("team0-step1-MagenticMemoryOrchestrator")
    
    Note: The shared memory is stored in a JSON file (.shared_memory.json)
    The path is set via SHARED_MEMORY_FILE environment variable.
    """
    memory_file = os.environ.get("SHARED_MEMORY_FILE")
    value = None
    if memory_file and os.path.exists(memory_file):
        with open(memory_file, "r") as f:
            data = json.load(f)
            value = data.get(key)

    # Also log a memory_select event into the shared traces file, if configured.
    try:
        step_env = os.environ.get("STEP_NUM")
        team_env = os.environ.get("TEAM_IDX")
        try:
            step_num = int(step_env) if step_env is not None else None
        except ValueError:
            step_num = None
        try:
            team_idx = int(team_env) if team_env is not None else None
        except ValueError:
            team_idx = None
        event = {
            "event": "memory_select",
            "timestamp": datetime.utcnow().isoformat(),
            "team_idx": team_idx,
            "step_num": step_num,
            "key": key,
            "found": value is not None,
        }
        _append_memory_event(event)
    except Exception:
        # Selection tracing must not break main run.
        pass

    return value


_memory_events_path: Optional[str] = None
_memory_events_fh = None


def _append_memory_event(event: dict) -> None:
    global _memory_events_path, _memory_events_fh

    events_file = os.environ.get("MEMORY_EVENTS_FILE")
    if not events_file:
        return

    if _memory_events_fh is None or _memory_events_path != events_file:
        if _memory_events_fh is not None:
            try:
                _memory_events_fh.close()
            except Exception:
                pass
        _memory_events_path = events_file
        # Buffered append; we flush on process exit.
        _memory_events_fh = open(events_file, "a", encoding="utf-8")

    _memory_events_fh.write(json.dumps(event, ensure_ascii=False) + "\n")


@atexit.register
def _close_memory_events_file() -> None:
    global _memory_events_fh
    if _memory_events_fh is None:
        return
    try:
        _memory_events_fh.flush()
        _memory_events_fh.close()
    except Exception:
        pass
    _memory_events_fh = None


# Create a context variable to hold the current team's log file and the current team id.
current_log_file = contextvars.ContextVar("current_log_file", default=None)
current_team_id = contextvars.ContextVar("current_team_id", default=None)
current_shared_memory = contextvars.ContextVar("current_shared_memory", default=None)
current_step_num = contextvars.ContextVar("current_step_num", default=None)

# Track whether we've seen (and intentionally skipped) the initial user message
# for each team. This prevents the original problem question from being stored
# in shared memory, while still allowing later user-like content if needed.
_initial_user_message_skipped: Dict[int, bool] = {}


class SharedMemory:
    """Thread-safe shared memory for parallel teams to store and retrieve outputs.

    This variant adds an LLM-based verifier so that only useful, plausible
    states are persisted and surfaced to other teams, and also logs structured
    events for downstream training.
    """
    
    def __init__(
        self,
        persistence_file: str = ".shared_memory.json",
        summarizer_client: Optional[ChatCompletionClient] = None,
        verifier_client: Optional[ChatCompletionClient] = None,
        use_decision_transformer: bool = False,
        decision_transformer: Optional[Any] = None,
        events_file: Optional[str] = None,
        enable_trace_files: bool = True,
        verbose: bool = True,
    ):
        self._memory: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._step_counters: Dict[int, int] = {}  # Track step numbers per team
        self._persistence_file = persistence_file
        self._summarizer_client = summarizer_client
        # Default verifier to the summarizer/orchestrator client if not provided explicitly.
        self._verifier_client = verifier_client or summarizer_client
        # Optional alternative verifier path (future): a learned decision transformer.
        # For now we keep the plumbing but use a random decision to avoid loading
        # additional model weights.
        self._use_decision_transformer = use_decision_transformer
        self._decision_transformer = decision_transformer
        self._enable_trace_files = enable_trace_files
        self._verbose = verbose

        # Where to log structured memory events (JSONL).
        self._events_file = (events_file or os.environ.get("MEMORY_EVENTS_FILE")) if enable_trace_files else None
        self._events_enabled = bool(self._events_file)
        # Placeholder probability for the decision-transformer path.
        self._dt_keep_prob = 0.5
        # Memory bank embedding
        self._memory_bank_embedding = None
        # Where to write DT trace snapshots (alongside shared memory file).
        self._traces_dir = None
        if self._use_decision_transformer and enable_trace_files:
            self._traces_dir = os.path.join(os.path.dirname(os.path.abspath(self._persistence_file)), "traces")
            os.makedirs(self._traces_dir, exist_ok=True)
        # Verification stats + trace indexing
        self._verify_total = 0
        self._verify_kept = 0
        if self._use_decision_transformer and enable_trace_files and self._decision_transformer is not None:
            scenario_root = os.path.dirname(os.path.dirname(os.path.abspath(self._persistence_file)))
            model_path = os.path.join(scenario_root, "full_model.pt")
            if not os.path.exists(model_path):
                tmp_path = model_path + ".tmp"
                self._decision_transformer.save(tmp_path)
                os.replace(tmp_path, model_path)

    @property
    def events_enabled(self) -> bool:
        return self._events_enabled
    
    async def _persist_to_file(self):
        """Persist memory to JSON file for cross-process access."""
        try:
            with open(self._persistence_file, 'w') as f:
                json.dump(self._memory, f)
        except Exception as e:
            # Don't fail if file write fails
            print(f"Warning: Failed to persist shared memory to file: {e}", flush=True)

    def _log_event(self, event: Dict[str, Any]) -> None:
        """Append a single JSON event to the events file, if configured."""
        if not self._events_file:
            return
        try:
            # Ensure directory exists (defensive)
            os.makedirs(os.path.dirname(self._events_file), exist_ok=True)
            with open(self._events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            # Logging should never break the main run.
            print(f"Warning: Failed to log shared memory event: {e}", flush=True)
    
    async def _summarize_step(self, source: str, content: str) -> str:
        if self._summarizer_client is None:
            # Fallback: crude summary if no client is available.
            text = content_to_str(content) if not isinstance(content, str) else content
            return text[:80].replace("\n", " ")
        
        messages = [
            UserMessage(
                content=(
                    "Summarize the agent step purpose and outcome in 15–20 words as a concise phrase, "
                    "no quotes or trailing punctuation.\n\n"
                    f"Source: {source}\n\nOutput:\n{content[:2000]}"
                ),
                source="SharedMemory",
            )
        ]
        response = await self._summarizer_client.create(messages)
        assert isinstance(response.content, str)
        return response.content.strip().replace("\n", " ")
    
    async def _verify_step_with_llm(
        self,
        summary: str,
        source: str,
        content: str,
        agent_input: Optional[str] = None,
        agent_input_source: Optional[str] = None,
    ) -> bool:
        """Ask an LLM whether this step is useful enough to store for other teams.

        We want to filter out clearly useless states such as failed code runs,
        empty or obviously failed web searches, and generic error messages,
        while keeping plausible results and non-trivial reasoning.
        """
        if self._verifier_client is None:
            # If there is no verifier configured, default to keeping the step.
            return True

        truncated_content = content[:8000] if isinstance(content, str) else content_to_str(content)[:8000]
        truncated_input = None
        if agent_input is not None:
            truncated_input = agent_input[:8000] if isinstance(agent_input, str) else content_to_str(agent_input)[:8000]
        input_header = ""
        if truncated_input:
            src = agent_input_source or "unknown"
            input_header = f"- agent input (from {src}):\n{truncated_input}\n\n"
        messages = [
            UserMessage(
                content=(
                    "You are a strict memory filter deciding whether to store an agent step for reuse "
                    "by other teams.\n\n"
                    "You are given:\n"
                    f"- step summary: \"{summary}\"\n"
                    f"- agent/source: {source}\n"
                    f"{input_header}"
                    f"- step content:\n{truncated_content}\n\n"
                    "Return ONLY `YES` or `NO`.\n\n"
                    "Answer `YES` only if this step contains a clearly useful result or information that\n"
                    "- reports a successful code execution with a concrete output, OR\n"
                    "- contains retrieved data from the web or files, OR\n"
                    "- captures non-trivial reasoning or insights that could help another team.\n\n"
                    "Answer `NO` for:\n"
                    "- failed or aborted code runs,\n"
                    "- error messages or stack traces with no successful result,\n"
                    "- clearly failed or empty web/file searches,\n"
                    "- meta-chatter, apologies, or generic planning without concrete results."
                ),
                source="SharedMemoryVerifier",
            )
        ]
        try:
            response = await self._verifier_client.create(messages)
            raw = response.content if isinstance(response.content, str) else str(response.content)
            decision = raw.strip().upper()
            return decision.startswith("YES")
        except Exception as e:
            # If verification fails, don't store to avoid polluting memory.
            print(f"Warning: verifier failed for shared memory entry: {e}", flush=True)
            return False

    async def _verify_step_with_decision_transformer(
        self,
        summary: str,
        source: str,
        content: str,
        agent_input: Optional[str] = None,
        agent_input_source: Optional[str] = None,
        team_idx: Optional[int] = None,
        step_num: Optional[int] = None,
    ) -> bool:
        """Verify via a decision transformer.
        """
        # if self._decision_transformer is None:
            # return random.random() < self._dt_keep_prob

        dt = self._decision_transformer
        input_text = agent_input or ""
        async with self._lock:
            memory_bank = self._memory_bank_embedding  # (1, M, D) or None

        with torch.no_grad():
            # Each returns (1, 1, D) via dt.embed_text(..., num_tokens=1)
            agent_input_embedding = dt.embed_text(input_text, num_tokens=1)
            agent_output_embedding = dt.embed_text(content, num_tokens=1)
            summary_embedding = dt.embed_text(summary, num_tokens=1)

            x = torch.cat([agent_input_embedding, agent_output_embedding, summary_embedding], dim=1)  # (1, 3, D)

            # forward() returns logits over valid tokens (YES/NO) after our wrapper fixes
            valid_logits = dt(x, memory_bank=memory_bank)
            if valid_logits.dim() == 2:
                valid_logits = valid_logits.squeeze(0)

            decision_idx = dt.sample_decision(valid_logits, temperature=1.2, epsilon=0.1)
            decision_str = dt.valid_strings[decision_idx] if hasattr(dt, "valid_strings") else str(decision_idx)

        keep = str(decision_str).strip().upper().startswith("YES")
        # Save trace state with correct identifiers (avoid overwrites).
        if self._enable_trace_files and self._traces_dir is not None and team_idx is not None and step_num is not None:
            dt.save_trace_state(
                step_num=step_num,
                team_idx=team_idx,
                trace_dir=self._traces_dir,
                memory_bank=memory_bank,
                verification_embeddings=x,
                valid_logits=valid_logits,
                decision=decision_idx,
                decision_str=decision_str,
            )
        if keep:
            # Maintain an in-memory bank of embeddings for future decisions.
            async with self._lock:
                if self._memory_bank_embedding is None:
                    self._memory_bank_embedding = summary_embedding
                else:
                    self._memory_bank_embedding = torch.cat([self._memory_bank_embedding, summary_embedding], dim=1)
        return keep

    async def _verify_step(
        self,
        summary: str,
        source: str,
        content: str,
        agent_input: Optional[str] = None,
        agent_input_source: Optional[str] = None,
        team_idx: Optional[int] = None,
        step_num: Optional[int] = None,
    ) -> bool:
        """Verify whether to store a step using configured verifier backend."""
        if self._use_decision_transformer:
            return await self._verify_step_with_decision_transformer(
                summary,
                source,
                content,
                agent_input=agent_input,
                agent_input_source=agent_input_source,
                team_idx=team_idx,
                step_num=step_num,
            )
        return await self._verify_step_with_llm(
            summary, source, content, agent_input=agent_input, agent_input_source=agent_input_source
        )
    
    async def store(
        self,
        team_idx: int,
        source: str,
        content: str,
        agent_input: Optional[str] = None,
        agent_input_source: Optional[str] = None,
    ) -> str:
        """Store a message in shared memory (if verified as useful) and return the key.

        The verifier is consulted after summarization; if it returns NO, the
        step is skipped and not written to the JSON memory file.
        """
        # Reserve a stable step number under lock so logs and memory align.
        async with self._lock:
            step_env = os.environ.get("STEP_NUM")
            if step_env is not None:
                try:
                    step_num = int(step_env)
                except ValueError:
                    self._step_counters[team_idx] = self._step_counters.get(team_idx, 0) + 1
                    step_num = self._step_counters[team_idx]
            else:
                self._step_counters[team_idx] = self._step_counters.get(team_idx, 0) + 1
                step_num = self._step_counters[team_idx]

        # Summarize + verify outside lock to avoid blocking other teams.
        summary = await self._summarize_step(source, content)
        key = f"team{team_idx}-step{step_num} - {summary}"
        keep = await self._verify_step(
            summary,
            source,
            content,
            agent_input=agent_input,
            agent_input_source=agent_input_source,
            team_idx=team_idx,
            step_num=step_num,
        )

        async with self._lock:
            self._verify_total += 1
            if keep:
                self._verify_kept += 1
            if self._verify_total % 50 == 0:
                pct = (self._verify_kept / self._verify_total) * 100.0 if self._verify_total else 0.0
                if self._verbose:
                    print(f"SharedMemory: keep rate so far: {self._verify_kept}/{self._verify_total} ({pct:.1f}%)")

        # Log training event outside lock.
        self._log_event(
            {
                "event": "memory_store",
                "timestamp": datetime.utcnow().isoformat(),
                "team_idx": team_idx,
                "step_num": step_num,
                "key": key,
                "source": source,
                "summary": summary,
                "content": content,
                "agent_input": agent_input,
                "agent_input_source": agent_input_source,
                "kept": keep,
            }
        )

        if not keep:
            if self._verbose:
                print(f"SharedMemory: REJECTED step for storage: {key}")
            return key

        if self._verbose:
            print(f"SharedMemory: ACCEPTED step for storage: {key}")
        async with self._lock:
            self._memory[key] = content
            await self._persist_to_file()

        return key
    
    async def get(self, key: str) -> Optional[str]:
        """Retrieve a value from shared memory by key."""
        async with self._lock:
            return self._memory.get(key)
    
    async def get_all(self) -> Dict[str, str]:
        """Get a copy of all shared memory."""
        async with self._lock:
            return self._memory.copy()
    
    async def get_keys(self) -> List[str]:
        """Get a list of all keys in shared memory."""
        async with self._lock:
            return list(self._memory.keys())
    
    def __len__(self) -> int:
        """Return the number of entries in shared memory (not thread-safe, for debugging)."""
        return len(self._memory)


# Save the original print function and event_logger.info method.
original_print = builtins.print
original_agentchat_event_logger_info = agentchat_event_logger.info
original_core_event_logger_info = core_event_logger.info

_tee_last_flush: Dict[int, float] = {}


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
            print("error in LogHandler.emit", flush=True)
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
        flush_requested = bool(kwargs.get("flush", False))
        now = time.monotonic()
        log_id = id(log_file)
        last = _tee_last_flush.get(log_id, 0.0)
        if flush_requested or (now - last) >= 1.0:
            log_file.flush()
            _tee_last_flush[log_id] = now


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


def enable_team_logging() -> None:
    builtins.print = tee_print
    agentchat_event_logger.info = team_specific_agentchat_event_logger_info
    core_event_logger.info = team_specific_core_event_logger_info


def disable_team_logging() -> None:
    builtins.print = original_print
    agentchat_event_logger.info = original_agentchat_event_logger_info
    core_event_logger.info = original_core_event_logger_info


async def intercept_messages_for_shared_memory(
    stream,
    team_idx: int,
    shared_memory: SharedMemory
):
    """Intercept messages from a stream and store them in shared memory (if verified).

    This also logs a generic step_message event for every intercepted
    message so downstream training pipelines can reconstruct the full
    problem-solving trace per team, not just the entries that were
    written into shared memory.
    """
    # Track the most recent message content to approximate the "input" that
    # triggered the next agent output. We do NOT persist this into shared
    # memory; it is only passed into the verifier decision.
    last_message_source: Optional[str] = None
    last_message_content: Optional[str] = None

    async for message in stream:
        # Skip TaskResult - it's the final result, not a message to store
        # Also skip StopMessage as it's just a signal
        if isinstance(message, (TaskResult, StopMessage)):
            yield message
            continue
        
        # Extract source and content from message
        source = getattr(message, 'source', 'unknown')

        # Skip the very first "user" message per team so we do not store the
        # original GAIA question in shared memory. That question is already
        # available to all teams via the task prompt and does not need to be
        # redundantly persisted as memory.
        if source == "user":
            if not _initial_user_message_skipped.get(team_idx, False):
                _initial_user_message_skipped[team_idx] = True
                yield message
                continue

        # Skip MagenticMemoryOrchestrator messages for memory storage, but
        # still log them as step events so the trace is complete.
        skip_memory_store = False
        if source == "MagenticMemoryOrchestrator":
            skip_memory_store = True
        
        # Convert message content to string
        if hasattr(message, 'to_model_text'):
            content = message.to_model_text()
        elif hasattr(message, 'content'):
            content = content_to_str(message.content)
        else:
            content = str(message)

        # Approximate the input that triggered this output as the immediately
        # preceding message (if from a different source).
        agent_input = None
        agent_input_source = None
        if last_message_content is not None and last_message_source is not None and last_message_source != source:
            agent_input = last_message_content
            agent_input_source = last_message_source

        # Log a generic step_message event for every intercepted message.
        step_env = os.environ.get("STEP_NUM")
        try:
            step_num = int(step_env) if step_env is not None else None
        except ValueError:
            step_num = None
        if shared_memory.events_enabled:
            try:
                event = {
                    "event": "step_message",
                    "timestamp": datetime.utcnow().isoformat(),
                    "team_idx": team_idx,
                    "step_num": step_num,
                    "source": source,
                    "message_type": type(message).__name__,
                    "content": content,
                }
                shared_memory._log_event(event)
            except Exception:
                # Tracing must never break the main run.
                pass
        
        # Store in shared memory (non-blocking, store in background) unless
        # explicitly skipped.
        if not skip_memory_store:
            try:
                await shared_memory.store(
                    team_idx,
                    source,
                    content,
                    agent_input=agent_input,
                    agent_input_source=agent_input_source,
                )
            except Exception as e:
                # Log but don't fail on shared memory errors
                print(f"Warning: Failed to store message in shared memory: {e}", flush=True)
        
        # Update last message trackers after processing this message.
        last_message_source = source
        last_message_content = content

        # Yield the message to continue the stream
        yield message


async def run_team(
    team: MagenticMemoryGroupChat, 
    team_idx: int, 
    task: str, 
    cancellation_token: CancellationToken, 
    logfile,
    shared_memory: Optional[SharedMemory] = None
):
    token_logfile = current_log_file.set(logfile)
    token_team_id = current_team_id.set(team_idx)
    token_shared_memory = None
    if shared_memory is not None:
        token_shared_memory = current_shared_memory.set(shared_memory)
    
    # Also set as environment variable for Console UI fallback
    # (since context variables may not be accessible in all async contexts)
    os.environ['TEAM_IDX'] = str(team_idx)
    
    try:
        # Get the raw stream
        stream = team.run_stream(
            task=task.strip(),
            cancellation_token=cancellation_token
        )
        
        # Intercept messages if shared memory is provided
        if shared_memory is not None:
            stream = intercept_messages_for_shared_memory(stream, team_idx, shared_memory)
        
        # Pass to Console for display
        task_result = await Console(stream)
        return team_idx, task_result
    finally:
        current_log_file.reset(token_logfile)
        current_team_id.reset(token_team_id)
        if token_shared_memory is not None:
            current_shared_memory.reset(token_shared_memory)
        if logfile is not None:
            logfile.close()


async def aggregate_final_answer(
    task: str,
    client: ChatCompletionClient,
    team_results,
    source: str = "Aggregator",
    cancellation_token: Optional[CancellationToken] = None,
) -> str:
        """
        team_results: {"team_key": TaskResult}
        team_completion_order: The order in which the teams completed their tasks
        """

        if len(team_results) == 1:
            final_answer = list(team_results.values())[0].messages[-1].content
            aggregator_logger.info(f"{source} (Response):\n{final_answer}")
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
                elif m.source == "MagenticMemoryOrchestrator":
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
            payload += (
                f"\n{'-'*75} \n"
                f"Team #{team_idx} stop reason:\n"
                f"\n{team_results[team_idx].stop_reason}\n"
            )
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
        aggregator_logger.info(f"{source} (Response):\n{final_answer}")

        return re.sub(r"FINAL ANSWER:", "FINAL AGGREGATED ANSWER:", response.content)


async def main(
    num_teams: int,
    num_answers: int,
    use_decision_transformer: bool = False,
    decision_transformer_weights_path: str = None,
    no_trace: bool = False,
    no_memory_events: bool = False,
) -> None:

    # Track overall scenario runtime
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

    # Prepare the prompt
    filename_prompt = ""
    if len(filename) > 0:
        filename_prompt = f"The question is about a file, document or image, which can be accessed by the filename '{filename}' in the current working directory."
    
    task = f"{prompt}\n\n{filename_prompt}"

    logs_dir = "logs"
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)

    # Create shared memory for inter-team communication
    # Use absolute path for JSON file so it's accessible from executor subprocesses
    # Store it in the current working directory where the script runs
    memory_file_path = os.path.abspath('.shared_memory.json')

    events_file_path = None
    if not no_memory_events:
        # Structured memory event log (JSONL). This lives alongside the shared
        # memory file in the per-run working directory and is intended for
        # offline training of a verifier model.
        events_file_path = os.path.abspath("memory_events.jsonl")
        os.environ["MEMORY_EVENTS_FILE"] = events_file_path
    else:
        os.environ.pop("MEMORY_EVENTS_FILE", None)

    decision_transformer = None
    if use_decision_transformer:
        DecisionTransformer = _load_decision_transformer()
        if DecisionTransformer is None:
            raise RuntimeError("DecisionTransformer could not be imported (got None).")
        # One model instance shared across teams; it uses SharedMemory's bank.
        decision_transformer = DecisionTransformer(team_idx=0, query=prompt)
        weights_path = decision_transformer_weights_path or os.environ.get("DECISION_TRANSFORMER_WEIGHTS")
        if weights_path:
            resolved_path = weights_path
            if not os.path.isabs(resolved_path) and not os.path.isfile(resolved_path):
                repo_root = "/autogen_python/packages/agbench/benchmarks/AssistantBench"
                candidate = os.path.join(repo_root, resolved_path)
                if os.path.isfile(candidate):
                    resolved_path = candidate
            if os.path.isfile(resolved_path):
                decision_transformer.load(resolved_path)
            else:
                print(f"Warning: decision transformer weights not found at '{weights_path}', skipping load.", flush=True)
        decision_transformer.eval()
        print(f"DecisionTransformer device: {decision_transformer.device}")

    shared_memory = SharedMemory(
        persistence_file=memory_file_path,
        summarizer_client=orchestrator_client,
        verifier_client=orchestrator_client,
        use_decision_transformer=use_decision_transformer,
        decision_transformer=decision_transformer,
        events_file=events_file_path,
        enable_trace_files=not no_trace,
        verbose=not no_trace,
    )
    
    # Set environment variable so executor subprocesses can find the file
    os.environ['SHARED_MEMORY_FILE'] = memory_file_path
    
    # Initialize the JSON file so it exists when executor tries to read it
    with open(memory_file_path, 'w') as f:
        json.dump({}, f)

    teams = []
    async_tasks = []
    tokens = []

    # For logging the first team to finish
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
        team = MagenticMemoryGroupChat(
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
            run_team(team, team_idx, task, cancellation_token, logfile, shared_memory)
        )
        async_tasks.append(async_task)

    # Wait until at least num_answers tasks have completed.
    team_results = {}
    for future in asyncio.as_completed(async_tasks):
        try:
            team_id, result = await future
            team_results[team_id] = result

            # Record the first team to finish (and its answer + runtime)
            if first_team_id is None:
                try:
                    if result.messages:
                        last_msg = result.messages[-1]
                        answer = getattr(last_msg, "content", None)
                        if isinstance(answer, str):
                            first_team_id = team_id
                            first_team_answer = answer
                            first_team_runtime = time.time() - scenario_start_time
                except Exception:
                    # Do not fail the run if we can't extract the first answer.
                    pass
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

    final_answer = await aggregate_final_answer(
        prompt, orchestrator_client, team_results
    )
    print(final_answer)

    # Compute overall runtime
    overall_runtime = time.time() - scenario_start_time

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

    # Log final verification keep rate
    total = getattr(shared_memory, "_verify_total", 0)
    kept = getattr(shared_memory, "_verify_kept", 0)
    rejected = total - kept if total else 0
    pct = (kept / total) * 100.0 if total else 0.0
    print(f"SharedMemory: FINAL keep rate: {kept}/{total} ({pct:.1f}%)", flush=True)
    print(f"SharedMemory: FINAL counts: kept={kept} rejected={rejected}", flush=True)
    
    # Note: we intentionally leave the shared memory JSON file around so it can
    # be inspected after the run if desired.


if __name__ == "__main__":
    num_teams = 3
    num_answers = 3
    use_decision_transformer = True
    decision_transformer_weights_path = os.environ.get("DECISION_TRANSFORMER_WEIGHTS")
    no_trace = False
    no_memory_events = False

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


    asyncio.run(
        main(
            num_teams,
            num_answers,
            use_decision_transformer=use_decision_transformer,
            decision_transformer_weights_path=decision_transformer_weights_path,
            no_trace=no_trace,
            no_memory_events=no_memory_events,
        )
    )

