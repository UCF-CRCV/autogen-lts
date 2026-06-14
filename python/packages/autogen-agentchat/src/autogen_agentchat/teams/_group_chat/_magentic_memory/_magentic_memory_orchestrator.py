import asyncio
import json
import os
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

from autogen_core import AgentId, CancellationToken, DefaultTopicId, MessageContext, event, rpc
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    UserMessage,
)
from autogen_core.utils import extract_json_from_str

from .... import TRACE_LOGGER_NAME
from ....base import Response, TerminationCondition
from ....messages import (
    BaseAgentEvent,
    BaseChatMessage,
    HandoffMessage,
    MessageFactory,
    MultiModalMessage,
    SelectSpeakerEvent,
    StopMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from ....state import MagenticMemoryOrchestratorState
from ....utils import remove_images
from .._base_group_chat_manager import BaseGroupChatManager
from .._events import (
    GroupChatAgentResponse,
    GroupChatMessage,
    GroupChatRequestPublish,
    GroupChatReset,
    GroupChatStart,
    GroupChatTeamResponse,
    GroupChatTermination,
    SerializableException,
)
from ._prompts import (
    ORCHESTRATOR_FINAL_ANSWER_PROMPT,
    ORCHESTRATOR_PROGRESS_LEDGER_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT,
    LedgerEntry,
)

trace_logger = logging.getLogger(TRACE_LOGGER_NAME)

# Helper to get context variables dynamically
def _get_context_vars():
    """Get context variables if available."""
    try:
        import sys
        for module_name in sys.modules:
            if 'MemoryParallelAgents' in module_name and 'scenario' in module_name:
                module = sys.modules[module_name]
                current_team_id = getattr(module, 'current_team_id', None)
                current_step_num = getattr(module, 'current_step_num', None)
                return current_team_id, current_step_num
        # Fallback: try direct import
        try:
            from agbench.benchmarks.GAIA.Templates.MemoryParallelAgents.scenario import current_team_id, current_step_num
            return current_team_id, current_step_num
        except ImportError:
            return None, None
    except Exception:
        return None, None


class MagenticMemoryOrchestrator(BaseGroupChatManager):
    """The MagenticMemoryOrchestrator manages a group chat with ledger based orchestration."""

    def __init__(
        self,
        name: str,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_names: List[str],
        participant_descriptions: List[str],
        max_turns: int | None,
        message_factory: MessageFactory,
        model_client: ChatCompletionClient,
        max_stalls: int,
        final_answer_prompt: str,
        output_message_queue: asyncio.Queue[BaseAgentEvent | BaseChatMessage | GroupChatTermination],
        termination_condition: TerminationCondition | None,
        emit_team_events: bool,
    ):
        super().__init__(
            name,
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_names,
            participant_descriptions,
            output_message_queue,
            termination_condition,
            max_turns,
            message_factory,
            emit_team_events=emit_team_events,
        )
        self._model_client = model_client
        self._max_stalls = max_stalls
        self._final_answer_prompt = final_answer_prompt
        self._max_json_retries = 10
        self._task = ""
        self._facts = ""
        self._plan = ""
        self._n_rounds = 0
        self._n_stalls = 0

        # Produce a team description. Each agent sould appear on a single line.
        self._team_description = ""
        for topic_type, description in zip(self._participant_names, self._participant_descriptions, strict=True):
            self._team_description += re.sub(r"\s+", " ", f"{topic_type}: {description}").strip() + "\n"
        self._team_description = self._team_description.strip()

    def _get_task_ledger_facts_prompt(self, task: str) -> str:
        return ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT.format(task=task)

    def _get_task_ledger_plan_prompt(self, team: str) -> str:
        return ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT.format(team=team)

    def _get_task_ledger_full_prompt(self, task: str, team: str, facts: str, plan: str) -> str:
        return ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT.format(task=task, team=team, facts=facts, plan=plan)

    def _get_progress_ledger_prompt(self, task: str, team: str, names: List[str]) -> str:
        return ORCHESTRATOR_PROGRESS_LEDGER_PROMPT.format(task=task, team=team, names=", ".join(names))

    def _get_task_ledger_facts_update_prompt(self, task: str, facts: str) -> str:
        return ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT.format(task=task, facts=facts)

    def _get_task_ledger_plan_update_prompt(self, team: str) -> str:
        return ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT.format(team=team)

    def _get_final_answer_prompt(self, task: str) -> str:
        if self._final_answer_prompt == ORCHESTRATOR_FINAL_ANSWER_PROMPT:
            return ORCHESTRATOR_FINAL_ANSWER_PROMPT.format(task=task)
        else:
            return self._final_answer_prompt

    async def _log_message(self, log_message: str) -> None:
        """Log a message with team and step number headers."""
        team_idx: Any = "?"
        team_env = os.environ.get("TEAM_IDX")
        if team_env is not None:
            try:
                team_idx = int(team_env)
            except ValueError:
                team_idx = team_env
        current_team_id_var, _ = _get_context_vars()
        if team_idx == "?" and current_team_id_var is not None:
            try:
                team_idx = current_team_id_var.get()
            except LookupError:
                pass
        
        step_num = self._n_rounds
        header = f"[Team {team_idx} | Step {step_num}]"
        trace_logger.debug(f"{header} {log_message}")

    def _append_memory_event(self, event: Dict[str, Any]) -> None:
        events_file = os.environ.get("MEMORY_EVENTS_FILE")
        if not events_file:
            return
        with open(events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _query_shared_memory(self) -> tuple[List[str], Dict[str, str]]:
        """Query shared memory file and return keys and all values.
        
        Returns:
            tuple: (list of all keys, dict of all key-value pairs)
        """
        try:
            memory_file = os.environ.get("SHARED_MEMORY_FILE")
            if not memory_file:
                trace_logger.debug("SHARED_MEMORY_FILE not set in environment")
                return [], {}
            if not os.path.exists(memory_file):
                trace_logger.debug("Shared memory file not found: %s", memory_file)
                return [], {}
            
            with open(memory_file, "r") as f:
                data = json.load(f)
            
            if not data:
                return [], {}
            
            keys = list(data.keys())
            trace_logger.debug("Found %s shared memory entries", len(keys))
            return keys, data
        except Exception as e:
            team_idx = '?'
            current_team_id_var, _ = _get_context_vars()
            if current_team_id_var is not None:
                try:
                    team_idx = current_team_id_var.get()
                except LookupError:
                    pass
            trace_logger.debug(f"[Team {team_idx}] Failed to query shared memory: {e}")
            return [], {}
    
    def _parse_selected_keys(self, response_text: str) -> List[str]:
        """Parse SELECTED_KEYS from LLM response.
        
        Looks for patterns like:
        - SELECTED_KEYS = ["key1"]
        - SELECTED_KEYS = ['key1', 'key2']
        - SELECTED_KEYS = [key1, key2]
        - SELECTED_KEYS=["key1"]
        
        Handles commas within quoted strings properly.
        
        Returns:
            List of selected keys, empty list if none found
        """
        # Pattern to match SELECTED_KEYS = [...] with various formats
        # This captures the content inside brackets, handling quoted and unquoted keys
        pattern = r'SELECTED_KEYS\s*=\s*\[([^\]]+)\]'
        match = re.search(pattern, response_text, re.IGNORECASE)
        
        if match:
            content = match.group(1).strip()
            keys = []
            
            # If content is empty, return empty list
            if not content:
                return []
            
            # Check if it's a single quoted string (e.g., "key with, commas")
            # Match either double or single quotes
            quoted_pattern = r'^["\'](.+)["\']$'
            quoted_match = re.match(quoted_pattern, content.strip())
            if quoted_match:
                # Single key with quotes
                key = quoted_match.group(1)
                keys.append(key)
            else:
                # Multiple keys or unquoted - need to parse carefully
                # Split by comma, but respect quotes
                i = 0
                current_key = ""
                in_quotes = False
                quote_char = None
                
                while i < len(content):
                    char = content[i]
                    
                    if char in ['"', "'"] and (i == 0 or content[i-1] != '\\'):
                        if not in_quotes:
                            in_quotes = True
                            quote_char = char
                        elif char == quote_char:
                            in_quotes = False
                            quote_char = None
                        current_key += char
                    elif char == ',' and not in_quotes:
                        # Comma outside quotes - separator
                        if current_key.strip():
                            keys.append(self._strip_surrounding_quotes(current_key))
                        current_key = ""
                    else:
                        current_key += char
                    
                    i += 1
                
                # Add the last key
                if current_key.strip():
                    keys.append(self._strip_surrounding_quotes(current_key))
            
            # Clean up keys - remove quotes and whitespace
            cleaned_keys = []
            for key in keys:
                key = self._strip_surrounding_quotes(key)
                if key:  # Only add non-empty keys
                    cleaned_keys.append(key)
            
            return cleaned_keys
        
        return []

    @staticmethod
    def _strip_surrounding_quotes(value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            return value[1:-1]
        return value
    
    def _normalize_key(self, key: str) -> str:
        """Normalize a key for matching (remove extra quotes, escape chars, etc.)."""
        # Remove escaped quotes and normalize quotes
        key = key.replace('\\"', '"').replace("\\'", "'")
        # Remove leading/trailing whitespace
        key = key.strip()
        return key
    
    def _find_matching_key(self, selected_key: str, available_keys: List[str]) -> Optional[str]:
        """Find the best matching key from available keys, handling quote variations."""
        normalized_selected = self._normalize_key(selected_key)
        
        # Try exact match first
        if normalized_selected in available_keys:
            return normalized_selected
        
        # Try matching after normalization
        for key in available_keys:
            if self._normalize_key(key) == normalized_selected:
                return key
        
        # Try fuzzy match - check if the selected key contains the team-step part
        # Extract team-step pattern (e.g., "team2-step1")
        team_step_match = re.search(r'team\d+-step\d+', normalized_selected)
        if team_step_match:
            team_step = team_step_match.group(0)
            # Find keys that start with the same team-step
            for key in available_keys:
                if key.startswith(team_step):
                    # Check if the rest is similar (fuzzy match)
                    if len(normalized_selected) > len(team_step) + 10:  # Has meaningful content
                        # Extract the summary part and compare
                        selected_summary = normalized_selected[len(team_step):].strip()
                        key_summary = key[len(team_step):].strip()
                        # If summaries are similar (one contains the other), consider it a match
                        if selected_summary.lower() in key_summary.lower() or key_summary.lower() in selected_summary.lower():
                            return key
        
        return None
    
    def _get_shared_memory_values(self, keys: List[str], data: Dict[str, str]) -> Dict[str, str]:
        """Fetch values for given keys from shared memory.
        
        Args:
            keys: List of keys to fetch (may not match exactly due to quote escaping)
            data: Dict of all key-value pairs

        Returns:
            Dict mapping original keys to values (only includes keys that exist)
        """
        try:
            result: Dict[str, str] = {}
            available_keys = list(data.keys())
            
            for selected_key in keys:
                # Try to find matching key
                matching_key = self._find_matching_key(selected_key, available_keys)
                if matching_key:
                    result[matching_key] = data[matching_key]
                    print(f"[DEBUG] Matched key '{selected_key[:50]}...' -> '{matching_key[:50]}...'", flush=False)
                else:
                    print(f"[DEBUG] WARNING: Could not match key '{selected_key[:100]}...'", flush=False)
                    # print(f"[DEBUG] Available keys: {[k[:50] for k in available_keys[:5]]}", flush=True)
            
            return result
        except Exception as e:
            team_idx = '?'
            current_team_id_var, _ = _get_context_vars()
            if current_team_id_var is not None:
                try:
                    team_idx = current_team_id_var.get()
                except LookupError:
                    pass
            trace_logger.debug(f"[Team {team_idx}] Failed to fetch shared memory values: {e}")
            return {}

    def _log_memory_use_events(self, selected_values: Dict[str, str]) -> None:
        """Log structured 'memory_use' events for selected shared memory entries.

        This is intentionally lightweight so that downstream training pipelines
        can join these events with the per-run memory_store logs.
        """
        if not selected_values:
            return

        events_file = os.environ.get("MEMORY_EVENTS_FILE")
        if not events_file:
            return

        try:
            import json
            from datetime import datetime

            # Determine team index from context vars, if available.
            team_idx: Any = "?"
            team_env = os.environ.get("TEAM_IDX")
            if team_env is not None:
                try:
                    team_idx = int(team_env)
                except ValueError:
                    team_idx = team_env
            current_team_id_var, _ = _get_context_vars()
            if team_idx == "?" and current_team_id_var is not None:
                try:
                    team_idx = current_team_id_var.get()
                except LookupError:
                    team_idx = "?"

            os.makedirs(os.path.dirname(events_file), exist_ok=True)
            with open(events_file, "a", encoding="utf-8") as f:
                for key, value in selected_values.items():
                    event: Dict[str, Any] = {
                        "event": "memory_use",
                        "timestamp": datetime.utcnow().isoformat(),
                        "team_idx": team_idx,
                        # Use the orchestrator's current round as an approximate step index.
                        "step_num": self._n_rounds,
                        "key": key,
                        "source": "MagenticMemoryOrchestrator",
                        "content": value,
                    }
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            # Logging must never break orchestration.
            trace_logger.debug("Failed to log memory_use events: %s", e)

    @rpc
    async def handle_start(self, message: GroupChatStart, ctx: MessageContext) -> None:  # type: ignore
        """Handle the start of a task."""

        # Check if the conversation has already terminated.
        if self._termination_condition is not None and self._termination_condition.terminated:
            early_stop_message = StopMessage(content="The group chat has already terminated.", source=self._name)
            # Signal termination.
            await self._signal_termination(early_stop_message)
            # Stop the group chat.
            return
        assert message is not None and message.messages is not None

        # Validate the group state given all the messages.
        await self.validate_group_state(message.messages)

        # Log the message to the output topic.
        await self.publish_message(message, topic_id=DefaultTopicId(type=self._output_topic_type))
        # Log the message to the output queue.
        for msg in message.messages:
            await self._output_message_queue.put(msg)

        # Outer Loop for first time
        # Create the initial task ledger
        #################################
        # Combine all message contents for task
        self._task = " ".join([msg.to_model_text() for msg in message.messages])
        planning_conversation: List[LLMMessage] = []

        # 1. GATHER FACTS
        # create a closed book task and generate a response and update the chat history
        planning_conversation.append(
            UserMessage(content=self._get_task_ledger_facts_prompt(self._task), source=self._name)
        )
        response = await self._model_client.create(
            self._get_compatible_context(planning_conversation), cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        self._facts = response.content
        planning_conversation.append(AssistantMessage(content=self._facts, source=self._name))

        # 2. CREATE A PLAN
        ## plan based on available information
        planning_conversation.append(
            UserMessage(content=self._get_task_ledger_plan_prompt(self._team_description), source=self._name)
        )
        response = await self._model_client.create(
            self._get_compatible_context(planning_conversation), cancellation_token=ctx.cancellation_token
        )

        assert isinstance(response.content, str)
        self._plan = response.content

        # Kick things off
        self._n_stalls = 0
        await self._reenter_outer_loop(ctx.cancellation_token)

    @event
    async def handle_agent_response(  # type: ignore
        self, message: GroupChatAgentResponse | GroupChatTeamResponse, ctx: MessageContext
    ) -> None:  # type: ignore
        try:
            if not isinstance(message, GroupChatAgentResponse):
                raise RuntimeError("MagenticMemoryOrchestrator does not support GroupChatTeamResponse messages.")
            delta: List[BaseAgentEvent | BaseChatMessage] = []
            if message.response.inner_messages is not None:
                for inner_message in message.response.inner_messages:
                    delta.append(inner_message)
            await self.update_message_thread([message.response.chat_message])
            delta.append(message.response.chat_message)

            if self._termination_condition is not None:
                stop_message = await self._termination_condition(delta)
                if stop_message is not None:
                    # Reset the termination conditions.
                    await self._termination_condition.reset()
                    # Signal termination.
                    await self._signal_termination(stop_message)
                    return

            await self._orchestrate_step(ctx.cancellation_token)
        except Exception as e:
            error = SerializableException.from_exception(e)
            await self._signal_termination_with_error(error)
            # Raise the error to the runtime.
            raise

    async def validate_group_state(self, messages: List[BaseChatMessage] | None) -> None:
        pass

    async def save_state(self) -> Mapping[str, Any]:
        state = MagenticMemoryOrchestratorState(
            message_thread=[msg.dump() for msg in self._message_thread],
            current_turn=self._current_turn,
            task=self._task,
            facts=self._facts,
            plan=self._plan,
            n_rounds=self._n_rounds,
            n_stalls=self._n_stalls,
        )
        return state.model_dump()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        orchestrator_state = MagenticMemoryOrchestratorState.model_validate(state)
        self._message_thread = [self._message_factory.create(message) for message in orchestrator_state.message_thread]
        self._current_turn = orchestrator_state.current_turn
        self._task = orchestrator_state.task
        self._facts = orchestrator_state.facts
        self._plan = orchestrator_state.plan
        self._n_rounds = orchestrator_state.n_rounds
        self._n_stalls = orchestrator_state.n_stalls

    async def select_speaker(self, thread: Sequence[BaseAgentEvent | BaseChatMessage]) -> List[str] | str:
        """Not used in this orchestrator, we select next speaker in _orchestrate_step."""
        return [""]

    async def reset(self) -> None:
        """Reset the group chat manager."""
        self._message_thread.clear()
        if self._termination_condition is not None:
            await self._termination_condition.reset()
        self._n_rounds = 0
        self._n_stalls = 0
        self._task = ""
        self._facts = ""
        self._plan = ""

    async def _reenter_outer_loop(self, cancellation_token: CancellationToken) -> None:
        """Re-enter Outer loop of the orchestrator after creating task ledger."""
        # Reset the agents
        for participant_topic_type in self._participant_name_to_topic_type.values():
            await self._runtime.send_message(
                GroupChatReset(),
                recipient=AgentId(type=participant_topic_type, key=self.id.key),
                cancellation_token=cancellation_token,
            )
        # Reset partially the group chat manager
        self._message_thread.clear()

        # Prepare the ledger
        ledger_message = TextMessage(
            content=self._get_task_ledger_full_prompt(self._task, self._team_description, self._facts, self._plan),
            source=self._name,
        )

        # Save my copy
        await self.update_message_thread([ledger_message])

        # Log it to the output topic.
        await self.publish_message(
            GroupChatMessage(message=ledger_message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )
        # Log it to the output queue.
        await self._output_message_queue.put(ledger_message)

        # Broadcast
        await self.publish_message(
            GroupChatAgentResponse(response=Response(chat_message=ledger_message), name=self._name),
            topic_id=DefaultTopicId(type=self._group_topic_type),
        )

        # Restart the inner loop
        await self._orchestrate_step(cancellation_token=cancellation_token)

    async def _orchestrate_step(self, cancellation_token: CancellationToken) -> None:
        """Implements the inner loop of the orchestrator and selects next speaker."""
        # Check if we reached the maximum number of rounds
        if self._max_turns is not None and self._n_rounds > self._max_turns:
            await self._prepare_final_answer("Max rounds reached.", cancellation_token)
            return
        self._n_rounds += 1
        
        # Set step number in context variable for Console UI to access
        _, current_step_num_var = _get_context_vars()
        if current_step_num_var is not None:
            try:
                current_step_num_var.set(self._n_rounds)
            except Exception:
                pass
        
        # Also set as environment variable for Console UI fallback
        os.environ['STEP_NUM'] = str(self._n_rounds)

        # Update the progress ledger
        context = self._thread_to_context()

        # Step 1: Query shared memory keys and ask orchestrator to select relevant ones
        consult_start = time.perf_counter()
        read_start = time.perf_counter()
        all_keys, shared_memory_data = self._query_shared_memory()
        memory_read_seconds = time.perf_counter() - read_start
        selected_keys = []
        llm_selection_seconds = 0.0
        inject_seconds = 0.0
        selected_values_count = 0
        
        await self._log_message(f"Queried shared memory: found {len(all_keys)} total entries")
        
        if all_keys:
            # Ask orchestrator to select relevant keys
            
            # # Build key list with previews
            # keys_with_previews = []
            # for key in recent_keys:
            #     preview = shared_memory_data.get(key, '')[:200]
            #     # Score keys by content quality (prioritize answers, findings, completed work)
            #     content = shared_memory_data.get(key, '').lower()
            #     score = 0
            #     if any(word in content for word in ['found', 'answer', 'result', 'extracted', 'identified', 'discovered', 'clicked', 'accessed', 'retrieved']):
            #         score += 10
            #     if any(word in content for word in ['scrolled', 'searched', 'looking', 'trying']):
            #         score -= 5  # Lower priority for navigation actions
            #     if 'final answer' in content or 'completed' in content or 'done' in content:
            #         score += 20  # Highest priority for completed work
                
            #     # Sort by score (highest first)
            #     keys_with_previews.append((score, key, preview))
            
            # # Sort by score (highest first) and show top entries
            # keys_with_previews.sort(reverse=True, key=lambda x: x[0])
            # top_keys = keys_with_previews[:10]  # Show top 10 by score
            
            # keys_formatted = "\n".join([
            #     f"  - {key} (score: {score})\n    Preview: {preview}..."
            #     for score, key, preview in top_keys
            # ])

            keys_formatted = "\n".join([f"  - {key}" for key in all_keys])
            
            key_selection_prompt = (
                f"SHARED MEMORY from parallel teams: Found {len(all_keys)} total entries.\n\n"
                f"Keys:\n"
                + keys_formatted + "\n\n"
                "Your task: Select ONE key that contains ANSWERS, FINDINGS, COMPLETED WORK, or KEY INFORMATION.\n"
                "CRITICAL: Your response MUST be EXACTLY in this format and ONLY this format:\n"
                "SELECTED_KEYS = [\"key_name_here\"]\n\n"
                "Example: If you see a key like 'team0-step5 - Extracted table data', you would respond:\n"
                "SELECTED_KEYS = [\"team0-step5 - Extracted table data\"]\n\n"
                "If none are relevant, respond with:\n"
                "SELECTED_KEYS = []\n\n"
                "DO NOT provide explanations, analysis, or answers. DO NOT describe what you found.\n"
                "DO NOT output the actual content from the keys.\n"
                "ONLY output the SELECTED_KEYS line. Nothing else."
            )
            
            selection_context = context.copy()
            selection_context.append(UserMessage(content=key_selection_prompt, source="SharedMemory"))
            
            # Make LLM call to get key selection
            try:
                await self._log_message(f"Requesting orchestrator to select relevant shared memory keys")
                llm_start = time.perf_counter()
                selection_response = await self._model_client.create(
                    self._get_compatible_context(selection_context),
                    cancellation_token=cancellation_token
                )
                llm_selection_seconds = time.perf_counter() - llm_start
                assert isinstance(selection_response.content, str)
                
                response_text = selection_response.content
                
                # Parse selected keys from response
                selected_keys = self._parse_selected_keys(response_text)
                await self._log_message(f"Orchestrator selected keys: {selected_keys}")
                
                # Limit to 1 key for now
                if len(selected_keys) > 1:
                    selected_keys = selected_keys[:1]
                    await self._log_message(f"Limited to 1 key: {selected_keys[0]}")
                
                # Fetch values for selected keys
                if selected_keys:
                    inject_start = time.perf_counter()
                    selected_values = self._get_shared_memory_values(selected_keys, shared_memory_data)
                    selected_values_count = len(selected_values)
                    if selected_values:
                        # Log structured memory_use events for downstream training.
                        self._log_memory_use_events(selected_values)

                        # Add selected values to context
                        # Use full shared-memory values for the model context;
                        # logging will apply its own truncation when printing.
                        values_text = "\n".join(
                            [f"[{key}]: {value}" for key, value in selected_values.items()]
                        )
                        shared_memory_content = (
                            f"=== SHARED MEMORY FROM OTHER TEAMS ===\n"
                            f"IMPORTANT: The following information comes from other parallel teams working on the same task.\n"
                            f"REVIEW THIS CAREFULLY before making decisions.\n\n"
                            f"{values_text}\n\n"
                            f"=== INSTRUCTIONS FOR USING SHARED MEMORY ===\n"
                            f"1. If another team already found the answer or key information, USE IT DIRECTLY instead of duplicating their work.\n"
                            f"2. If another team found partial results, BUILD ON THEIR WORK rather than starting from scratch.\n"
                            f"3. If another team tried an approach that failed, AVOID repeating the same mistake.\n"
                            f"4. If another team found useful data/methods, ADAPT AND APPLY them to your current step.\n"
                            f"5. Only proceed with new work if the shared memory doesn't contain relevant information.\n\n"
                            f"When deciding what to do next, EXPLICITLY REFERENCE the shared memory above if it's relevant to your decision."
                        )
                        context.append(
                            UserMessage(
                                content=shared_memory_content,
                                source="SharedMemory"
                            )
                        )
                        await self._log_message(f"Included {len(selected_values)} selected shared memory entries in context")
                    else:
                        await self._log_message(f"No values found for selected keys: {selected_keys}")
                    inject_seconds = time.perf_counter() - inject_start
                else:
                    # print(f"[DEBUG] WARNING: No keys selected by orchestrator. Response was: {response_text[:200]}", flush=False)
                    await self._log_message("No keys selected by orchestrator")
            except Exception as e:
                trace_logger.exception("Error during shared memory key selection: %s", e)
                await self._log_message(f"Error during shared memory key selection: {e}")
                # Continue without shared memory if there's an error
            finally:
                consult_total_seconds = time.perf_counter() - consult_start
                self._append_memory_event(
                    {
                        "event": "memory_consult",
                        "timestamp": datetime.utcnow().isoformat(),
                        "team_idx": os.environ.get("TEAM_IDX"),
                        "step_num": self._n_rounds,
                        "available_keys_count": len(all_keys),
                        "selected_keys_count": len(selected_keys),
                        "selected_values_count": selected_values_count,
                        "memory_read_seconds": memory_read_seconds,
                        "llm_selection_seconds": llm_selection_seconds,
                        "inject_seconds": inject_seconds,
                        "total_seconds": consult_total_seconds,
                    }
                )
        else:
            consult_total_seconds = time.perf_counter() - consult_start
            self._append_memory_event(
                {
                    "event": "memory_consult",
                    "timestamp": datetime.utcnow().isoformat(),
                    "team_idx": os.environ.get("TEAM_IDX"),
                    "step_num": self._n_rounds,
                    "available_keys_count": 0,
                    "selected_keys_count": 0,
                    "selected_values_count": 0,
                    "memory_read_seconds": memory_read_seconds,
                    "llm_selection_seconds": 0.0,
                    "inject_seconds": 0.0,
                    "total_seconds": consult_total_seconds,
                }
            )

        progress_ledger_prompt = self._get_progress_ledger_prompt(
            self._task, self._team_description, self._participant_names
        )
        context.append(UserMessage(content=progress_ledger_prompt, source=self._name))
        progress_ledger: Dict[str, Any] = {}
        assert self._max_json_retries > 0
        key_error: bool = False
        for _ in range(self._max_json_retries):
            if self._model_client.model_info.get("structured_output", False):
                response = await self._model_client.create(
                    self._get_compatible_context(context), json_output=LedgerEntry
                )
            elif self._model_client.model_info.get("json_output", False):
                response = await self._model_client.create(
                    self._get_compatible_context(context), cancellation_token=cancellation_token, json_output=True
                )
            else:
                response = await self._model_client.create(
                    self._get_compatible_context(context), cancellation_token=cancellation_token
                )
            ledger_str = response.content
            try:
                assert isinstance(ledger_str, str)
                output_json = extract_json_from_str(ledger_str)
                if len(output_json) != 1:
                    raise ValueError(
                        f"Progress ledger should contain a single JSON object, but found: {len(progress_ledger)}"
                    )
                progress_ledger = output_json[0]

                # If the team consists of a single agent, deterministically set the next speaker
                if len(self._participant_names) == 1:
                    progress_ledger["next_speaker"] = {
                        "reason": "The team consists of only one agent.",
                        "answer": self._participant_names[0],
                    }

                # Validate the structure
                required_keys = [
                    "is_request_satisfied",
                    "is_progress_being_made",
                    "is_in_loop",
                    "instruction_or_question",
                    "next_speaker",
                ]

                key_error = False
                for key in required_keys:
                    if (
                        key not in progress_ledger
                        or not isinstance(progress_ledger[key], dict)
                        or "answer" not in progress_ledger[key]
                        or "reason" not in progress_ledger[key]
                    ):
                        key_error = True
                        break

                # Validate the next speaker if the task is not yet complete
                if (
                    not progress_ledger["is_request_satisfied"]["answer"]
                    and progress_ledger["next_speaker"]["answer"] not in self._participant_names
                ):
                    key_error = True
                    break

                if not key_error:
                    break
                await self._log_message(f"Failed to parse ledger information, retrying: {ledger_str}")
            except (json.JSONDecodeError, TypeError):
                key_error = True
                await self._log_message("Invalid ledger format encountered, retrying...")
                continue
        if key_error:
            raise ValueError("Failed to parse ledger information after multiple retries.")

        await self._log_message(f"Progress Ledger: {progress_ledger}")

        # Check for task completion
        if progress_ledger["is_request_satisfied"]["answer"]:
            await self._log_message("Task completed, preparing final answer...")
            await self._prepare_final_answer(progress_ledger["is_request_satisfied"]["reason"], cancellation_token)
            return

        # Check for stalling
        if not progress_ledger["is_progress_being_made"]["answer"]:
            self._n_stalls += 1
        elif progress_ledger["is_in_loop"]["answer"]:
            self._n_stalls += 1
        else:
            self._n_stalls = max(0, self._n_stalls - 1)

        # Too much stalling
        if self._n_stalls >= self._max_stalls:
            await self._log_message("Stall count exceeded, re-planning with the outer loop...")
            await self._update_task_ledger(cancellation_token)
            await self._reenter_outer_loop(cancellation_token)
            return

        # Broadcast the next step
        message = TextMessage(content=progress_ledger["instruction_or_question"]["answer"], source=self._name)
        await self.update_message_thread([message])  # My copy

        await self._log_message(f"Next Speaker: {progress_ledger['next_speaker']['answer']}")
        # Log it to the output topic.
        await self.publish_message(
            GroupChatMessage(message=message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )
        # Log it to the output queue.
        await self._output_message_queue.put(message)

        # Broadcast it
        await self.publish_message(  # Broadcast
            GroupChatAgentResponse(response=Response(chat_message=message), name=self._name),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

        # Request that the step be completed
        next_speaker = progress_ledger["next_speaker"]["answer"]
        # Check if the next speaker is valid
        if next_speaker not in self._participant_name_to_topic_type:
            raise ValueError(
                f"Invalid next speaker: {next_speaker} from the ledger, participants are: {self._participant_names}"
            )
        participant_topic_type = self._participant_name_to_topic_type[next_speaker]
        await self.publish_message(
            GroupChatRequestPublish(),
            topic_id=DefaultTopicId(type=participant_topic_type),
            cancellation_token=cancellation_token,
        )

        # Send the message to the next speaker
        if self._emit_team_events:
            select_msg = SelectSpeakerEvent(content=[next_speaker], source=self._name)
            await self.publish_message(
                GroupChatMessage(message=select_msg),
                topic_id=DefaultTopicId(type=self._output_topic_type),
            )
            await self._output_message_queue.put(select_msg)

    async def _update_task_ledger(self, cancellation_token: CancellationToken) -> None:
        """Update the task ledger (outer loop) with the latest facts and plan."""
        context = self._thread_to_context()

        # Update the facts
        update_facts_prompt = self._get_task_ledger_facts_update_prompt(self._task, self._facts)
        context.append(UserMessage(content=update_facts_prompt, source=self._name))

        response = await self._model_client.create(
            self._get_compatible_context(context), cancellation_token=cancellation_token
        )

        assert isinstance(response.content, str)
        self._facts = response.content
        context.append(AssistantMessage(content=self._facts, source=self._name))

        # Update the plan
        update_plan_prompt = self._get_task_ledger_plan_update_prompt(self._team_description)
        context.append(UserMessage(content=update_plan_prompt, source=self._name))

        response = await self._model_client.create(
            self._get_compatible_context(context), cancellation_token=cancellation_token
        )

        assert isinstance(response.content, str)
        self._plan = response.content

    async def _prepare_final_answer(self, reason: str, cancellation_token: CancellationToken) -> None:
        """Prepare the final answer for the task."""
        context = self._thread_to_context()

        # Get the final answer
        final_answer_prompt = self._get_final_answer_prompt(self._task)
        context.append(UserMessage(content=final_answer_prompt, source=self._name))

        response = await self._model_client.create(
            self._get_compatible_context(context), cancellation_token=cancellation_token
        )
        assert isinstance(response.content, str)
        message = TextMessage(content=response.content, source=self._name)

        await self.update_message_thread([message])  # My copy

        # Log it to the output topic.
        await self.publish_message(
            GroupChatMessage(message=message),
            topic_id=DefaultTopicId(type=self._output_topic_type),
        )
        # Log it to the output queue.
        await self._output_message_queue.put(message)

        # Broadcast
        await self.publish_message(
            GroupChatAgentResponse(response=Response(chat_message=message), name=self._name),
            topic_id=DefaultTopicId(type=self._group_topic_type),
            cancellation_token=cancellation_token,
        )

        if self._termination_condition is not None:
            await self._termination_condition.reset()
        # Signal termination
        await self._signal_termination(StopMessage(content=reason, source=self._name))

    def _thread_to_context(self) -> List[LLMMessage]:
        """Convert the message thread to a context for the model."""
        context: List[LLMMessage] = []
        for m in self._message_thread:
            if isinstance(m, ToolCallRequestEvent | ToolCallExecutionEvent):
                # Ignore tool call messages.
                continue
            elif isinstance(m, StopMessage | HandoffMessage):
                context.append(UserMessage(content=m.content, source=m.source))
            elif m.source == self._name:
                assert isinstance(m, TextMessage | ToolCallSummaryMessage)
                context.append(AssistantMessage(content=m.content, source=m.source))
            else:
                assert isinstance(m, (TextMessage, MultiModalMessage, ToolCallSummaryMessage))
                context.append(UserMessage(content=m.content, source=m.source))
        return context

    def _get_compatible_context(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        """Ensure that the messages are compatible with the underlying client, by removing images if needed."""
        if self._model_client.model_info["vision"]:
            return messages
        else:
            return remove_images(messages)
