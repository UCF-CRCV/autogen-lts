"""State management for agents, teams and termination conditions."""

from ._states import (
    AssistantAgentState,
    BaseGroupChatManagerState,
    BaseState,
    ChatAgentContainerState,
    MagenticOneOrchestratorState,
    MagenticMemoryOrchestratorState,
    RoundRobinManagerState,
    SelectorManagerState,
    SocietyOfMindAgentState,
    SwarmManagerState,
    TeamState,
)

__all__ = [
    "BaseState",
    "AssistantAgentState",
    "BaseGroupChatManagerState",
    "ChatAgentContainerState",
    "RoundRobinManagerState",
    "SelectorManagerState",
    "SwarmManagerState",
    "MagenticOneOrchestratorState",
    "MagenticMemoryOrchestratorState",
    "TeamState",
    "SocietyOfMindAgentState",
]
