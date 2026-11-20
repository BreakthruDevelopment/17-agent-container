"""The model provider contract (Strategy)."""

from abc import ABC, abstractmethod

from agent_container.core.messages import ModelRequest, ModelResponse


class ModelProvider(ABC):
    """Talks to a model. `BedrockProvider` (17.3) and `ReplayProvider` are the implementations.

    Every model call in the track goes through here. Implementations must:

    * be safe to await concurrently (app 20 fans out subagents);
    * return `ModelResponse.stop_reason` exactly as the model said it, never mapped or defaulted;
    * leave the request untouched.

    `agent_container.testing.ModelProviderContract` checks all of that; every implementation runs
    it.
    """

    @abstractmethod
    async def converse(self, request: ModelRequest) -> ModelResponse:
        """Send one request and return the model's reply."""
