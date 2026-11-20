"""Reusable contract suites. Import them into a test module and subclass; do not copy them.

from agent_container.testing import ModelProviderContract

class TestMyProvider(ModelProviderContract):
    def make_provider(self, replies):
        return MyProvider(...)  # one that answers with exactly these replies, in order
"""

from agent_container.testing.contracts import ModelProviderContract, ToolContract

__all__ = ["ModelProviderContract", "ToolContract"]
