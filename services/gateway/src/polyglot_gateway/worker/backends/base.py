from typing import Protocol, runtime_checkable


@runtime_checkable
class BackendClient(Protocol):
    def translate(self, text: str, model_name: str) -> str: ...
