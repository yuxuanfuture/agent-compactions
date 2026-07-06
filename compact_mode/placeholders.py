from __future__ import annotations


class PlaceholderMode:
    def __init__(self, name: str) -> None:
        self.name = name

    def build_chat_request(self, *args, **kwargs):
        raise NotImplementedError(
            f"compact mode {self.name!r} is reserved but not implemented yet"
        )
