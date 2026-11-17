"""Tool specification shared by environments and strategies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], str]

    def to_api(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
