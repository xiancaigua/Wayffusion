from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


MODULE_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = MODULE_ROOT / "manifest.yaml"


@dataclass(frozen=True)
class AgentMemoryCard:
    card_id: str
    title: str
    file_path: Path
    trust: str
    tags: tuple[str, ...]
    summary: str

    def read_text(self) -> str:
        return self.file_path.read_text(encoding="utf-8")


def load_manifest() -> dict[str, Any]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def list_cards() -> list[AgentMemoryCard]:
    manifest = load_manifest()
    cards = []
    for item in manifest.get("cards", []):
        cards.append(
            AgentMemoryCard(
                card_id=str(item["id"]),
                title=str(item["title"]),
                file_path=MODULE_ROOT / str(item["file"]),
                trust=str(item["trust"]),
                tags=tuple(str(tag) for tag in item.get("tags", [])),
                summary=str(item.get("summary", "")),
            )
        )
    return cards


def get_card(card_id: str) -> AgentMemoryCard:
    for card in list_cards():
        if card.card_id == card_id:
            return card
    raise KeyError(f"Unknown agent memory card: {card_id}")


def search_cards(
    query: str | None = None,
    tags: list[str] | None = None,
    trust: str | None = None,
) -> list[AgentMemoryCard]:
    query_text = (query or "").strip().lower()
    tag_filter = set(tags or [])
    results = []
    for card in list_cards():
        haystack = " ".join([card.card_id, card.title, card.summary, *card.tags]).lower()
        if query_text and query_text not in haystack and query_text not in card.read_text().lower():
            continue
        if trust is not None and card.trust != trust:
            continue
        if tag_filter and not tag_filter.issubset(set(card.tags)):
            continue
        results.append(card)
    return results
