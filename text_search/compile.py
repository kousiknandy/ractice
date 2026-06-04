from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Node:
    children: dict[str, "Node"] = field(default_factory=dict)
    fail: "Node | None" = None
    word: str | None = None
    outputs: list[str] = field(default_factory=list)


class Trie:
    def __init__(self, words: Iterable[str]) -> None:
        self.root: Node = Node()
        self.words: list[str] = []
        seen: set[str] = set()
        for w in words:
            if not w or w in seen:
                continue
            seen.add(w)
            self.words.append(w)
            self._insert(w)
        self._build_failures()

    def _insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, Node())
        node.word = word

    def _build_failures(self) -> None:
        root = self.root
        root.fail = root
        queue: deque[Node] = deque()
        for child in root.children.values():
            child.fail = root
            if child.word is not None:
                child.outputs.append(child.word)
            queue.append(child)
        while queue:
            r = queue.popleft()
            for ch, u in r.children.items():
                state = r.fail
                assert state is not None
                while state is not root and ch not in state.children:
                    state = state.fail
                    assert state is not None
                cand = state.children.get(ch)
                u.fail = cand if cand is not None and cand is not u else root
                u.outputs = list(u.fail.outputs)
                if u.word is not None:
                    u.outputs.append(u.word)
                queue.append(u)
