from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Node:
    """One Aho-Corasick automaton state.

    `children` are goto edges to the next state. `fail` is the link followed on
    mismatch -- it points to the longest proper suffix of the text consumed so
    far that is still a prefix in the trie, so no input has to be re-scanned.
    `word` is set iff this state is the terminal of a vocabulary word.
    `outputs` lists every vocab word matching at this state (this state's own
    word plus everything reachable through fail-link ancestors), precomputed
    during construction so the scanner does not walk the fail chain per char.
    """

    children: dict[str, "Node"] = field(default_factory=dict)
    fail: "Node | None" = None
    word: str | None = None
    outputs: list[str] = field(default_factory=list)


class Trie:
    """Aho-Corasick automaton over a fixed vocabulary.

    Construction is two-phase: insert each word into the trie, then BFS to
    fill in fail links and merged `outputs` lists. After construction the
    automaton scans any text in a single linear pass, reporting every match
    of every vocab word, including overlaps -- which is why we use AC instead
    of running KMP once per word over the corpus.

    Example for vocabulary {he, she, hers} (* marks a word terminal):

        root
         |
         +-- h
         |    |
         |    +-- e *
         |         |
         |         +-- r
         |              |
         |              +-- s *
         |
         +-- s
              |
              +-- h
                   |
                   +-- e *

    Failure links (each points to the longest proper suffix of the path
    from root that is itself a trie prefix):

        h, s, he*, her  -> root
        sh              -> h
        she*            -> he*    (lands on a word terminal: reaching state
                                   [she] also reports a match for "he")
        hers*           -> s

    The merged `outputs` at state [she] is therefore ["she", "he"], so the
    scanner emits both matches by reading one list, with no fail-chain walk
    per character.
    """

    def __init__(self, words: Iterable[str]) -> None:
        """Build the trie from `words` then post-process to add fail/output links.

        Empty strings and duplicates are dropped; `self.words` preserves the
        de-duplicated input order so callers can iterate the vocabulary
        deterministically (useful for stable per-word output in the CLI).
        """
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
        """Create the path of nodes spelling `word` and mark the terminal.

        `setdefault` reuses any shared prefix with words already inserted, so
        the trie stays compact for vocabularies with common prefixes.
        """
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, Node())
        node.word = word

    def _build_failures(self) -> None:
        """BFS by depth, computing each node's fail link from its parent's.

        For a child reached from `r` via `ch`, walk `r.fail`'s chain until a
        state with an outgoing edge on `ch` is found (or root). BFS order
        guarantees `r.fail` is already final by the time we process its
        children, which is what makes this work in one pass.

        We also set `outputs = fail.outputs + [own word]` here so the scanner
        can enumerate every overlapping match by reading a single list per
        state, instead of walking the fail chain per input character.
        """
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
