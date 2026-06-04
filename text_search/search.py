from __future__ import annotations

from compile import Trie


def search(text: str, trie: Trie) -> dict[str, list[int]]:
    """Scan `text` against `trie` in one linear pass; return per-word offsets.

    For each character we either follow a goto edge or, on mismatch, follow
    fail links until we land on a state that has the needed edge (or root).
    At each visited state we emit every word in `node.outputs` at start
    offset `i - len(word) + 1`, which yields all overlapping matches without
    any extra fail-chain walk -- that work was done at construction time.

    The result dict is keyed by every word in `trie.words` (empty list when
    a word was searched but never matched), so callers can distinguish
    "searched but absent" from "not in the vocabulary".
    """
    matches: dict[str, list[int]] = {w: [] for w in trie.words}
    root = trie.root
    node = root
    for i, ch in enumerate(text):
        while node is not root and ch not in node.children:
            assert node.fail is not None
            node = node.fail
        node = node.children.get(ch, root)
        for w in node.outputs:
            matches[w].append(i - len(w) + 1)
    return matches
