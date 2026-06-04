from __future__ import annotations

from compile import Trie


def search(text: str, trie: Trie) -> dict[str, list[int]]:
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
