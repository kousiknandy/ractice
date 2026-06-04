from __future__ import annotations

import sys

from compile import Trie
from search import search


USAGE = "usage: cli.py FILE WORD [WORD ...]"


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(USAGE, file=sys.stderr)
        return 2
    path = argv[1]
    words = argv[2:]
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    trie = Trie(words)
    results = search(text, trie)
    printed: set[str] = set()
    for w in words:
        if w in printed:
            continue
        printed.add(w)
        offsets = results.get(w, [])
        if offsets:
            print(f"{w}: {' '.join(str(o) for o in offsets)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
