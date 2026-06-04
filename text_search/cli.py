from __future__ import annotations

import sys

from compile import Trie
from search import search


USAGE = "usage: cli.py FILE WORD [WORD ...]"


def main(argv: list[str]) -> int:
    """fgrep-style entry point: read FILE, search for WORDs, print offsets.

    Output is one line per matched word in the form `word: off1 off2 ...`,
    in the order the words were given on the command line. Words with zero
    matches are silently skipped, mirroring fgrep -- which does not announce
    patterns that never fired. Returns 0 on success and 2 on usage error.
    """
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
