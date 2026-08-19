import hashlib, os, sys
from collections import defaultdict
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

def files(topdir):
    for dirname, _, filenames in os.walk(topdir):
        for file in filenames:
            f = os.path.join(dirname, file)
            s = os.path.getsize(f)
            if s:
                yield f, s

def sizefilter(files):
    szmap = defaultdict(list)
    for f, s in files:
        szmap[s].append((f,s))
        if len(szmap[s]) >= 2:
            if len(szmap[s]) == 2:
                yield szmap[s][0]
            yield f,s

PREFIXLEN = 64

def hashfile(filename, size, maxsz=0):
    hasher = hashlib.sha256()
    if size == 0: return (filename, 0, None)
    with open(filename, "rb") as f:
        if maxsz > 0:
            chunk = f.read(maxsz)
            hasher.update(chunk)
            return (filename, size, hasher.hexdigest())
        while chunk := f.read(1024):
            hasher.update(chunk)
    return (filename, size, hasher.hexdigest())

def fastcollisions(files):
    hashmap = defaultdict(list)
    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = {executor.submit(hashfile, f, s, PREFIXLEN)
                   for f,s in islice(files, 8)}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                f, s, h = f.result()
                hashmap[h].append((f,s))
                if len(hashmap[h]) >= 2:
                    if len(hashmap[h]) == 2:
                        yield hashmap[h][0]
                    yield f,s,h
                try:
                    f, s = next(files)
                    pending.add(executor.submit(hashfile, f, s, PREFIXLEN))
                except StopIteration:
                    pass
                
def slowcollisions(files):
    pass

f1 = files(sys.argv[1] if len(sys.argv) > 1 else "data")
f2 = sizefilter(f1)
f3 = fastcollisions(f2)

print(list(f3))
