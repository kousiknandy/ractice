import hashlib, os, sys
from collections import defaultdict
from itertools import islice
from concurrent.futures import ThreadPoolExecutor
from functools import partial

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

def hashfile(fileinfo, maxsz=0):
    filename, size = fileinfo
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

fasthash = partial(hashfile, maxsz=PREFIXLEN)
slowhash = partial(hashfile, maxsz=0)

def collisions(results, individual):
    for k, v in results.items():
        if len(v) == 1: continue
        if individual:
            yield from v
        else:
            yield v

suspects = partial(collisions, individual=True)
duplicates = partial(collisions, individual=False)

def djb_hash(s, m):
    h = 5381
    for c in s:
        h = 33 * h + ord(c)
    return h % m

class MapReduce:
    def __init__(self, mapper, reducer, partitions):
        self.mapper = mapper
        self.reducer = reducer
        self.partitions = partitions

    def partition(self, results):
        parts = [defaultdict(list) for _ in range(self.partitions)]
        for f, s, h in results:
            parts[djb_hash(h, self.partitions)][h].append((f,s))
        return parts

    def __call__(self, files):
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = executor.map(self.mapper, files)
        parts = self.partition(results)
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = executor.map(self.reducer, parts)
        for r in results:
            yield from r
        
f1 = files(sys.argv[1] if len(sys.argv) > 1 else "data")
f2 = sizefilter(f1)
m1 = MapReduce(fasthash, suspects, 2)
m2 = m1(f2)
m3 = MapReduce(slowhash, duplicates, 2)
m4 = m3(m2)

for m in m4: print(m)
