import hashlib, os, sys
from collections import defaultdict

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
            
hash_count = 0

def hashfile(filename, size, maxsz=0):
    global hash_count
    
    hasher = hashlib.sha256()
    if size == 0: return (filename, 0, None)
    hash_count += 1
    with open(filename, "rb") as f:
        if maxsz > 0:
            chunk = f.read(maxsz)
            hasher.update(chunk)
            return (filename, size, hasher.hexdigest())
        while chunk := f.read(1024):
            hasher.update(chunk)
    return (filename, size, hasher.hexdigest())

PREFIXLEN = 64

def fastfilter(files):
    hashmap = defaultdict(list)
    for f, s in files:
        f, s, h = hashfile(f, s, PREFIXLEN)
        hashmap[h].append((f, s, h))
        if len(hashmap[h]) >= 2:
            if len(hashmap[h]) == 2:
                yield hashmap[h][0]
            yield f,s,h

def slowfilter(files):
    hashmap = defaultdict(list)
    for f, s, h in files:
        if s <= PREFIXLEN:
            hashmap[h].append(f)
        else:
            f, _, h = hashfile(f, s, 0)
            hashmap[h].append(f)
    for _, files in hashmap.items():
        if len(files) == 1:
            continue
        yield files

f1 = files(sys.argv[1] if len(sys.argv) > 1 else "data")
f2 = sizefilter(f1)
f3 = fastfilter(f2)
f4 = slowfilter(f3)

for f in f4: print(f)
print(hash_count)
