import hashlib
import os
from collections import defaultdict

def filenames(topdir):
    for dirname, _, filenames in os.walk(topdir):
        for file in filenames:
            yield os.path.join(dirname, file)

def hashfile(filename, maxsz=0):
    hasher = hashlib.sha256()
    size = os.path.getsize(filename)
    if size == 0: return (filename, 0, None)
    with open(filename, "rb") as f:
        if maxsz > 0:
            chunk = f.read(maxsz)
            hasher.update(chunk)
            return (filename, size, hasher.hexdigest())
        while chunk := f.read(1024):
            hasher.update(chunk)
    return (filename, size, hasher.hexdigest())

FASTSZ = 64
hashdict = defaultdict(list)
for F in filenames("data"):
    f, s, h = hashfile(F,FASTSZ)
    hashdict[h].append((f,s))
for k,v in hashdict.items(): print(k, v)
print("*"*30)
finaldict = defaultdict(list)
for h, v in hashdict.items():
    if len(v) == 1: continue
    for f, s in v:
        if s <= FASTSZ:
            finaldict[h].append((f, s)) 
        else:
            f, s, nh = hashfile(f)
            finaldict[nh].append((f, s))
for k,v in finaldict.items(): print(k, v)
print("*"*30)
for k,v in finaldict.items():
    if len(v) == 1: continue
    print(tuple(f for f, _ in v))
        
