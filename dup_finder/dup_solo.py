import hashlib
import os
from collections import defaultdict
import sys

def filenames(topdir):
    for dirname, _, filenames in os.walk(topdir):
        for file in filenames:
            yield os.path.join(dirname, file)

hash_count = 0

def hashfile(filename, maxsz=0):
    global hash_count
    
    hasher = hashlib.sha256()
    size = os.path.getsize(filename)
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

FASTSZ = 64
    
def fast_collisions(filenames):
    hashdict = defaultdict(list)
    for F in filenames:
        f, s, h = hashfile(F,FASTSZ)
        hashdict[h].append((f,s))
    for k,v in hashdict.items():
        if len(v) > 1:
            yield k, v

def final_collisions(collisions):    
    finaldict = defaultdict(list)
    for hash, files in collisions:
        # print("suspect", files)
        for f, s in files:
            if s <= FASTSZ:
                finaldict[hash].append((f, s)) 
            else:
                f, s, nh = hashfile(f)
                finaldict[nh].append((f, s))
    for k,v in finaldict.items():
        if len(v) == 1: continue
        yield tuple(f for f, _ in v)
        
f = filenames(sys.argv[1] if len(sys.argv) > 1 else "data")
c1 = fast_collisions(f)
c2 = final_collisions(c1)

for c in c2: print(c)
print(hash_count)
