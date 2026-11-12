"""Query-independent identity metadata for exact duplicate source vectors.

This does not search vectors. It expands only bitwise-identical raw-vector
classes reached by ANN, retaining strict original CTID tie ordering. Near
duplicates remain distinct. No query text, query vector or answer is stored.
"""
import hashlib
import heapq
import json
import locale
from pathlib import Path
import sqlite3

import numpy as np

VERSION = 'bird-raw-duplicate-identities-v1'


def build(directory, records, *, count, max_k, binding, progress=lambda n: None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    temporary = directory/'working.sqlite'
    conn = sqlite3.connect(temporary)
    try:
        conn.execute('PRAGMA cache_size=-65536')
        conn.execute('CREATE TABLE points(pid INTEGER PRIMARY KEY,h BLOB,raw BLOB,ctid TEXT)')
        batch=[]; seen=0
        for pid, ctid, raw in records:
            raw=bytes(raw)
            if pid != seen:
                raise ValueError('Duplicate metadata requires complete ordered point IDs')
            batch.append((pid,hashlib.sha256(raw).digest(),raw,ctid)); seen+=1
            if len(batch)>=2048:
                conn.executemany('INSERT INTO points VALUES (?,?,?,?)',batch)
                conn.commit();batch.clear();progress(seen)
        conn.executemany('INSERT INTO points VALUES (?,?,?,?)',batch);conn.commit()
        if seen!=count:
            raise ValueError('Source/index point count mismatch')
        conn.execute('CREATE INDEX hashes ON points(h)');conn.commit()
        groups=np.full(count,-1,dtype=np.int64); members={}; duplicate_points=0
        hashes=conn.execute('SELECT h FROM points GROUP BY h HAVING count(*)>1')
        for (digest,) in hashes:
            # Hashes accelerate grouping; full byte equality is authoritative,
            # including in the event of a digest collision.
            exact={}
            for pid,raw,ctid in conn.execute('SELECT pid,raw,ctid FROM points WHERE h=?',(digest,)):
                entry=exact.setdefault(raw,dict(ids=[],best=[]))
                entry['ids'].append(pid)
                entry['best'].append((locale.strxfrm(ctid),ctid,pid))
                if len(entry['best'])>2*max_k:
                    entry['best']=heapq.nsmallest(max_k,entry['best'])
            for item in exact.values():
                if len(item['ids'])<2:continue
                representative=item['ids'][0]
                groups[item['ids']]=representative
                members[str(representative)]=[p for _,_,p in heapq.nsmallest(max_k,item['best'])]
                duplicate_points+=len(item['ids'])
        np.save(directory/'groups.npy',groups,allow_pickle=False)
        (directory/'members.json').write_text(json.dumps(members,separators=(',',':')))
        files={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
               for name in ('groups.npy','members.json')}
        manifest=dict(version=VERSION,binding=binding,count=count,max_k=max_k,
            duplicate_groups=len(members),duplicate_points=duplicate_points,files=files,
            raw_byte_equality_verified=True,query_dependent_data=False)
        (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        progress(count)
        return manifest
    finally:
        conn.close();temporary.unlink(missing_ok=True)


class DuplicateIdentities:
    def __init__(self,directory,*,binding,count):
        directory=Path(directory)
        self.manifest=json.loads((directory/'manifest.json').read_text())
        if (self.manifest['version']!=VERSION or self.manifest['binding']!=binding
                or self.manifest['count']!=count):
            raise ValueError('Duplicate metadata belongs to different immutable assets')
        for name,digest in self.manifest['files'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Duplicate identity metadata changed')
        self.groups=np.load(directory/'groups.npy',mmap_mode='r',allow_pickle=False)
        if self.groups.shape!=(count,):raise ValueError('Invalid duplicate membership map')
        self.members=json.loads((directory/'members.json').read_text())

    def expand(self,positions,*,k):
        if not 1<=k<=self.manifest['max_k']:
            return list(positions),False
        result=list(dict.fromkeys(int(p) for p in positions))
        if any(p<0 or p>=len(self.groups) for p in result):raise ValueError('Unknown ANN point ID')
        for group in sorted({int(self.groups[p]) for p in result}-{ -1 }):
            result.extend(self.members[str(group)][:k])
        expanded=list(dict.fromkeys(result))
        return expanded,len(expanded)>len(set(positions))
