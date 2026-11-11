#!/usr/bin/env python3
"""Verify every imported file against the prepared release source manifest."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def verify():
    manifest=json.loads((ROOT/'docs/source_manifest.json').read_text())
    failures=[];unchanged=0
    for row in manifest['files']:
        p=ROOT/row['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['release_sha256']:
            failures.append(row['path'])
        if row['release_sha256']==row['september_sha256']:unchanged+=1
    return {'checked':len(manifest['files']),'unchanged_from_september':unchanged,
            'documented_portability_edits':sum(r.get('change_category')=='deployment_paths' for r in manifest['files']),
            'documented_cleanup_edits':sum(r.get('change_category')=='release_cleanup' for r in manifest['files']),
            'runtime_updated_files':sum(bool(r.get('runtime_update')) for r in manifest['files']),
            'failures':failures}

if __name__=='__main__':
    result=verify();print(json.dumps(result,indent=2));raise SystemExit(bool(result['failures']))
