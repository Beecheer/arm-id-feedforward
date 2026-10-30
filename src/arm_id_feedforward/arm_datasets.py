#!/usr/bin/env python3
"""Shallow dataset catalog and content-hash resolution for immutable provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CATALOG = Path.cwd()/'datasets.json'


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def load_catalog(path=CATALOG):
    doc=json.loads(path.read_text())
    if doc.get('kind')!='arm_dataset_catalog' or doc.get('version')!=1:
        raise ValueError('不支持的数据索引格式')
    return doc


def resolve_provenance(recorded_path, expected_sha256, catalog=CATALOG):
    """Keep archived JSON bytes intact; locate moved files by verified content."""
    recorded=Path(recorded_path)
    candidates=[recorded, catalog.parent/recorded] if recorded_path else []
    if catalog.is_file():
        candidates += [catalog.parent/item['path'] for item in load_catalog(catalog)['files']
                       if item['sha256']==expected_sha256]
    for path in candidates:
        if path.is_file() and sha256(path)==expected_sha256:return path.resolve()
    raise FileNotFoundError(f'未找到校验值为 {expected_sha256} 的来源文件：{recorded_path}')


def flat_recording(group, stage):
    doc=json.loads((group/'dataset.json').read_text())
    if doc.get('kind')!='arm_recording_group' or doc.get('version')!=1:
        raise ValueError('不支持的数据组格式')
    entry=doc['records'][stage]
    capture=group/entry['capture'];plan=group/entry['plan']
    quality=json.loads((group/entry['quality']).read_text())
    metadata=json.loads(capture.with_suffix('.json').read_text())
    if sha256(capture)!=entry['sha256'] or entry['sha256']!=quality['capture_sha256']:
        raise ValueError(f'{capture}: 原始数据与索引/质量报告不一致')
    if sha256(plan)!=metadata['trajectory_sha256']:
        raise ValueError(f'{plan}: 轨迹与采集元数据不一致')
    return capture,quality


def check(path=CATALOG):
    doc=load_catalog(path)
    for item in doc['files']:
        source=path.parent/item['path']
        if not source.is_file() or sha256(source)!=item['sha256']:
            raise ValueError(f'文件缺失或内容变化：{source}')
    for group in doc['groups']:
        for stage in group['records']:flat_recording(path.parent/group['path'],stage)
    return doc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check'])
    parser.add_argument('--catalog',type=Path,default=CATALOG)
    args=parser.parse_args();doc=check(args.catalog)
    print(f'数据校验通过：{len(doc["groups"])} 组，{len(doc["files"])} 个索引文件。')


if __name__=='__main__':main()
