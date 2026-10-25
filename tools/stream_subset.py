#!/usr/bin/env python3
"""
Extract a subset of the LingBot-Depth dataset (robbyant/mdm_depth) by streaming.

Each archive is a single large .tar.zst (50-320 GB), so instead of downloading it,
this script pipes `curl | zstd -dc` into a streaming tar reader and only writes the
members that pass the filters. Files are grouped by their camera directory
(the parent directory, with a trailing color/gtdepth/rawdepth component removed),
and all modalities of a group are kept or dropped together.

Usage:
    # list groups only
    python tools/stream_subset.py --file RobbySimVal_batch_0001.tar.zst --list --max-groups 20

    # keep every 4th camera group and every 10th frame, skipping scenes already in the eval subset
    python tools/stream_subset.py --file RobbySimVal_batch_0001.tar.zst --out data/mdm_depth \\
        --group-stride 4 --frame-stride 10 --exclude-root data/mdm_depth --exclude-depth 2
"""

import argparse
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HF_BASE = 'https://huggingface.co/datasets/robbyant/mdm_depth/resolve/main'
MODALITY_DIRS = {'color', 'gtdepth', 'rawdepth'}
FRAME_RE = re.compile(r'(\d+)')


def group_of(path: str) -> str:
    parent = Path(path).parent
    if parent.name in MODALITY_DIRS:
        parent = parent.parent
    return str(parent)


def frame_id(path: str):
    m = FRAME_RE.match(Path(path).name)
    return int(m.group(1)) if m else None


def open_stream(url: str):
    curl = subprocess.Popen(['curl', '-sfL', url], stdout=subprocess.PIPE)
    zstd = subprocess.Popen(['zstd', '-dc'], stdin=curl.stdout, stdout=subprocess.PIPE)
    curl.stdout.close()  # let curl receive SIGPIPE if zstd exits
    return curl, zstd


def main():
    parser = argparse.ArgumentParser(description='Stream-extract a subset of robbyant/mdm_depth')
    parser.add_argument('--file', type=str, help='Archive name in the HF dataset repo')
    parser.add_argument('--url', type=str, help='Full archive URL (overrides --file)')
    parser.add_argument('--out', type=str, default='data/mdm_depth', help='Output root directory')
    parser.add_argument('--include', type=str, default=None, help='Regex a member path must match')
    parser.add_argument('--group-stride', type=int, default=1, help='Keep every N-th camera group')
    parser.add_argument('--frame-stride', type=int, default=1, help='Keep frames whose id %% N == 0')
    parser.add_argument('--max-groups', type=int, default=None, help='Stop after N kept groups')
    parser.add_argument('--exclude-root', type=str, default=None,
                        help='Skip groups whose first --exclude-depth path components exist under this directory '
                             '(e.g. the extracted eval subset, to keep train/eval scenes disjoint)')
    parser.add_argument('--exclude-depth', type=int, default=2, help='Path depth compared with --exclude-root')
    parser.add_argument('--strip-prefix', type=str, default='lingbot-depth_pub/', help='Prefix removed from member paths')
    parser.add_argument('--list', action='store_true', help='Only list groups, write nothing')
    args = parser.parse_args()

    if args.url is None and args.file is None:
        parser.error('either --file or --url is required')
    url = args.url or f'{HF_BASE}/{args.file}'
    include = re.compile(args.include) if args.include else None
    out_root = Path(args.out)
    exclude_root = Path(args.exclude_root) if args.exclude_root else None

    curl, zstd = open_stream(url)
    tf = tarfile.open(fileobj=zstd.stdout, mode='r|')

    seen_groups = {}  # group -> kept?
    kept_groups = 0
    kept_files = 0
    kept_bytes = 0
    read_bytes = 0
    start = time.time()
    last_log = start

    try:
        for member in tf:
            if not member.isfile():
                continue
            read_bytes += member.size
            name = member.name
            if name.startswith(args.strip_prefix):
                name = name[len(args.strip_prefix):]

            group = group_of(name)
            if group not in seen_groups:
                if args.max_groups is not None and kept_groups >= args.max_groups:
                    break
                keep = (len(seen_groups) % args.group_stride == 0)
                if include is not None and not include.search(group):
                    keep = False
                if exclude_root is not None and (exclude_root / Path(*Path(group).parts[:args.exclude_depth])).exists():
                    keep = False
                seen_groups[group] = keep
                if keep:
                    kept_groups += 1
                    if args.list:
                        print(f'{read_bytes / 1e9:8.2f} GB  {group}', flush=True)
            now = time.time()
            if now - last_log > 30:
                last_log = now
                print(f'[{now - start:6.0f}s] read {read_bytes / 1e9:7.2f} GB | '
                      f'groups {kept_groups}/{len(seen_groups)} | '
                      f'kept {kept_files} files, {kept_bytes / 1e9:.2f} GB | at {group}', flush=True)
            if not seen_groups[group] or args.list:
                continue

            if include is not None and not include.search(name):
                continue
            fid = frame_id(name)
            if fid is not None and fid % args.frame_stride != 0:
                continue

            dst = out_root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, open(dst, 'wb') as f:
                f.write(src.read())
            kept_files += 1
            kept_bytes += member.size
    finally:
        zstd.stdout.close()
        for p in (zstd, curl):
            if p.poll() is None:
                p.terminate()
            p.wait()

    print(f'Done in {time.time() - start:.0f}s: read {read_bytes / 1e9:.2f} GB, '
          f'kept {kept_groups}/{len(seen_groups)} groups, {kept_files} files, {kept_bytes / 1e9:.2f} GB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
