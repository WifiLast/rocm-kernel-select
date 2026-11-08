#!/bin/bash
# Keep fastvideo_kernel_amd in step with upstream FastVideo/fastvideo-kernel
# (FASTVIDEO_KERNEL_UPSTREAM overrides the checkout path).
#
#   ./sync_upstream.sh patch   write patches/amd.patch = upstream_snapshot/python -> python
#                              (run after editing python/)
#   ./sync_upstream.sh sync    after a FastVideo git pull: write the patch, copy the new
#                              upstream python/ + tests/, re-apply the patch, record the commit
# Rejected hunks are left as *.rej for manual merging.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
UP="${FASTVIDEO_KERNEL_UPSTREAM:-/mnt/data/image_to_video/gen/FastVideo/fastvideo-kernel}"
cd "$HERE"

make_patch() {
    mkdir -p patches
    diff -ruN -x __pycache__ -x "*.egg-info" upstream_snapshot/python python > patches/amd.patch || true
    echo "patches/amd.patch: $(grep -c '^+++ ' patches/amd.patch) files"
}

case "${1:-}" in
    patch)
        make_patch
        ;;
    sync)
        make_patch
        rm -rf upstream_snapshot/python python.new
        cp -r "$UP/python" upstream_snapshot/python
        cp -r "$UP/python" python.new
        find upstream_snapshot python.new -name __pycache__ -prune -exec rm -rf {} +
        (cd python.new && patch -p1 --forward < ../patches/amd.patch) || echo "WARNING: some hunks failed, see python.new/**/*.rej"
        cp -r python/*.egg-info python.new/ 2>/dev/null || true  # editable install metadata
        rm -rf python && mv python.new python
        rsync -a --delete --exclude __pycache__ "$UP/tests/" tests/
        git -C "$UP/.." rev-parse --short HEAD > upstream_snapshot/COMMIT
        sed -i "s/^__version__ = .*/__version__ = \"$(sed -n 's/^__version__ = \"\(.*\)\"/\1/p' "$UP/python/fastvideo_kernel/version.py")+amd\"/" python/fastvideo_kernel/version.py
        echo "synced to upstream $(cat upstream_snapshot/COMMIT); update the version in pyproject.toml if it changed"
        ;;
    *)
        echo "usage: $0 patch|sync" >&2
        exit 2
        ;;
esac
