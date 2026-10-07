#!/bin/sh
# Build 2430010316.zip in the layout required by the Deep Learning project brief (src/ instead of code/).
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
D="$TMP/2430010316"
mkdir -p "$D"
cp -r "$HERE/capstone/." "$D/"
rm -f "$D/models/"*.pt
mkdir -p "$D/src" "$D/notebooks"
cp "$HERE/code/"*.py "$D/src/"
cp "$HERE/notebooks/"*.ipynb "$D/notebooks/"
sed -i 's#\.\./code/#src/#g; s#`\.\./notebooks/#`notebooks/#g' "$D/README.md" "$D/run_all.sh" "$D/run_pretrained.sh" "$D/src/"*.py
rm -f "$D/data/cache_"*.npz
(cd "$TMP" && zip -qr "$HERE/2430010316.zip" 2430010316)
rm -rf "$TMP"
echo "wrote $HERE/2430010316.zip"
