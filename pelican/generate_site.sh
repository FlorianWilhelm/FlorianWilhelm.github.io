#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT="$( readlink -f "$DIR/../" )"

pelican $DIR/content -o $OUTPUT
cd $OUTPUT
rm -rf !(pelican)
python -m pelican.server
