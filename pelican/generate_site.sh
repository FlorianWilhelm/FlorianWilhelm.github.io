#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT="$( readlink -f "$DIR/../" )"

# Clean output directory
cd $OUTPUT
shopt -s extglob
rm -rf !(pelican|_config.yml|CNAME|README.md|.nojekyll)
# Generate content
cd $DIR
pelican $DIR/content -o $OUTPUT
# Run webserver
cd $OUTPUT
python -m pelican.server
