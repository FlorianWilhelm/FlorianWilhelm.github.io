#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT="$( readlink -f "$DIR/../" )"

# Set language to english
export LC_ALL="C"
export LANG="C"

# Clean output directory
cd $OUTPUT
shopt -s extglob
rm -rf !(pelican|_config.yml|CNAME|README.md|google49264a6d4745bc7a.html)
# Generate content
cd $DIR
pelican $DIR/content -o $OUTPUT
# Copy Error 404 page to root for github.io
cp $OUTPUT/404-not-found/index.html $OUTPUT/404.html
# Run webserver
cd $OUTPUT
python -m pelican.server
