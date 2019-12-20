set -e

#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ "$(uname)" == "Darwin" ]; then
    OUTPUT="$DIR/../"
else # Linux / Unix
    OUTPUT="$( readlink -f "$DIR/../" )"
fi

# Set language to english
export LC_ALL="C"
export LANG="C"

# Clean output directory
cd $OUTPUT
shopt -s extglob
rm -rf !(pelican|_config.yml|CNAME|README.md|google49264a6d4745bc7a.html|googlef0310dc40eb99e21.html|environment.yaml)
# Generate content
cd $DIR
pelican $DIR/content -o $OUTPUT
# Copy Error 404 page to root for github.io
cp $OUTPUT/404-not-found/index.html $OUTPUT/404.html
# Run webserver
cd $OUTPUT
echo "Open http://localhost:8000"
python -m pelican.server
