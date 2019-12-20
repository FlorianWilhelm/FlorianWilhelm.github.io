Blog of Florian Wilhelm
=======================

This is my blog hosted on https://florianwilhelm.info/. 
Feel free to let me know about errors and typos.

Installation
------------

```bash
git clone --recurse-submodules https://github.com/FlorianWilhelm/FlorianWilhelm.github.io.git
# or instead of `--recurse-submodules` also `git submodule update --init` after cloning
conda env create -f environment.yaml
```


Usage
-----

In order to generate the site:

```bash
conda activate pelican
cd pelican
./generate_site.sh
```
