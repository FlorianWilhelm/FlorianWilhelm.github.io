Blog of Florian Wilhelm
=======================

This is my blog hosted on https://florianwilhelm.info/. 
Feel free to let me know about errors and typos.

Installation
------------

```bash
git clone --recurse-submodules https://github.com/FlorianWilhelm/FlorianWilhelm.github.io.git
# or instead of `--recurse-submodules` rather `git submodule update --init`
conda env create -f environment.yaml
conda activate pelican
cd pelican
./generate_site.sh
```


Usage
-----

In order to generate the site:

```bash
conda env create -f environment.yaml
conda activate pelican
cd pelican
# make sure to use bash
./generate_site.sh
```
