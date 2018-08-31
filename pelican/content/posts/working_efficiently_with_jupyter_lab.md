---
title: “Which car fits my life?” - mobile.de’s approach to recommendations
date: 2017-07-02 18:00
modified: 2017-07-02 18:00
category: talk
tags: machine-learning, python, recommender systems
authors: Florian Wilhelm, Arnab Dutta
status: draft
---

```python
import sys
import logging

import numpy as np
import scipy as sp
import sklearn

%load_ext autoreload
%autoreload 2

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import seaborn as sns
sns.set_context("poster")
sns.set(rc={'figure.figsize': (16, 9.)})
sns.set_style("whitegrid")

import pandas as pd
pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

from private_selling import price_cmp
```


alias spark_jupyter='PYSPARK_PYTHON=python3.4 PYSPARK_DRIVER_PYTHON="jupyter" PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser --port=8899" /usr/bin/pyspark2 --master yarn --deploy-mode client --num-executors 20  --executor-memory 10g --executor-cores 5 --conf spark.dynamicAllocation.enabled=false'


Die Sache mit dem Zugriff auf die letzten Element wie in dem Twitter post von Bernhard Schäfer


Das Semikolon erwaehnen um den output der Plots zu unterdruecken.


Tipp von Marcel mit Cursor und dann Shift + Tab um schnell mal das zu kriegen was normalerweise ?? und ? macht.
LaTeX und auch die Magic commands wie auch Shell mit ! erwaehnen.

Multicursor support
Jupyter supports mutiple cursors, similar to Sublime Text. Simply click and drag your mouse while holding down Alt.


Slideshow https://github.com/damianavila/RISE ? 

https://medium.com/netflix-techblog/notebook-innovation-591ee3221233


Splitscreen!!!