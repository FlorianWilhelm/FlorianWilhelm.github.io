---
title: Working efficiently with Jupyter Lab Notebooks
date: 2017-07-02 18:00
modified: 2017-07-02 18:00
category: article
tags: python, jupyter
authors: Florian Wilhelm
status: draft
---

## Motivation

If you have ever done something analytical or anything closely related to data science in Python, there is just no way you have not heard of Jupyter or IPython notebooks. In a nutshell, a notebook is a document displayed in your browser which contains source code, e.g. Python and R, as well as rich text elements like paragraphs, equations, figures, links, etc. This combination makes it extremely useful for explorative tasks where the source code, documentation and even visualisations of your analysis are strongly intertwined. Due to this unique characteristic, Jupyter notebooks have achieved a strong adoption particularly in the data science community. But as Pythagoras already noted "If there be light, then there is darkness." and with Jupyter notebooks is no difference of course.

Being in the data science domain for quite some years, I have seen good but also a lot of ugly. Notebooks that are beautifully designed and perfectly convey ideas and concepts by having the perfect balance between text, code and visualisations like in my all time favourite [Probabilistic Programming and Bayesian Methods for Hackers]. In strong contrast to this, and actually more often to find in practise, are notebooks with cells containing pages of incomprehensible source code, distracting you from the actual analysis. Also sharing these notebooks is quite often an unnecessary pain. Notebooks that need you to tamper with the `PYTHONPATH` or to start Jupyter from a certain directory for modules to import correctly. In this blog post I will introduce several best practices and techniques that will help you to create notebooks which are focused, easy to comprehend and to work with. 

## History

Before we get into the actual subject let's take some time to understand how [Project Jupyter] evolved and where it came from. This will also clarify the confusion people sometimes have over IPython, Jupyter and JupyterLab notebooks. In 2001 Fernando Pérez was quite dissatisfied with the capabilities of Python's interactive prompt compared to the commercial notebook environments of Maple and Mathematica which he really liked. In order to improve upon this situation he laid the foundation for a notebook environment by building [IPython] (Interactive Python), a command shell for interactive computing. IPython quickly became a success as the [REPL] of choice for many users but it was only a small step towards a graphical interactive notebook environment. Several years and many failed attempts later, it took until late 2010 for Grain Granger and several others to develop a first graphical console, named [QTConsole] and based on [QT]. As the speed of development picked up, IPython 0.12 was released only one year later in December 2011 and included for the first time a browser-based IPython notebook environment. People were psyched about the possibilities *IPython notebook* provided them and the adoption rose quickly. 

In 2014, [Project Jupyter] started as a spin-off project from IPython for several reasons. At that time IPython encompassed an interactive shell, the notebook server, the QT console and other parts in a single repository with the obvious organisational downsides. After the spin-off, IPython concentrated on providing solely an interactive shell for Python while Project Jupyter itself started as an umbrella organisation for several components like [Jupyter notebook] and [QTConsole], which were moved over from IPython, as well as many others. Another reason for the split was the fact that Jupyter wanted to support other languages besides Python like [R], [Julia] and more. The name Jupyter itself was chosen to reflect the fact that the three most popular languages in data science are supported among others, thus Jupyter is actually an acronym for **Ju**lia, **Pyt**hon, **R**. 

But evolution never stops and the source code of Jupyter notebook built on web technologies of 2011 started to show its age. As the code grew bigger, people also started to realise that it actually is more than just a notebook. Some parts of it rather deal with managing files, running notebooks and parallel workers. This eventually led again to the idea of splitting these functionalities and laid the foundation of [JupyterLab]. JupyterLab is an interactive development environment for working with notebooks, code and data. It has full support for Jupyter notebooks and enables you to use text editors, terminals, data file viewers, and other custom components side by side with notebooks in a tabbed work area. Since February 2018 it's officially considered to be [ready for users] and the 1.0 release is expected to happen end of 2018. 

According to my experience in the last months, JupyterLab is absolutely ready and I recommend everyone to migrate to it. In this post, I will thus focus on JupyterLab and the term notebook or sometimes even Jupyter notebook actually refers to a notebook that was opened with JupyterLab. Practically this means that you run `jupyter lab` instead of `jupyter notebook`. If you are interested in more historical details read the blog posts of [Fernando Pérez] and [Karlijn Willems].


### Installation

### Efficient Workflow

### Useful extensions

### Neat Tricks

### PySpark

### Conclusion


```python
import sys
import logging

import numpy as np
import scipy as sp
import sklearn
import statsmodels.api as sm
from statsmodels.formula.api import ols

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


R Y als shortcut um mal eben den output zu loeschen.

Splitscreen!!!

[Project Jupyter]: http://jupyter.org/
[QT]: https://www.qt.io/
[IPython]: https://ipython.org/
[REPL]: https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop
[JupyterLab]: https://jupyterlab.readthedocs.io/
[ready for users]: https://blog.jupyter.org/jupyterlab-is-ready-for-users-5a6f039b8906
[Probabilistic Programming and Bayesian Methods for Hackers]: https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
[Jupyter notebook]: https://jupyter-notebook.readthedocs.io/
[QTConsole]: https://qtconsole.readthedocs.io/
[R]: https://www.r-project.org/
[Julia]: https://julialang.org/
[Fernando Pérez]: http://blog.fperez.org/2012/01/ipython-notebook-historical.html
[Karlijn Willems]: https://www.datacamp.com/community/blog/ipython-jupyter

