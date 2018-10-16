---
title: Working efficiently with JupyterLab Notebooks
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

But evolution never stops and the source code of Jupyter notebook built on the web technologies of 2011 started to show its age. As the code grew bigger, people also started to realise that it actually is more than just a notebook. Some parts of it rather deal with managing files, running notebooks and parallel workers. This eventually led again to the idea of splitting these functionalities and laid the foundation for [JupyterLab]. JupyterLab is an interactive development environment for working with notebooks, code and data. It has full support for Jupyter notebooks and enables you to use text editors, terminals, data file viewers, and other custom components side by side with notebooks in a tabbed work area. Since February 2018 it's officially considered to be [ready for users] and the 1.0 release is expected to happen end of 2018. 

According to my experience in the last months, JupyterLab is absolutely ready and I recommend everyone to migrate to it. In this post, I will thus focus on JupyterLab and the term notebook or sometimes even Jupyter notebook actually refers to a notebook that was opened with JupyterLab. Practically this means that you run `jupyter lab` instead of `jupyter notebook`. If you are interested in more historical details read the blog posts of [Fernando Pérez] and [Karlijn Willems].


## Preparation & Installation

The first good practice can actually be learnt before even starting JupyterLab. Since we want our analysis to be reproducible and shareable with colleagues it's a good practice to create a clean, isolated environment for every task. For Python you got basically to options [virtualenv] (also descendants like [pipenv]) or [conda] to achieve this. Since in the field of data science [conda] is more common, we will use it in this tutorial. For this, I assume you have [Miniconda] installed on your system. Besides this, every programmer's machine should have [Git] installed and set up.

In the spirit of Phil Karlton who supposedly said "There are only two hard things in Computer Science: cache invalidation and naming things.", we gonna select a specific task, namely an analysis based on the all familiar [Boston housing dataset], to help us finding crisp names. Based on our task we create an environment `bostong_housing` including Python and some common data science libraries with:

```commandline
conda create -n boston_housing python=3.6 jupyterlab scikit-learn seaborn
```

After less than a minute the environment is ready to be used and we can activate it with `conda activate boston_housing`.


## Efficient Workflow

### 1. Use Packages

The code in notebooks tends to grow and grow to the point of being incomprehensible. To overcome this problem, the only way is to extract parts of it into Python modules once in a while. Since it only makes sense to extract functions and classes into Python modules, I often start cleaning up a messy notebook by thinking about the actual task a group of cells is accomplishing. This helps me to refactor those cells into a proper function which I can then migrate into a Python module. 

At that point where you create your custom modules, things get trickier. By default Python will only allow you to import modules that are installed in your environment or in your current working directory. Due to this behaviour many people start creating their custom modules in the directory holding their notebook. Since JupyterLab is nice enough to set the current working directory to the directory containing you notebook everything is fine at the beginning. But as the number of notebooks that share certain functionality imported from modules grow, the single directory containing notebooks and modules will get messier as you go. The obvious split of notebooks and modules into different folders or even organizing your notebooks into different folders will not work with this approach since then your imports fail. 


This observation brings us to one of the most important best practices: **develop your code as a Python package**. A Python package will allow you to keep structure your code nicely over several modules and even subpackages, you can easily create unit tests and the best part of it is that distributing and sharing it with your colleagues comes for free. *But creating a Python package is so much overhead; surely it's not worth this small little analysis I will complete in a half a day anyway and then forget about it*, I hear you say. Well, how often is this actually true? Things always start out small but then get bigger and messier if you don't adhere to a certain structure right from the start. But more importantly coming back to he first part of your comment, if you know how, it's no overhead at all!





`notebooks` folder

`putup boston_housing`

`python setup.py develop`


putub 

### Isolated environment

### Remote sessions

## Useful extensions

## Neat Tricks

## PySpark

## Conclusion


Verweis auf das Repo.


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


M Y als shortcut um mal eben den output zu loeschen.

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
[Miniconda]: https://conda.io/miniconda.html
[conda]: https://conda.io/
[virtualenv]: https://virtualenv.pypa.io/
[pipenv]: https://pipenv.readthedocs.io/
[git]: https://git-scm.com/
[Boston housing dataset]: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

