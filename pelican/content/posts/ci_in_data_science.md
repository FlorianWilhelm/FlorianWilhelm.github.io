---
title: Continuous Integration in Data Science
date: 2017-04-15 18:00
modified: 2017-06-17 18:00
category: post
tags: python, data science, production
authors: Florian Wilhelm
status: draft
---

# Motivation

A common pattern in most data science projects I participated in is that it's all 
fun and games until someone puts it into production. From that point in time on
no one will any longer give you a pat on the back for a high accuracy and smart
algorithm. All of a sudden the crucial question is how to deploy your model,
which version, how can updates be rolled out, which requirements are needed and so on.

The worst case in such a moment is to realize that up until now your proof of concept
model is not an application but rather a stew of Python/R scripts which were deployed 
by cloning a git repo and run by some Jenkins jobs with a dash of Bash.

Bringing data science to production is a hot topic right now and there are many facets 
to it. This is the first in a series of posts about *data science in production* and
focuses on aspects of modern software engineering like *packaging*, *versioning* as
well as *continuous integration* in general.

# Scripts versus Packages

Being a data scientist does not free you from proper software engineering. Of course
most models start with a simple script or a Jupyter notebook maybe, just the essence
of the your idea to test it quickly. But as your model evolves, the number of lines
of code grow, it's always a good idea to think about the structure of your code and to
move away from writing simple scripts to proper applications or libraries. 

In case of a Python model that means grouping functionality into different modules 
[separating different concerns][] which could be organised in Python packages on a higher
level. Maybe certain parts of the model are even so general that they could be packaged 
into an own library for greater reusability also for other projects. In the context
of Python, a bundle of software to be installed like a library or application is denoted 
with the term "package". Another synonym is "distribution" which is easily to be confused with
a Linux distribution. Therefore the term package is more commonly used although there is an
ambiguity with the kind of package you import in your Python source code (i.e. a container of modules).
 
So what is now the key difference between a bunch of Python scripts with some modules 
and a proper package? A Python package adheres a certain structure and thus  
can be shipped and installed by others. Simple as it sounds this is a major
advantage over having just some Python modules inside a repository. With a package it is possible
to make distinct code releases with different versions that can be stored for later reference. 
Dependencies like *numpy* and *scikit-learn* can be specified and dependency resolution is automated
by tools like [pip][] and [conda][]. Why is this so important? When bugs in production occur 
it's incredibly useful to know which state of your code actually is in production. Is it still
version 0.9 or already 1.0? Did the bug also occur in the last release? Most debugging starts
with reproducing the bug locally on your machine. But what if the release is already half a 
year old and there where major changes in its requirements? Maybe the bug is caused by one of
its dependencies? If your package also includes its dependencies with pinned versions 
restoring the exact some state as in production but inside a local virtual or conda environment 
will be a matter of seconds.


# Packaging and Versioning

Python's history of packaging has had its dark times but nowadays things have pretty much settled 
and now there is pretty much only one obvious tool to do it, namely [setuptools][]. 
A Python [packaging tutorial][] explains the various steps needed to set up a proper ``setup.py``
but it takes a long time to really master the subtleties of Python packaging and even then it
is quite cumbersome. This is the reason many developers refrain from building Python packages.
Another reason is that even if you have a proper Python package set up, proper versioning is
still a manual and thus error-prone process. Therefore the tool [setuptools_scm][] exists which
draws the current version automatically from git so a new release is as simple as creating a new tag.
Following the the famous Unix principle "Do one thing and do it well" also a Python package is
composed of many specialised tools. Besides [setuptools][] and [setuptools_scm][] there 
is [sphinx][] for documentation, testing tools like [pytest][] and [tox][] as well as many other
little helpers to consider when setting up a Python package. Already scared off of Python packaging?

Luckily there is one tool to rule them all, [PyScaffold][], which provides a proper Python 
package within a second.


# Continuous Integration


# Conclusion

ToDo:
- separation of concerns (modularity)
- reusability (unix principle) do one thing and do it well
- unstable, testing, stable
- anaconda different channels
- devpi different indices 
- automatisation
- semantic versioning
- Those who do not understand UNIX are condemned to reinvent it, poorly.


[devpi]: https://doc.devpi.net
[separating different concerns]: https://en.wikipedia.org/wiki/Separation_of_concerns
[pip]: https://pip.pypa.io/
[conda]: https://conda.io/
[packaging tutorial]: https://packaging.python.org/tutorials/distributing-packages/
[setuptools_scm]: https://github.com/pypa/setuptools_scm
[sphinx]: http://www.sphinx-doc.org/
[pytest]: https://docs.pytest.org/
[tox]: https://tox.readthedocs.io/
[PyScaffold]: http://pyscaffold.org/

