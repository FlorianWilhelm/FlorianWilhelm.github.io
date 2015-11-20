Handling Big Data with Python
=============================

:date: 2013-10-17 14:20
:modified: 2015-12-22 19:30
:category: Talks
:tags: python, scikit-learn, big data
:slug: handling-big-data-with-python
:authors: Florian Wilhelm
:summary: Challenges of Big Data and how Blue Yonder addresses them with Python.

The talk presented at the PyCon 2013 in Cologne gives a small introduction of how
`Blue Yonder <http://www.blue-yonder.com/>`_ applies machine learning and Predictive
Analytics in various fields as well as the challenges of Big Data.
Using the example of Blue Yonder's machine learning software NeuroBayes, I show
the made efforts and hit dead ends in order to provide a flexible and yet easy to
use interface for NeuroBayes to Data Scientists.
Since NeuroBayes is written in FORTRAN for performance reasons different interface
approaches were tried which lead us eventually to a Python interface. In the talk
I elaborate on the up- and downsides of the different approaches and the various
reasons why Python won the race with an emphasize on the benefits of the Python ecosystem itself.
Also, I discuss performance as well as scalability issues with Python and how we address them at Blue Yonder.
In detail, I show the application of Cython to speed up calculations in the Python interface
layer as well as distributed computing in a private cloud called Stratosphere.
Scalability and efficiency is of utmost importance when data processing is time critical.
The overall goal is to give the audience an overview how Python fits in the software ecosystem of a company handling Big Data.

.. youtube:: CxinlY8yGUM
  :width: 800
  :height: 500
