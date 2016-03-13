---
title: Radial basis function networks with TensorFlow
date: 2016-01-24 12:30
modified: 2016-01-22 19:30
category: article
tags: python, TensorFlow, neural network
authors: Florian Wilhelm
status: draft
summary:
---

[TensorFlow][] from Google is an open source software library for artificial neural
networks (esp. deep learning) and there is a lot of buzz around it right now.
Some find it [terrific][] while others seem to be quite [disappointed][]. I
think that although TensorFlow is not revolutionary as some say, it surely does
a lot of things right especially when it comes to the [interface and architecture][].
My verdict is that TensorFlow is here to stay and this should mean to every
data scientist that it's at least beneficial to become familiar with it.

In order to do so, my strategy is always the same. After having read some [basic tutorials][],
I pick myself a task that comprises some more advanced features of a library and
define a set of achievements I want to fulfill. In order to use TensorFlow as part
of a predictive application that could be deployed productively, I need it to be:

* comfortable to use similar to a regressor in [Scikit-Learn][],
* easy to debug (e.g. visualization of the graph, accessibility of variables etc.) and
* able to store a trained network for later usage.

The new part for me is that I now write, in the spirit of Einstein's "You do not really
understand something unless you can explain it to your grandmother.", an article
about my learnings and experiences.

There are already a lot of neural network examples For TensorFlow to find on the web.
Besides its own examples there is [TensorFlow-Examples][] and in terms of Scikit-Learn
usage [skflow][] is definitely noteworthy. Mainly to do something different, our
task will be to develop an old-school *radial basis function (RBF) network* with TensorFlow.
So let's start with the definition what that exactly is. An RBF network is an
artificial neural network consisting of three layers, input, output and a hidden
layer that uses radial basis functions as activation functions. An RBF is a
real-valued function whose value depend only on the radius, i.e. the distance to
the origin. So every function $$\phi$$ satisfying the property
$$\phi(\mathbf{x}) = \phi(\|\mathbf{x}\|)$$ is an RBF.

<img class="noZoom" src="/images/rbf-network.png" alt="Radial basis function network">


$$\varphi (\mathbf {x} )=\sum _{i=1}^{N}a_{i}\rho (||\mathbf {x} -\mathbf {c} _{i}||)$$


{% notebook ./notebooks/rbf_networks_with_tensorflow.ipynb %}

[TensorFlow]: https://www.tensorflow.org/
[terrific]: http://www.kdnuggets.com/2015/12/tensor-flow-terrific-deep-learning-library.html
[disappointed]: http://www.kdnuggets.com/2015/11/google-tensorflow-deep-learning-disappoints.html
[interface and architecture]: https://github.com/zer0n/deepframeworks/blob/master/README.md
[basic tutorials]: https://www.tensorflow.org/versions/master/tutorials/index.html
[Scikit-Learn]: http://scikit-learn.org/stable/
[California housing dataset]: http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
[TensorFlow-Examples]: https://github.com/aymericdamien/TensorFlow-Examples
[skflow]: https://github.com/tensorflow/skflow
