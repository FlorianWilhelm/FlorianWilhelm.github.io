---
title: Explaining the Idea behind Automatic Relevance Determination and Bayesian Interpolation
date: 2016-03-13 22:00
modified: 2016-03-13 22:00
category: talk
tags: scikit-learn, machine-learning, bayesian
authors: Florian Wilhelm
status: published
summary:
---

This talk presented at the [PyData Amsterdam 2016][] explains the idea of Bayesian
model selection techniques, especially the Automatic Relevance Determination.
The slides of this talk are available on [SlideShare][].

Even in the era of Big Data there are many real-world problems where the number
of input features has about the some order of magnitude than the number of samples.
Often many of those input features are irrelevant and thus inferring the relevant
ones is an important problem in order to prevent over-fitting. Automatic Relevance
Determination solves this problem by applying Bayesian techniques.

In order to motivate Automatic Relevance Determination (ARD) an intuition for
the problem of choosing a complex model that fits the data well vs a simple model
that generalizes well is established. Thereby the idea behind Occam's razor is
presented as a way of balancing bias and variance. This leads us to the mathematical
framework of Bayesian interpolation and model selection to choose between different
models based on the data.

To derive ARD as gently as possible the mathematical basics of a simple linear model
are repeated as well as the idea of regularization to prevent over-fitting.
Based on that, the Bayesian Ridge Regression (BayesianRidge in Scikit-Learn) is
introduced. Generalizing the concept of Bayesian Ridge Regression even more gets
us eventually to the the idea behind ARD (ARDRegression in Scikit-Learn).

With the help of a practical example, we consolidate what has been learned so far
and compare ARD to an ordinary least square model. Now we dive deep into the
mathematics of ARD and present the algorithm that solves the minimization problem
of ARD. Finally, some details of Scikit-Learn's ARD implementation are discussed.

[PyData Amsterdam 2016]: http://pydata.org/amsterdam2016/schedule/presentation/17/
[SlideShare]: http://www.slideshare.net/FlorianWilhelm2/explaining-the-idea-behind-automatic-relevance-determination-and-bayesian-interpolation-59498957
