---
title: Extending Scikit-Learn with your own regressor
date: 2014-07-25 12:00
modified: 2015-12-22 19:30
category: talk
tags: python, scikit-learn, machine-learning
authors: Florian Wilhelm
---

[Scikit-Learn](http://scikit-learn.org/) is a well-known and popular framework for
machine learning that is used by Data Scientists all over the world.
In this tutorial presented at the [EuroPython 2014](https://ep2014.europython.eu/) in Berlin,
I show in a practical way how you can add your own estimator following the interfaces of Scikit-Learn.
First a small introduction to the design of Scikit-Learn and its inner workings is given.
Then I show how easily Scikit-Learn can be extended by creating an own estimator.
In order to demonstrate this, I extend Scikit-Learn by the popular and robust
[Theil-Sen Estimator](http://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator)
that was not in Scikit-Learn until version 0.16.
I also motivate this estimator by outlining some of its superior properties compared
to the ordinary least squares method (LinearRegression in Scikit-Learn).

{% youtube u2tnvWyO3U0 800 500 %}
