---
title: Honey, I shrunk the target variable
date: 2020-03-10 14:00
modified: 2020-03-10 14:00
category: post
tags: python, data science
authors: Florian Wilhelm
status: draft
---

## Motivation

For me it is often a joyful sight to see how young, up and coming data scientists jump right into the feature engineering when
facing some new supervised learning problem... but it also makes me contemplating. So full of vigour and enthusiasm, 
they are often completely absorbed by the idea of minimizing whatever metric they were given or maybe some random metric 
they chose themselves, like the [root-mean-square error (RMSE)](). 
In their drive, they construct many derived features using clever transformations and sometimes they do not even stop at
the target variable. Why should they? If the target variable is for instance positive and quite right-skewed, why not transform it using the
logarithm to make it more normally distributed? Isn't this better or even required for simple models like linear regression,
anyways? A little $\log$ never killed dog, so what could possibly go wrong? 

&nbsp;

<figure>
<p align="center">
<img class="noZoom" src="/images/shrunk_meme.jpg" alt="Couple looking at spoon with magnifier">
</p>
</figure>

&nbsp;

As you might have guessed from these questions, it's not that easy, and transforming your target variable puts you
directly into the danger zone. In this blog post, I want to elaborate a bit on why this is so from a mathematical perspective.
Without spoiling too much I hope, for the too busy or plain lazy readers, the main take-away is:

> **TLDR**: Applying any non-[affine transformation]() to your target variable, might have unwanted effects on the metric you are minimizing.
            So if you don't know exactly what you are doing, just don't.


## Let's get started

Before we start with the gory mathematical details, let's first pick and explore a typical use-case where most inexperienced
data scientists might be tempted to transform the target variable without a second thought. In order to demonstrate this, I chose 
the [used-cars database from Kaggle]() and if you want to follow along, you find the code here in the notebooks of my Github 
[used-cars-log-trans repository]().



[used-cars database from Kaggle]: https://www.kaggle.com/orgesleka/used-cars-database
[root-mean-square error (RMSE)]: https://en.wikipedia.org/wiki/Root-mean-square_deviation
[affine transformation]: https://en.wikipedia.org/wiki/Affine_transformation
[used-cars-log-trans repository]: https://github.com/FlorianWilhelm/used-cars-log-trans
