---
title: Finally! Bayesian Hierarchical Modeling at Scale.
date: 2018-07-25 18:00
modified: 2018-07-25 18:00
category: post
tags: data science, mathematics, production
authors: Florian Wilhelm
status: draft
---

## Introduction

Since the advent of deep learning, it's all about *Artificial Intelligence*. Even software which is applying traditional
techniques from e.g. instrumentation and control engineering, is nowadays considered *AI*. For instance the famous robots
of Boston Dynamics are not based on deep reinforcement learning as many people think but much more traditional engineering
methods. This hype around AI, which is very often equated with deep learning, seems to draw that much attention such that
great advances of more traditional methods seem to go almost completely unnoticed. In this blog post, I want to draw your 
attention to *bayesian hierarchical modeling* and how modern techniques and frameworks allow you to finally apply this
cool method on really large data sets.

So for starters, what is *bayesian hierarchical modeling* and why should I care? I assume you already have a basic knowledge about
Bayesian inference, otherwise [Probabilitic Programming and Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
is a really starting point to explore the Bayesian rabbit hole. In simple words, Bayesian inference allows you to define
a model with the help probability distributions and also incorporate your prior knowledge about the parameters of your model.
This leads to a *directed acyclic graphical model* (aka Bayesian network) which is explainable, visual and easy to reason about.
But that's not even everything, you also get [Uncertainty Quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification) for free, 
meaning that the model's parameters are not mere point estimates but whole distributions telling you how certain you are
about their values. 

A classical statistical method that most data scientists have learned about early on is *linear regression*. It can also
be easily interpreted in a Bayesian way giving you the possibility to define prior knowledge about the parameters, maybe
that they have to be close to zero or that they are non-negative. Then again, many of these priors you might come up with could also 
be seen as regularizers in a non-Bayesian way, and treated like that, many efficient techniques exist to solve a such formulations.
So where the Bayesian framework now shines is if you consider the following problem setting I stole from the wonderful 
presentation [A Bayesian Workflow with PyMC and ArviZ](https://www.youtube.com/watch?v=WbNmcvxRwow) by Corrie Bartelheimer.

Imagine you want to estimate the price of an apartment in Berlin given the living area in square meters. Making a linear
regression with all data points you have, i.e. a *pooled model*, will lead to a robust estimation of your parameter, i.e. coefficient,
but a wide residual distribution. This is due to the fact that the price of an apartment also heavily depends on the district it is
located in. Now grouping your data with respect to the respective districts results and making a linear regression for each,
i.e. an *unpooled model*, will lead to a much more narrow residual distribution but also a high variance in your parameters since
some district might only have three data points. To combine the advantages of a pooled and unpooled model, one
would intuitively demand that for each district the prior knowledge of the parameter from the pooled model should be used
and updated according to the data we have about a certain district. If we have only a few data points we would only 
deviate a bit from our prior knowledge about the parameter but in case we have lots of data points the parameter for the
respective district could have a huge difference compared to the parameter of the pooled model. Thus the pooled model
acts as an informed prior for the parameters within the unpooled model leading altogether to an *hierarchical model*. 

&nbsp;
<figure>
<img class="noZoom" src="/images/hierarchical_model.png" alt="hierarchical model">
<figcaption align="center">Hierarchical model as a combination of a pooled and an unpooled model. 
Image taken from <a href="https://widdowquinn.github.io/Teaching-Stan-Hierarchical-Modelling/07-partial_pooling_intro.html">Bayesian Multilevel Modelling using PyStan</a>.
</figcaption>
</figure>
&nbsp;

## Recent Advances

So far I mostly used [PyMC3](https://docs.pymc.io/) for bayesian inference or *probabilistic programming* as the authors
of PyMC3 like to call it. I love it for it's elegant design and consequently its expressiveness. The documentation is great
and thus you can pretty much hack away with your model ideas. The only problem I always had with it is that for me it never
scaled so well with somewhat larger data-sets, i.e. more than 100k data points. There is a technical and 
methodical reason for it. Regarding the former, PyMC3 uses [Theano](http://deeplearning.net/software/theano/) to speed
up its computations by transpiling to C. Theano inspired many frameworks like [Tensorflow](https://www.tensorflow.org/) 
and [PyTorch](https://pytorch.org/) but is considered deprecated today and cannot rival the speed of modern frameworks
anymore. For the latter, I mostly used sampling methods, e.g. [Markov chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) (MCMC), 
with PyMC3, which are computationally demanding, while [variational inference (VI) methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)
are much faster. But also using VI, which PyMC3 supports, never really allowed me to deal with larger data-sets rendering
Bayesian Hierarchical Modeling (BHM) a wonderful tool that sadly could not be applied in many suitable projects due to its computational costs
due to large data-sets.

Luckily, the world of data science moves on with an incredible speed, and some time ago I had a nice project at my hand that
could make good use of BHM. Thus, I gave it another shot and also looked beyond PyMC3. My first candidate was [Pyro](http://pyro.ai/)
which uses Stochastic Variational Inference (SVI) by default and calls itself a *deep universal probabilistic programming* framework.
Instead of Theano it based on PyTorch and thus allows for [just-in-time (JIT) compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation),
which sped up my test case already quite a bit. Pyro also emphasizes vectorization, thus allowing for fast parallel computation, e.g. [SIMD](https://en.wikipedia.org/wiki/SIMD) operations.
In total the speed-up compared to PyMC3 was amazing in my test-case letting me almost forget the two downsides of Pyro compared
to PyMC3. Firstly, the documentation of Pyro is not as polished and secondly, it's just so much more complicated to use and understand 
but your mileage may vary. 

Digging through the website of Pyro I then stumbled on [NumPyro](https://github.com/pyro-ppl/numpyro) that has a similar
interface as Pyro but uses [JAX](https://github.com/google/jax) instead of PyTorch as its backend. JAX is like numpy on 
steroids. It's crazy fast as it uses [XLA](https://www.tensorflow.org/xla), which is a domain-specific compiler for linear algebra
operations. Additionally it allows for automatically differentiation like [Autograd](https://github.com/hips/autograd),
whose maintainers moved over to JAX. Long story short, NumPyro blew the benchmark results of Pyro out of the water.
For the first time (at least for what I know), NumPyro allows you do bayesian inference with lots of parameters like in
BHM on large data! In the rest of this post, I want to show how NumPyro can be applied in a typical demand prediction
use-case. Hopefully some readers will find it useful and it mitigates a bit the pain coming from the lack of 
NumPyro's documentation and examples.

## Use-Case


 


Use-cases
- Kanabalisation


A saying in the French mathematics world is *Poisson sans boisson est poison!*. 

No intercept needed! because every day is a day
Mention the fact that sampling site names as string are error-prone.

https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/
