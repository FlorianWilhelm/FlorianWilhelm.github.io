---
title: Finally! Bayesian Hierarchical Modeling at Scale
date: 2018-07-25 18:00
modified: 2018-07-25 18:00
category: post
tags: data science, mathematics, production
authors: Florian Wilhelm
status: draft
---

## Introduction

Since the advent of deep learning, everything is or has to be about *Artificial Intelligence*, so it seems. Even software which is applying traditional
techniques from e.g. instrumentation and control engineering, is nowadays considered *AI*. For instance, the famous robots
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
acts as an informed prior for the parameters within the unpooled model leading altogether to an *hierarchical model*,
which is sometimes also referred to as *partially pooled model*. 

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
interface as Pyro but uses [JAX](https://github.com/google/jax) instead of PyTorch as its backend. JAX is like [NumPy](https://numpy.org/) on 
steroids. It's crazy fast as it uses [XLA](https://www.tensorflow.org/xla), which is a domain-specific compiler for linear algebra
operations. Additionally it allows for automatically differentiation like [Autograd](https://github.com/hips/autograd),
whose maintainers moved over to develop JAX. Long story short, NumPyro blew the benchmark results of Pyro out of the water.
For the first time (at least for what I know), NumPyro allows you do bayesian inference with lots of parameters like in
BHM on large data! In the rest of this post, I want to show how NumPyro can be applied in a typical demand prediction
use-case on some public data-set. The data-set in my actual use-case was much bigger but you have to just trust me on this one ;-)
Hopefully some readers will find this post useful and maybe it mitigates a bit the pain coming from the lack of NumPyro's documentation and examples.

## Use-Case & Modelling

Imagine you have many retail stores and want to make individual demand predictions for them. For stores that were opened
a long time ago, this should be no problem but how do you deal with stores that first opened a week ago? Like in the example
of apartment prices in different districts, BHM helps you to deal exactly with this problem. We take the 
[Rossmann data-set from Kaggle](https://www.kaggle.com/c/rossmann-store-sales) to simulate this problem by removing the data
of some of the stores. The data consists of Here's our experiment and study protocol:

1. Join the data from Kaggle's `train.csv` dataset, which specifies on a daily basis for each store our target variable (`Sales`) as well
 as other features like if a promotion happened (`promo`), with the general store features from the `store.csv` dataset.
2. Perform some really basic feature engineering and encoding of some categorical features
3. Split the data into train and test where we treat the stores from train as being opened for a long time and the ones
  from test as newly opened. 
4. Fit our hierarchical model on train dataset to infer the "global" parameters on the upper hierarchy
5. Take only the first 7 days for each store in the test data, which we assume to know, and fit our model only inferring
 the local, i.e. store-specific, parameters from the lower hierarchy while keeping the global ones fixed.
6. Compare the inferred parameters of a test store to:
    1. the inferred local parameters of a simple Poisson model. We expect to them to be completely off due to the lack
       of data and thus overfitting.
    2. the inferred local parameters of our model if we had given it the whole time series from test, i.e. not only the first 7 days.
       In this case, we assume that we are already pretty close since the priors given by the global parameters nudge them
       in the right direction even with only little data.
       
All code of this little experiment can be found under my [bhm-at-scale repository](https://github.com/FlorianWilhelm/bhm-at-scale)
so that you can follow along easily.
The steps 1-3 are performed in the [preprocessing notebook](https://github.com/FlorianWilhelm/bhm-at-scale/blob/master/notebooks/01-preprocessing.ipynb)
and are actually not that interesting, thus we will skip it here. Fitting the model and some evaluation is done in the
[model notebook](https://github.com/FlorianWilhelm/bhm-at-scale/blob/master/notebooks/02-model.ipynb) while some visualisation
is done in the [visualize notebook](https://github.com/FlorianWilhelm/bhm-at-scale/blob/master/notebooks/03-visualize.ipynb).

But before we start to get technical, let's take a minute and frame again the forecasting problem from a more mathematical side.
The data of each store is a time-series of feature vectors and target scalars. We want to find a mapping such that the
feature vector of each time-step is mapped to a value close to the target scalar of the respective time-step. Since our target
value, i.e. the number of sales, is a non-negative integer we could assume a [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution) and consequently
perform a [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression) in a hierarchical way. This would be kind of okay
if we were only interested in a point estimation and thus would not care about the variance of the predictive posterior distribution. 
The Poisson distribution only has one parameter $\lambda$ that allows you to define the mean $\mu$ and the variance $\sigma^2$ then just equals the mean as there is no way to adjust the variance independently.   
In many practical use-cases, there is [overdispersion](https://en.wikipedia.org/wiki/Overdispersion), meaning that the variance is larger than the mean
and thus a so called *dispersion parameter* $r\in(0,\infty)$ is introduced for instance in the [negative binomial distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution), i.e.

$$\mathrm{NB}(y;\mu,r) = \frac{\Gamma(r+y)}{y!\cdot\Gamma(r)}\cdot\left(\frac{r}{r+\mu}\right)^r\cdot\left(\frac{\mu}{r+\mu}\right)^y,$$

where $\Gamma$ is the [Gamma function](https://en.wikipedia.org/wiki/Gamma_function). Now we have

$$\sigma^2=\mu + \frac{1}{r}\mu^2$$

and using $r$ we are thus able to adjust the variance from $\mu$ to $\infty$. Another name for the negative binomial 
distribution is Gamma-Poisson distribution and this is the name under which we find it in NumPyro. I also find this name 
much more catchy since you can imagine a Poisson distribution with its only parameter drawn from a Gamma distribution that
has two parameters $\alpha$ and $\beta$. This also intuitively explains why the variance of NB is bounded below by its mean.

Uncertainty Quantification is crucial for sales forecasts in retail although peculiarly, no one really cares about forecasts in retail anyway.
What retailers really care about is optimal replenishment, meaning that they want to have a system telling them how much
to order so that there is an optimal amount of stocks available in their store. In order to provide optimal replenishment
suggestions you need sales forecasts that provide probability distributions, not only point estimations. With the help
of those distributions the replenishment system basically runs an optimization with respect to some cost function, e.g.
cost of missed sale is weighted 3 times the cost of an written-off product, and further constraints, e.g. if products can only be ordered in bundles of 10. 
For these reasons we will use the NB that allows us the quantify the uncertainties in our sales predictions.

So now that we settled with NB as the distribution that we want to fit to the daily sales of our stores $\mathbf{y}$, we can think
about incorporating our features $\mathbf{x}$. We want to use a linear model to map $\mathbf{x}$ to $\mathbf{\mu}$ such
that we can use it later to calculate $\alpha$ and $\beta$ of NB. 

<figure>
<p align="center">
<img class="noZoom" src="/images/bhm_model.png" alt="hierarchical model" width="60%">
</p>
<figcaption align="center">Hierarchical model as a combination of a pooled and an unpooled model. 
Image taken from <a href="https://widdowquinn.github.io/Teaching-Stan-Hierarchical-Modelling/07-partial_pooling_intro.html">Bayesian Multilevel Modelling using PyStan</a>.
</figcaption>
</figure>

Model that generates the data and reference to plates.


https://minimizeregret.com/post/2018/01/04/understanding-the-negative-binomial-distribution/


Use-cases
- Kanabalisation


A saying in the French mathematics world is *Poisson sans boisson est poison!*. 

No intercept needed! because every day is a day
Mention the fact that sampling site names as string are error-prone.

https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/
https://docs.pymc.io/notebooks/multilevel_modeling.html#:~:text=Hierarchical%20or%20multilevel%20modeling%20is,parameters%20are%20given%20probability%20models.&text=A%20hierarchical%20model%20is%20a,are%20nested%20within%20one%20another.