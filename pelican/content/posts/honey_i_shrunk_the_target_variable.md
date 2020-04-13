---
title: Honey, I shrunk the target variable
date: 2020-03-10 14:00
modified: 2020-03-10 14:00
category: post
tags: python, data science, mathematics
authors: Florian Wilhelm
status: draft
---

## Motivation

For me it is often a joyful sight to see how young, up and coming data scientists jump right into the feature engineering when
facing some new supervised learning problem... but it also makes me contemplating. So full of vigour and enthusiasm, 
they are often completely absorbed by the idea of minimizing whatever error measure they were given or maybe some random one 
they chose themselves, like the [root-mean-square error (RMSE)]. 
In their drive, they construct many derived features using clever transformations and sometimes they do not even stop at
the target variable. Why should they? If the target variable is for instance non-negative and quite right-skewed, why not transform it using the
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
directly into the danger zone. In this blog post, I want to elaborate on why this is so from a mathematical perspective.
Without spoiling too much I hope, for the too busy or plain lazy readers, the main take-away is:

> **TLDR**: Applying any non-[affine transformation] to your target variable, might have unwanted effects on the error measure you are minimizing.
            So if you don't know exactly what you are doing, just don't.


## Let's get started

Before we start with the gory mathematical details, let's first pick and explore a typical use-case where most inexperienced
data scientists might be tempted to transform the target variable without a second thought. In order to demonstrate this, I chose 
the [used-cars database from Kaggle] and if you want to follow along, you find the code here in the notebooks of my Github 
[used-cars-log-trans repository]. As the name suggests, the data set contains used cars having car features like `vehicleType`,
`yearOfRegistration` & `monthOfRegistration`, `gearbox`, `powerPS`, `model`, `kilometer` (mileage), `fuelType`, `brand` and `price`.

Let's say the business unit basically asks us to determine the proper market value of a car given the features above to determine
if its price is actually a good deal, fair deal or a bad deal. The obvious way to approach this problem is create a model
that predicts the price of a car, which we assume to be its value, given its features. 
Since we have roughly 370,000 cars in our data set, for most cars 
we will have many similar cars and thus our model will predict a price that is some kind of average of their prices.
Consequently, we can think of this predicted price (let's call it `pred_price`) as the actual market value. 
To determine if the actual `price` of a vehicle is a good, fair or bad deal, we would then calculate for instance the relative
error 

$$\frac{\mathrm{pred\_price} - \mathrm{price}}{\mathrm{price}}$$

in the simplest case. If the relative error is close to zero we would call it fair, if it is much larger than zero it's a 
good deal and a bad deal if it is much smaller than zero. For the actual subject of this blog post, that serves us already
as a good motivation for the development of some regression model that will predict the price given some car features.
The attentive reader has certainly noticed that the prices in our data set will be biased towards a higher price and thus
also our predicted "market value". This is due to the fact that we don't know for which price the car was eventually sold.
We only know the amount of money the seller wanted to have which is of course higher or equal than what he or she gets in the end.
For the sake of simplicity, we assume that we have raised this point with the business unit, they noted it duly and we
thus neglect it for our analysis.

## Choosing the right error measure

At this point, a lot of inexperienced data scientists would directly get into business of feature engineering and
building some kind of fancy model. Nowadays most machine learning frameworks like [Scikit-Learn] are so easy to use
that one might even forget the error measure that is optimized as in most cases it will be the [mean squared error] by default.
But does the [mean squared error] really make sense for this use-case? First of all is our target measured in some currency,
so why would try to minimize some squared difference? $\mathrm{\euro}^2$? Very clearly, even taking the square root in the 
end, i.e. [root mean squared error], would not change a thing about this fact. Still, we would weight one large residual
higher than many small residuals which sum up to the exact same value as if 10 times a residual of 10.- € is somehow
less severe than a single residual of 100.- €. You see where I am getting at. In our use-case an error measure like the 
[mean absolute error] (MAE) might be the more natural choice compared to the [mean squared error] (MSE).

On the other hand, is it really that important if a car costs you 1000.- € more or less? It definitely does if you
are looking at cars at around 10,000.- € but it might be neglectable if your luxury vehicle is around 100,000.- € anyway.
Consequently, the [mean absolute percentage error] (MAPE) might even be a better fit than the MAE for this use-case.
Having said that, we will keep all those error measures in mind but use to default MSE criterion in our machine-learning
algorithm for the sake of simplicity and to help me making the actual point of this blog post ;-)


## Distribution of the target variable

Our data contains not only cars that are for sale but people searching for a car having a certain price. Additionally,
we have people offering damaged cars, wanting to trade their car for another or just an insanely enormous amount of money. 
For our use-case we gonna keep only real offerings of an undamaged car with a reasonable price between 200.- € and 50,000.- €.
This is how the distribution of the price looks like.

<figure>
<img class="noZoom" src="/images/histtv_price_distribution.png" alt="distribution of price">
<figcaption align="center">Distribution plot of the price variable using 1,000.- € bins.</figcaption>
</figure>
&nbsp;

It surely does look like a [log-normal distribution] and just to have visual check, fitting a log-normal distribution
with the help of the wonderful SciPy gets us this.

<figure>
<img class="noZoom" src="/images/histtv_price_log-normal_fit.png" alt="log-normal fit">
<figcaption align="center">Log-normal distribution fitted to the distribution of prices.</figcaption>
</figure>
&nbsp;

Seeing this, you might feel the itch to just apply now the logarithm to our target variable, just to make it look
more *normal*. And isn't this some basic assumption of a linear model anyway? 

Well, this is a common misconception. The dependent variable, i.e. target variable, of a linear model doesn't need to
be normally distribution, only the residuals are. This can be seen easily by revisiting the formula of a linear model. 
For the observed outcome $y_i$ and our model prediction $\hat y_i$ of the $i$-th sample, we have

\begin{equation}
\begin{split}
y^\star(\mathbf{x}_i) &=  \sum_{j=1}^M w_j \phi_j(\mathbf{x}_i) , \\
y(\mathbf{x}_i) &= y^\star(\mathbf{x}_i) + \epsilon, 
\end{split}\label{eqn:linear-model}
\end{equation}
&nbsp;

where $\mathbf{x}_i$ is the original feature vector, $\phi_j$, $j=1, \ldots, M$ a set of (potentially non-linear) functions
and $\epsilon\sim\mathcal{N}(0, \sigma)$ some random noise with standard deviation $\sigma$.

To make it even a tad more illustrative, imagine you want to predict the average alcohol level (in same strange log scale)
of a person celebrating Carnival only using a single binary feature, i.e. did the person have a one-night-stand over Carnival or not. 
Under these assumptions we simple generate some data using the linear model from above and plot it:

```python
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

N = 10000  # number of people
x = np.random.randint(2, size=N)
y = x + 0.28*np.random.randn(N)

sns.distplot(y)
plt.xlim(-2, 10)
``` 

Obviously, this results in the a bimodal distribution also known as the notorious [Cologne Cathedral distribution] as some may call it.
Thus, although using a linear model, we generated a non-normally distributed target variable with residuals that are normally distributed.

<figure>
<img class="noZoom" src="/images/dom_distribution.png" alt="log-normal fit">
<figcaption align="center">Bimodal distribution generated with a linear model obviously resembling the cathedral of Cologne.</figcaption>
</figure>
&nbsp;

Based on common mnemonic techniques, and assuming this example was surprising, physical, sexual and humorous enough for you, 
you will never forget that the residuals of a linear model are normally distribution and *not* the target variable in general. 
Only in the case that you used a linear model having only an intercept, i.e. $M=1$ and $\phi_1(\mathbf{x})\equiv 1$,
the target distribution equals the residual distribution (up to some shift) in all data sets. But seriously, who does that in real life?


## Analysis of the residual distribution

Now that we learnt about the distribution of the residual, we want to further analyse it. Especially with respect to
the error measure that we are trying to minimize as well as the transformation we apply to the target variable beforehand.
Let's take a look at the definition of the MSE again, i.e.
\begin{equation}
\frac{1}{n}\sum_{i=1}^n (y_i - \hat y_i)^2,\label{eqn:sum_residual}
\end{equation}
where $\hat y_i = \hat y(\mathbf{x}_i)$ is our prediction given the feature vector $\mathbf{x}_i$ and $y_i$
is the observed outcome for the sample $i$. In reality we might only have a single or maybe a few samples sharing
exactly the same feature vector $(\mathbf{x}_i)$ and thus also the same model prediction $\hat y_i$. In order to do same actual analysis, 
we assume now that we have an infinite number of observed outcomes for a given feature vector. Now
assume we keep $\mathbf{x}_i$ fixed and would $\eqref{eqn:sum_residual}$ having all those observed outcomes. Let's drop the index $i$
from $\hat y_i$ as it depends only on our fixed $\mathbf{x}_i$. Since we have now an infinite number of outcomes we need
to incorporate the probability $p(y)$ of a given outcome $y$. Consequently, $\eqref{eqn:sum_residual}$ becomes
\begin{equation}
\int_{-\infty}^\infty (y - \hat y)^2p(y)\mathrm{d}y,\label{eqn:int_residual}
\end{equation}
as you might have expected. Now this is awesome, as it allows us to apply some good, old-school calculus. By the way, when I am talking about the
*residual distribution* I am actually referring to the distribution $y - \hat y$ with $y$ being distributed as $p(y)$ or $y\sim p(y)$ for short.
Thus the residual distribution is determined by $p(y)$ except for a shift of $\hat y$.  So what kind of assumptions can we make about it? 
In case of a linear model as in $\eqref{eqn:linear-model}$, we assume $p(y)$ to be normally distributed but it could also be anything else.
In our car pricing use-case, we now that $p(y)$ will be non-negative as no one is gonna give you money if you take the car. Let me know if you have counter-example ;-)
This rules out a normal distribution and thus a log-normal distribution might be an obvious assumption for $p(y)$ but we will come back later to that.

For now, we gonna consider $\eqref{eqn:int_residual}$ again and note that our model, whatever it is, will somehow try to minimize $\eqref{eqn:int_residual}$ by choosing a proper $\hat y$.
So let's do that analytically by deriving $\eqref{eqn:int_residual}$ with respect to $\hat y$ and setting to $0$, we have that
$$
\frac{d}{d\hat y}\int_{-\infty}^\infty (y - \hat y)^2p(y)\mathrm{d}y = -2\int_{-\infty}^\infty yp(y)\mathrm{d}y + 2\hat y \stackrel{!}{=} 0,
$$
and subsequently
\begin{equation}
\hat y = \int_{-\infty}^\infty yp(y)\mathrm{d}y.\label{eqn:expected-value}
\end{equation}
Looks familiar? Yes, this is just the definition of the [expected value in the continuous case]! So whenever we are 
using the RMSE or MSE as error measure, we are actually calculating the expected value of $p(y)$. So what happens if
we do calculate this for the MAE? In this case we have
$$
\int_{-\infty}^\infty \vert y-\hat y\vert p(y)\, \mathrm{d}y=\int_{\hat y}^\infty (y-\hat y) p(y)\, \mathrm{d}y-\int_{-\infty}^{\hat y} (y-\hat y)p(y)\, \mathrm{d}y,
$$ 
and deriving by $\hat y$ again, we have
$$
\int_{-\infty}^{\hat y}  p(y)\, \mathrm{d}y - \int_{\hat y}^\infty  p(y)\, \mathrm{d}y \stackrel{!}{=} 0.
$$
We thus have $\hat y = P(X\leq \frac{1}{2})$, which is, lo and behold, the [median] of the distribution $p(y)$!

A small recap at this point. We just learnt that minimizing the MSE or RMSE (also [l2-norm] as a fancier name) leads
to the estimation of the expected value of $p(y)$ while minimizing MAE (also known as l1-norm) gets us the median of $p(y)$.
Also remember that our feature vector $\mathbf{x}_i$ is still fixed, so $y\sim p(y)$ just describes the random fluctuations around
some true value $y^\star$ that we just don't know and $\hat y$ is our best guess for it. If we assume a normal distribution
there is no reason to abandon all the nice mathematical properties of the l2-norm since the result will be theoretically the same as
minimizing the l1-norm. It may make a huge different though, if we are dealing with a non-symmetrical distribution like
the log-normal distribution.

Let's just demonstrate this using our used cars example. We have already seen that the distribution of price is rather
log-normally than normally distributed. If we now use the simplest model we can think of, having only a single variable, 
(yeah, here comes the linear model with just an intercept again), the target distribution directly determines the residual
distribution. Now, we fit the variable `yhat` to the price vector `y` using RMSE and MSE to compare the results to 
the mean and median, respectively.  

```python
>>> def rmse(yhat, y):
>>>     yhat = np.resize(yhat, y.size)
>>>     # not taking the root, i.e. MSE, would not change the actual result
>>>     return np.sqrt(np.mean((y - yhat)**2))
 
>>> def mae(yhat, y):
>>>     yhat = np.resize(yhat, y.size)
>>>     return np.mean(np.abs(y - yhat))

>>> y = df.price.to_numpy()
>>> sp.optimize.minimize(rmse, 1., args=(y,))
      fun: 7174.003600843465
 hess_inv: array([[7052.74958795]])
      jac: array([0.])
  message: 'Optimization terminated successfully.'
     nfev: 36
      nit: 5
     njev: 12
   status: 0
  success: True
        x: array([6703.59325181])

>>> np.mean(y)
6704.024314214464

>>> sp.optimize.minimize(mae, 1., options=dict(gtol=2e-4), args=(y,))
      fun: 4743.492333474732
 hess_inv: array([[7862.69627309]])
      jac: array([-0.00018311])
  message: 'Optimization terminated successfully.'
     nfev: 120
      nit: 8
     njev: 40
   status: 0
  success: True
        x: array([4099.9946168])

>>> np.median(y)
4100.0
```

As expected, by looking at the `x` in the output of `minimize`, we approximated the mean by minimizing the RMSE and the median by minimizing the MSE.

## Shrinking the target variable

There is still some elephant in the room that we haven't talked about yet. What happens now if we shrink our target
variable by applying a log transformation and then minimize the MSE?

```python 
>>> y_log = np.log(df.price.to_numpy())
>>> sp.optimize.minimize(rmse, 8., args=(y_log,), tol=1e-16)
      fun: 1.066675943730279
 hess_inv: array([[1.11749076]])
      jac: array([0.])
  message: 'Optimization terminated successfully.'
     nfev: 30
      nit: 5
     njev: 10
   status: 0
  success: True
        x: array([8.29160403])
``` 
So if whe now transform the result `x` which is roughly `8.3` back using `np.exp(8.3)` we get a rounded result of `4024`.
*Wait a second! What just happened!?* We would have expected the final result to be around `6703` because that's the
mean value we had before. Somehow, transforming the target variable, minimizing the same error measure as before and applying the inverse
transformation changed the result. Now our result of `4024` looks rather like an approximation of the median... well...
it actually is assuming a log-normal distribution as we will fully understand soon. 
If we had applied some full-blown machine learning model, the difference would have been much smaller since the variance 
of the residual distribution would have been much smaller. Still, 
we would have missed our goal of minimizing the (R)MSE. Instead we would have unknowingly minimized the MAE, which
might actually be better suited for our use-case at hand. Nevertheless, a data scientist should know what he or she
is doing and a lucky punch without a clue of what happened, just doesn't suit a scientist.

Before we showed that the distribution of prices, and thus our target, resembles a log-normal distribution. So let's assume now that we
have a log-normal distribution, and thus we have $\log(\mathrm{price})\sim\mathcal{N}(\mu,\sigma^2)$. Consequently,
the [probability density function] (pdf) is
\begin{equation}
\tilde p(x) = \frac {1}{x}\cdot {\frac {1}{ {\sqrt {2\pi\sigma^2 \,}}}}\exp \left(-{\frac {(\ln(x) -\mu )^{2}}{2\sigma ^{2}}}\right),\label{eqn:log-normal}
\end{equation}
where the only difference to a normal distribution is $ln(x)$ instead of $x$ and the additional factor $\frac{1}{x}$.
Also note that that parameters $\mu$ and $\sigma$ are tne well-known parameters of the normal distribution but for the
log-transformed target.
So when we now minimize the RMSE of the log-transformed prices as we did before, we actually infer the parameter
$\mu$ of a normal distribution, which is the expected value and also the *median*, i.e. $\operatorname {P} (\mathrm{price}\leq \mu)= 0.5$. 
Applying any kind of strictly monotonic increasing transformation $\varphi$ to the price, we trivially see that 
$\operatorname {P} (\varphi(\mathrm{price})\leq \varphi(\mu)) = 0.5$ and thus the median as well as any other quantile
is equivariant under the transformation $\varphi$. In our specific case from above, we have $\varphi(x) = \exp(x)$ and
thus the result is not surprising at all from a mathematical point of view.

The expected value is not so well-behaved under transformations as the median. Using the definition of the expected
value $\ref{eqn:expected-value}$ we can easily show that only transformations $\phi$ of the form $\phi(x)=ax + b$,
with scalars $a$ and $b$, allow us to transform the target, determine the expected value and apply the
inverse transformation to get the expected value of the original distribution. For the transformed random variable
$\phi(X)$ we have that the expected value
$$
\begin{align*}
E[\phi(X)] &= E[aX + b] \\
            &= \int (ax + b)p(x)\mathrm{d}x \\
            &= a\int xp(x)\mathrm{d}x + b \\
            &=aE[X] + b =\phi(E[X]),
\end{align*}
$$
where we used the fact that probability density functions are normalized, i.e. $\int p(x)\mathrm{d}x=1$. What a relief!
That means at least affine transformations are fine when we minimize the (R)MSE. 
This is especially important if you belong to the illustrious circle of deep learning specialists. In some cases, 
the target variable of a regression problem is standardized or [min-max scaled] during training and transformed back afterwards.
Since these normalization techniques are affine transformations we are on the safe side, though.  

Coming back to our example where we know that our residual transformation is quite log-normally distributed. Can we 
somehow still receive the mean of the untransformed target variable? Yes, we can, actually. Using the parameter $\mu$ that
we already determined above we just calculate the variance $\sigma^2$ and have $\exp(\mu + \frac{\sigma^2}{2})$ for the mean
of the untransformed distribution. More details on how to do this can be found in the notebook
of the [used-cars-log-trans repository]. Way more interesting (at least for the mathematically interested reader) 
is the actually the question *Why does this work?*.

This is easy to see using some high-school calculus. Let $\tilde y = \log(y)$, $p(y)$ be the pdf of the normal distribution
and $\tilde p(y)$ the pdf of the log-normal distribution $\eqref{eqn:log-normal}$. 
Using [integration by substitution] and noting that $\mathrm{d}y = e^{\tilde y}\mathrm{d}\tilde y$, we have
\begin{equation}
\int y \tilde p(y)\mathrm{d}y = \int e^{\tilde y} \tilde p(\tilde y)e^{\tilde y}\mathrm{d}\tilde y = \int e^{\tilde y} p(\tilde y)\mathrm{d}\tilde y,\label{eqn:mean-log-normal}
\end{equation}
where in the last equation the additional factor of the log-normal distribution was canceled out with $e^{\tilde y}$ and thus
became the pdf of a normal distribution due to our substitution. Writing out the exponent in $p(x)$, which is $-\frac{(\tilde y-\mu)^2}{2\sigma^2}$ 
and completing the square with $\tilde y$, we have 
$$
\begin{align*}
\tilde y - \frac{(\tilde y-\mu)^2}{2\sigma^2} &= -\frac{\mu^2 - 2\mu\tilde y + \tilde{y}^2 - 2\sigma^2\tilde y}{2\sigma^2} \\
            &= -\frac{(\tilde y - (\mu + \sigma^2))^2}{2\sigma^2} + \mu + \frac{\sigma^2}{2}.
\end{align*}
$$
Using this result, we can rewrite the last expression of $\eqref{eqn:mean-log-normal}$ by shifting the parameter $\mu$ of the
normal distribution by $\sigma^2$. Denoting with $p_s(y)$ the shifted pdf, we have
$$
\int e^{\tilde y} p(\tilde y) \mathrm{d}\tilde y = e^{\mu + \frac{1}{2}\sigma^2}\int p_s(\tilde y)\mathrm{d}\tilde y = e^{\mu + \frac{\sigma^2}{2}},
$$
and subsequently we have proved that the expected value of the log-normal distribution indeed is $\exp(\mu + \frac{\sigma^2}{2})$.

A little recap of this section's most important points to remember. When minimizing l2, i.e. (R)MSE, only affine transformations
allow us the determine the expected value of original target by applying the inverse transformation to the expected value
of the transformed target variable. When minimizing l2, i.e. MAE, all strictly monotonic increasing transformations can be
applied to determine the median from the transformed target variable.


## Transforming the target for fun and profit

So we have seen that not everything is as it seems or as we might have expected by doing some rather academical analysis.
But can we somehow use this knowledge in our use-case of predicting the market value of a used car?
And this is the point where I close the circle to the beginning of the story. I already argued that the RMSE might not
be the right error measure to minimize. Also we saw that log-transforming the target and still minimizing the RMSE 
gave us an approximation to the result we got if we had minimized the MAE. This is a neat trick if our regression method
only allows minimizing the MSE or is too slow or unstable when minimizing MAE directly. A word of caution again, this
only works if the residual distribution approximates a log-normal distribution.

Well, this is all quite nice, but how about minimizing some relative measure like the MAPE? Assuming that our regression
method does not support minimizing it directly, does the log-transformation do any good here? 
Intuitively, since we know that multiplicative, and thus relative, relationships become additive in log-space, 
we might expect it to be advantageous and indeed it does help. Before we do some experiments, let's first look at some
other relative error measure, the Root Mean Square Percentage Error (RMSPE), i.e.
$$
\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\frac{y_i-\hat y_i}{y_i}\right)^2}.
$$
This error measure was used for evaluation in the [Rossmann Store Sales] Kaggle challenge. Since the RMSPE is a rather
unusual and uncommon error measure, most participants log-transformed the target and minimized the RMSE without giving
too much thought. Let's now evaluate how the error measures RMSE, MAE, MAPE and RMSPE behave in our use-case with 
the raw as well as the log-transformed target before we come back to this.



Definition of log-normal and parameters
We fit MSE that means we get the mean which equals the median in case of a normal distribution. Transforming 
a distribution has the transformed median. Small proof using the the definition of median.
 


For normally distributed residual error and affine transformation.
This shows again why the normal distribution is a mathematician's BFF, no matter how you or it changes over time, it will still be true to you and itself.


* MAE is not continuously differentiable. second derivative is zero
* Affine transformations are fine if residuals are normally distributed
* Back and forth transformation leads to median but is actual expected value (page 2-3 on notes)
* Mention the RMSPE from Kaggle and show calculation
* Show relative error too?

One important thing to note is that when we are fitting the mean squared error, i.e. $\

$\frac{1}{n}\sum_{i=1}^n \Vert y_i - \hat y_i \Vert^2$
$\frac{1}{\vert \mathcal{I}\vert}\sum_{i\in \mathcal{I}} \Vert y_i - \hat y \Vert^2$ 
where the index set $\mathcal I$ is such that $\hat y = f(x)$ is constant within a small neighbourhood of a fixed $x$. 

$\int_{-\infty}^\infty \vert y-\hat y\vert p(y)\, \mathrm{d}y=\int_{\hat y}^\infty (y-\hat y) p(y)\, \mathrm{d}y-\int_{-\infty}^{\hat y} (y-\hat y)p(y)\, \mathrm{d}y$
Differentiating by $\hat y$ and setting to $0$ in order to minimize, we have $\int_{-\infty}^{\hat y}  p(y)\, \mathrm{d}y - \int_{\hat y}^\infty  p(y)\, \mathrm{d}y \stackrel{!}{=} 0$. By the definition of the median, we thus have $\hat y = P(X\leq \frac{1}{2})$, i.e. the median, for any distribution $p(y)$.



[root mean squared error]: https://en.wikipedia.org/wiki/Root-mean-square_deviation
[Scikit-Learn]: https://scikit-learn.org/
[used-cars database from Kaggle]: https://www.kaggle.com/orgesleka/used-cars-database
[root-mean-square error (RMSE)]: https://en.wikipedia.org/wiki/Root-mean-square_deviation
[affine transformation]: https://en.wikipedia.org/wiki/Affine_transformation
[used-cars-log-trans repository]: https://github.com/FlorianWilhelm/used-cars-log-trans
[mean absolute error]: https://en.wikipedia.org/wiki/Mean_absolute_error
[mean squared error]: https://en.wikipedia.org/wiki/Mean_squared_error
[mean absolute percentage error]: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
[log-normal distribution]: https://en.wikipedia.org/wiki/Log-normal_distribution
[Cologne Cathedral distribution]: https://en.wikipedia.org/wiki/Cologne_Cathedral
[expected value in the continuous case]: https://en.wikipedia.org/wiki/Expected_value#Absolutely_continuous_case
[median]: https://en.wikipedia.org/wiki/Median#Probability_distributions
[l2-norm]: https://en.wikipedia.org/wiki/Sequence_space#%E2%84%93p_spaces
[probability density function]: https://en.wikipedia.org/wiki/Probability_density_function
[min-max scaled]: https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
[integration by substitution]: https://en.wikipedia.org/wiki/Integration_by_substitution
[Rossmann Store Sales]: https://www.kaggle.com/c/rossmann-store-sales/
