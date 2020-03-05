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
they chose themselves, like the [root-mean-square error (RMSE)]. 
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
directly into the danger zone. In this blog post, I want to elaborate on why this is so from a mathematical perspective.
Without spoiling too much I hope, for the too busy or plain lazy readers, the main take-away is:

> **TLDR**: Applying any non-[affine transformation] to your target variable, might have unwanted effects on the metric you are minimizing.
            So if you don't know exactly what you are doing, just don't.


## Let's get started

Before we start with the gory mathematical details, let's first pick and explore a typical use-case where most inexperienced
data scientists might be tempted to transform the target variable without a second thought. In order to demonstrate this, I chose 
the [used-cars database from Kaggle] and if you want to follow along, you find the code here in the notebooks of my Github 
[used-cars-log-trans repository]. As the name suggests, the data set contains used cars having car features like `vehicleType`,
`yearOfRegistration` & `monthOfRegistration`, `gearbox`, `PowerPS`, `model`, `kilometer` (mileage), `fuelType`, `brand` and `price`.

Let's say the business unit basically asks us to determine the proper market value of a car given the features above to determine
if its price is actually a good deal, fair deal or a bad deal. The obvious way to approach this problem is create a model
that predicts the price of a car, which we assume to be its value, given its features. 
Since we have roughly 370,000 cars in our data set, for most cars 
we will have many similar cars and thus our model will predict a price that is some kind of average of their prices.
Consequently, we can think of this predicted price (let's call it `pred_price`) as the actual market value. 
To determine if the actual `price` of a vehicle is a good, fair or bad deal, we would then calculate for instance the relative
error 

$$\frac{\mathrm{pred_price} - \mathrm{price}}{\mathrm{pred_price}}$$

in the simplest case. If the relative error is close to zero we would call it fair, if it is much larger than zero it's a 
good deal and a bad deal if it is much smaller than zero. For the actual subject of this blog post, that serves us already
as a good motivation for the development of some regression model that will predict the price given some car features.
The attentive reader has certainly noticed that the prices in our data set will be biased towards a higher price and thus
also our predicted "market value". This is due to the fact that we don't know for which price the car was eventually sold.
We only know the amount of money the seller wanted to have which is of course higher or equal than what he or she gets in the end.
For the sake of simplicity, we assume that we have raised this point with the business unit, they noted it duly and we
thus neglect it for our analysis.

## Choosing the right metric

At this point, a lot of inexperienced data scientists would directly get into business of feature engineering and
building some kind of fancy model. Nowadays most machine learning frameworks like [Scikit-Learn] are so easy to use
that one might even forget the metric you are optimizing as in most cases it will be the [mean squared error] by default.
But does the [mean squared error] really make sense for this use-case? First of all is our target measured in some currency,
so why would try to minimize some squared difference? $\mathrm{\euro}^2$? Very clearly, even taking the square root in the 
end, i.e. [root mean squared error], would not change a thing about this fact. Still, we would weight one large residual
higher than many small residuals which sum up to the exact same value as if 10 times a residual of 10,- € is somehow
less severe than a single residual of 100,- €. You see where I am getting at. In our use-case a metric like the 
[mean absolute error] might be the more natural choice compared to the [mean squared error].

Obsession with MSE





* Linear model
* Notice distribution of residuals, not the target variable
* Example of dom distribution, people having affiars on carneval
* Thinking about metrics, [mean squared error] -> expectation value, [mean absolute error] -> Median
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
