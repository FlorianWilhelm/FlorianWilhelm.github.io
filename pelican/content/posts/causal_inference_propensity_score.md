---
title: Causal Inference and the Propensity Score
date: 2016-03-13 22:00
modified: 2016-03-13 22:00
category: post
tags: scikit-learn, machine-learning, python, causal inference
authors: Florian Wilhelm
status: draft
---

In the field of machine learning and particularly in supervised learning, correlation is key in order to predict the target variable with the help of the feature variables. Rarely do we think about causation and the actual effect of a single feature variable resp. covariate on the target resp. response. Some even go so far saying that "correlation trumps causation" like in the book "Big Data: A Revolution That Will Transform How We Live, Work, and Think" by Viktor Mayer-Schönberger and Kenneth Cukier. Following their reasoning with Big Data there is no need anymore to think about causation since [nonparametric models][nonparametric] will do just fine using only correlation. For many practical use-cases this point of view seems to be acceptable but surely not for all.

Consider for instance you are managing an advertisement campaign with a budget allowing you to send discount vouchers to 10,000 of your customer base. Obviously, you want to maximise the outcome of the campaign, meaning you want to focus on those customers that buy just because they received a voucher. If $y_{1i}$ resp. $y_{0i}$ is the amount of money spent by a customer $i$ described by $x_i$ having received resp. not having received a voucher, we want to find a subset $I'\subset{}I$ where $I$ is the set of all former customers with $|I'|=10,000$ so that $$\sum_{i\in I'}y_{1i} - y_{0i}$$ is maximised. Another example would be estimating the effect of additional booking options in an online market place. A common use-case of an online car marketplace is to give an estimation to a seller what the effect of an additional booking option (e.g. highlighting, top of page etc.) has on the selling time of a given car.

These two examples outline the need for methods to estimate the actual causal effect of a controllable covariate onto the response. Before we dip our toes into the deep sea of causal inference, a few general aspects are to be considered. The first and maybe not so obvious point when coming from a supervised learning background is that causal inference is thinking about *what did not happen*. That means that we don't know for instance how much a customer that received a voucher would have ordered if he had not received one. This fundamental problem basically renders it an unsupervised learning problem. More interesting aspects of causal inference are summarized in the blog post [10 Things to Know About Causal Inference][10things] by Macartan Humphreys. The question remains how can we estimate the causal effect of a controllable covariate?

## Strongly ignorable

The answer to this question lies in another important and actually the original use-case of causal inference which is the analysis of therapy effects. In a best-case scenario the effect of a therapy can be determined in a randomized trail comparing the response of a treatment group to a control group. In a randomized trail the allocation of the participants to the test or control group is random and thus independent of any covariates $X$.

Following the original paper of Rosenbaum & Rubin[^rosenbaum], in a randomized trial the treatment assignment $Z$ and the (unobservable) potential outcome ${Y_1, Y_0}$ are conditionally independent given the covariates $X$, i.e. $${Y_1, Y_0} ⫫ Z \mid X.$$ Furthermore, we assume that each participant in the experiment has a chance to receive each treatment, i.e. $0 < p(Z=1|x) < 1$. The treatment assignment is said to be *strongly ignorable* if those two conditions hold for our observed covariates $x$. As already mentioned, in an randomized experiment the treatment assignment is strongly ignorable. 
 
### Causal effect in a randomized trail
 
In a randomized trail, the strong ignorability of $Z$ allows us to estimate the effect of the treatment by comparing the response of the treatment group with the one of the control group. The obvious approach for estimating the individual effect of for instance an additional booking option on a car's selling time with machine learning methods like a random forest is as following:

1. train the model with the covariates $X$ and $Z$ as feature and response $Y$ as target,
2. check the quality of the trained model on a test set,
3. predict for a given $x$ the response $\hat{y}_1$ with $Z=1$ and $\hat{y}_0$ with $Z=0$,
4. calculate the effect with $\hat{y}_1 - \hat{y}_0$ or $\frac{\hat{y}_1}{\hat{y}_0}$.

In a real-world, big data problem we often have no control over the experimental setup, we are just left with the data.
 In this case things are really bad since there is no way mathematical way to check if the treatment is strongly ignorable[^pearl1]. According to Pearl[^pearl2] by now assuming strong ignorability we are basically assuming that our covariate set $X$ is *admissible*, i.e. $p(y|\mathrm{do}(z))=p(y|x,z)$. Here, the Pearl's $\mathrm{do}$-notation $p(y|\mathrm{do}(z))$ denotes the “causal effect” of $Z$ on $Y$, i.e. the distribution of $Y$ after setting variable $X$ to a constant $X = x$ by external intervention. In practice the assumption of admissibility of $X$ is often used in order to estimate a causal effect. This also led to wrong results in some studies as well as controversies[^pearl1] and therefore one should always be at least aware that the whole causal analysis stands and falls with it. 
 
 So far we have not only assumed admissibility of $X$ but also a randomized trail for our approach. Therefore we need to check beforehand if $X ⫫ Z$ before applying the former approach. This can be done for instance by using $X$ to predict $Z$. If this is not possible and thus $p(z|x) = p(z)$ the former approach is a viable way. But what if $Z$ is not independent of $X$ as it happens quite often in real-world data. For instance car dealers of expensive car brands might be more willing to spend money and therefore tend to use more booking options. The data from the last marketing campaign presumingly includes a bias induced by the current strategy of the marketing department on how to pick customers that get a voucher. In these cases, we have to isolate the effect of $Z$ from our covariates $X$.
 
### Propensity score
 
By predicting $Z$ based on $X$ without even knowing we have estimated the *propensity score*, i.e. $p(z=1|x)$. This of course assumes that we have used some classification methods that returns probabilities for the classes $z=1$ and $z=0$. 

{% notebook causal_inference_propensity_score.ipynb %}



[^rosenbaum]: Paul R. Rosenbaum, Donald B. Rubin; "The Central Role of the Propensity Score in Observational Studies for Causal Effects"; Biometrika, Vol. 70, No. 1. (Apr., 1983), [pp. 41-55](http://www.stat.cmu.edu/~ryantibs/journalclub/rosenbaum_1983.pdf)
[^pearl1]: Judea Pearl; "CAUSALITY - Models, Reasoning and Inference"; 2nd Edition, 2009, [pp. 348-352](http://bayes.cs.ucla.edu/BOOK-09/ch11-3-5-final.pdf)
[^pearl2]: Judea Pearl; "CAUSALITY - Models, Reasoning and Inference"; 2nd Edition, 2009, [pp. 341-344](http://bayes.cs.ucla.edu/BOOK-09/ch11-3-2-final.pdf)
[^austin]: Peter C. Austin; "An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies"; Multivariate Behav Res. 2011 May; 46(3): [pp. 399–424](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/)

[nonparametric]: https://en.wikipedia.org/wiki/Nonparametric_statistics
[10things]: http://egap.org/methods-guides/10-things-you-need-know-about-causal-inference