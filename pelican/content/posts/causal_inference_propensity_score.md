---
title: Causal Inference and Propensity Score Methods
date: 2017-04-15 18:00
modified: 2017-06-17 18:00
category: post, talk
tags: scikit-learn, machine-learning, python, causal inference
authors: Florian Wilhelm
status: published
---

In the field of machine learning and particularly in supervised learning, correlation is crucial to predict the target variable with the help of the feature variables. Rarely do we think about causation and the actual effect of a single feature variable or covariate on the target or response. Some even go so far as to say that "correlation trumps causation" like in the book "Big Data: A Revolution That Will Transform How We Live, Work, and Think" by Viktor Mayer-Schönberger and Kenneth Cukier. Following their reasoning, with Big Data there is no need to think about causation anymore, since [nonparametric models][nonparametric] will do just fine using correlation alone. For many practical use cases, this point of view may seem acceptable — but surely not for all.

Consider for instance you are managing an advertisement campaign with a budget allowing you to send discount vouchers to 10,000 of your customers. Obviously, you want to maximise the outcome of the campaign, meaning you want to focus on customers that buy because they received a voucher. If $y_{1i}$ and $y_{0i}$ are the amounts of money spent by a customer $i$ described by $x_i$ that either received or did not receive a voucher, we want to find a subset $I'\subset{}I$ where $I$ is the set of all former customers with $|I'|=10,000$ so that $\sum_{i\in I'}y_{1i} - y_{0i}$ is maximised. 

Another example would be to estimate the effect of additional booking options in an online marketplace. A common use case of an online vehicle marketplace is to provide estimations to sellers on the effect that additional booking options (e.g. highlighting, top of page etc.) may have on the selling time of a given vehicle. Likewise, many dealers will be interested in knowing how changing the price of a vehicle will affect the probability of selling the vehicle within a certain period of time or the expected selling time.

These two examples outline the need for methods to estimate the actual causal effect of a controllable covariate onto the response. Before we dip our toes into the deep sea of causal inference, let's consider a few general aspects. The first and maybe not so obvious point when coming from a supervised learning background is that causal inference is thinking about *what did not happen*. That means that we don't know for instance how much a customer that received a voucher would have ordered if they had not received one. This fundamental problem basically renders it an unsupervised learning problem. More interesting aspects of causal inference are summarized in a blog post by Macartan Humphreys on [10 Things to Know About Causal Inference][10things]. The question remains how can we estimate the causal effect of a controllable covariate?

## Strongly ignorable

The answer to this question lies in another important (and actually the original) use case of causal inference, which is the analysis of therapy effects. In a best-case scenario, the effect of a therapy can be determined in a randomized trial by comparing the response of a treatment group to a control group. In a randomized trial, the allocation of participants to the test or control group is random and thus independent of any covariates $X$. Following the original paper of Rosenbaum & Rubin [^rose], in a randomized trial the treatment assignment $Z$ and the (unobservable) potential outcomes ${Y_1, Y_0}$ are conditionally independent given the covariates $X$, i.e. $${Y_1, Y_0} ⫫ Z \mid X.$$ Furthermore, we assume that each participant in the experiment has a chance to receive each treatment, i.e. $0 < p(Z=1|x) < 1$. The treatment assignment is said to be *strongly ignorable* if those two conditions hold for our observed covariates $x$. 
 
## Causal effect in a randomized trial
 
In a randomized trial, the strong ignorability of $Z$ allows us to estimate the effect of the treatment by comparing the response of the treatment group with that of the control group. The following approach may be used to estimate the individual effect of an additional booking option on a vehicle's selling time with machine learning methods like a random forest:

1. Train the model with the covariates $X$ and $Z$ as feature and response $Y$ as target,
2. predict for a given $x$ the response $\hat{y}_1$ with $Z=1$ and $\hat{y}_0$ with $Z=0$,
3. calculate the effect with $\hat{y}_1 - \hat{y}_0$ or $\frac{\hat{y}_1}{\hat{y}_0}$.

In a real-world big-data problem we often have no control over the experimental setup — we are just left with the data. This happens for instance in observational studies. Imagine for instance the treatment with an experimental drug having strong side-effects that might cure a life-threatening disease. A controlled, randomized experiment where the control groups gets a placebo might be impracticable or even unethical. 

In such a nonrandomized experiment, there is no proper mathematical way to check if the treatment is strongly ignorable[^pearl1]. According to Pearl[^pearl2] by now assuming strong ignorability we are basically assuming that our covariate set $X$ is *admissible*, i.e. $p(y|\mathrm{do}(z))=\sum_{x}p(y|x,z)p(x)$. Here, Pearl's $\mathrm{do}$-notation $p(y|\mathrm{do}(z))$ denotes the “causal effect” of $Z$ on $Y$, i.e. the distribution of $Y$ after setting variable $Z$ to a constant $Z = z$ by external intervention. In practice, the assumption of admissibility of $X$ is often used to estimate a causal effect. This led to incorrect results in some studies as well as controversies[^pearl1], so one should always be aware that the entire causal analysis depends on the validity of this assumption. 
 
So far, we have not only assumed admissibility of $X$ but also a randomized trial for our approach. Therefore, we should check beforehand that $X ⫫ Z$, (which is a necessary but not sufficient condition), before applying the aforementioned approach. This can be verified by using $X$ to predict $Z$. If this is not possible, and thus $p(z|x) = p(z)$, the former approach is viable. But what if $Z$ is not independent of $X$ — as is often the case with real-world data. For instance, dealers of expensive vehicle brands might be more willing to spend money and therefore tend to use more booking options. The data from the last marketing campaign presumingly includes a bias induced by the current strategy of the marketing department on how to pick customers that get a voucher. In these cases, we have to isolate the effect of $Z$ from our covariates $X$.
 
## Propensity score
 
By predicting $Z$ based on $X$, we have estimated the *propensity score*, i.e. $p(Z=1|x)$. This of course assumes that we have used a classification method that returns probabilities for the classes $Z=1$ and $Z=0$. Let $e_i=p(Z=1|x_i)$ be the propensity score of the $i$-th observation, i.e. the propensity of the $i$-th participant getting the treatment ($Z=1$). 

We can use the propensity score to define weights $w_i$ to create a synthetic sample in which the distribution of measured baseline covariates is independent of treatment assignment[^austin], i.e. $$w_i=\frac{z_i}{e_i}+\frac{1-z_i}{1-e_i},$$ where $z_i$ indicates if the $i$-th subject was treated. 

The covariates from our data sample $x_i$ are then weighted by $w_i$ to eliminate the correlation between $X$ and $Z$, which is a technique known as *inverse probability of treatment weighting* (IPTW). This allows us to estimate the causal effect via the following approach:
 
 1. Train a model with covariates $X$ to predict $Z$,
 2. calculate the propensity scores $e_i$ by applying the trained model to all $x_i$,
 3. train a second model with covariates $X$ and $Z$ as features and response $Y$ as target by using $w_i$ as sample weight for the $i$-th observation,
 4. use this model to predict the causal effect like in the randomized trial approach.
 
IPTW is based on a simple intuition. For a randomized trial with $p(Z=1)=k$ the propensity score would be equal for all patients, i.e. $e_i=\frac{1}{k}$ and thus $w_i=k$. In a nonrandomized trial, we would assign low weights to samples where the assignment of treatment matches our expectation and high weights otherwise. By doing so, we draw the attention of the machine learning algorithm to the observations where the effect of treatment is most prevalent, i.e. least confounded with the covariates.
 
## Python implementation
 
We can set up a synthetic experiment to demonstrate and evaluate this method with the help of Python and Scikit-Learn. A synthetic experiment is appropriate to address the fundamental problem of causal inference described above. With real data, we  don't know what would have happened if we had not treated someone, sent a voucher or not booked that additional option and vice versa. Therefore, we derive a model that describes the relationship of $X$ to $Y$ as well as the effect of $Z$ on $Y$. We use this model to generate observational data where for each sample $x_i$ we either have $Z=1$ or $Z=0$ and thus incomplete information in our data. Our task is now to estimate the effect of $Z$ with the help of the generated data.
   
The following portion of this article is available for [download]({filename}/notebooks/causal_inference_propensity_score.ipynb) as a Jupyter notebook.

{% notebook causal_inference_propensity_score.ipynb %}

## Final notes

“With great power comes great responsibility” so they say, and IPTW is surely a weapon of math destruction that needs to be handled carefully. Keep in mind that a controlled randomized experiment remains the gold standard by which to  estimate a causal effect and should always be preferred. But when reality hits us hard and we have just the data, i.e. no influence on the experiment generating it, we saw that IPTW improves causal inference to some extent. 

Furthermore, the demonstrated technique relies on several things. First, $X$ needs to be *admissible* — which cannot be practically checked. But we *can* investigate how the treatment was assigned in our data. In our campaign example, that would mean asking the marketing department about their former strategy when sending vouchers to customers. Then we could verify that the data includes all factors on which their strategy relied. Second, we need an accurate method of estimating the propensity scores for this approach to work. For the sake of simplicity, our demonstration did not check the prediction quality of our machine learning models on a test set, which would be advisable in a real application. The application of machine learning models should always encompass training, validation, test splits and a proper cost functional.

That said, propensity score techniques like IPTW can be very useful. Results can be improved further by first using only the covariates to estimate the recovery time, followed by a residual training with the treatment and the sample weighting to further guide the machine learning algorithm by isolating the causal effect of the treatment — but this is beyond the scope of this post. An overview of other propensity score methods like propensity score matching, stratification on the propensity score and covariate adjustment using the propensity score are well explained in the [propensity score methods introduction][propensity_introduction] by Peter Austin[^austin].

## PyData meetup talk

A talk about this blog post was presented at PyData meetup in Berlin, April 19th:

{% youtube tUq4esYY6CY 800 500 %}

## References

[^stuart]: E. Stuart; [The why, when, and how of propensity score methods for estimating causal effects](http://www.preventionresearch.org/wp-content/uploads/2011/07/SPR-Propensity-pc-workshop-slides.pdf); Johns Hopkins Bloomberg School of Public Health, 2011 
[^rose]: Paul R. Rosenbaum, Donald B. Rubin; "The Central Role of the Propensity Score in Observational Studies for Causal Effects"; Biometrika, Vol. 70, No. 1., Apr., 1983, [pp. 41-55](http://www.stat.cmu.edu/~ryantibs/journalclub/rosenbaum_1983.pdf)
[^pearl1]: Judea Pearl; "CAUSALITY - Models, Reasoning and Inference"; 2nd Edition, 2009, [pp. 348-352](http://bayes.cs.ucla.edu/BOOK-09/ch11-3-5-final.pdf)
[^pearl2]: Judea Pearl; "CAUSALITY - Models, Reasoning and Inference"; 2nd Edition, 2009, [pp. 341-344](http://bayes.cs.ucla.edu/BOOK-09/ch11-3-2-final.pdf)
[^austin]: Peter C. Austin; "An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies"; Multivariate Behav Res. 2011 May; 46(3): [pp. 399–424](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/)

[nonparametric]: https://en.wikipedia.org/wiki/Nonparametric_statistics
[10things]: http://egap.org/methods-guides/10-things-you-need-know-about-causal-inference
[propensity_introduction]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/
