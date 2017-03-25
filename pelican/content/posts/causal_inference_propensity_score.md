---
title: Causal Inference and the Propensity Score
date: 2017-04-09 22:00
modified: 2017-04-09 22:00
category: post
tags: scikit-learn, machine-learning, python, causal inference
authors: Florian Wilhelm
status: draft
---

In the field of machine learning and particularly in supervised learning, correlation is key in order to predict the target variable with the help of the feature variables. Rarely do we think about causation and the actual effect of a single feature variable resp. covariate on the target resp. response. Some even go so far saying that "correlation trumps causation" like in the book "Big Data: A Revolution That Will Transform How We Live, Work, and Think" by Viktor Mayer-Schönberger and Kenneth Cukier. Following their reasoning with Big Data there is no need anymore to think about causation since [nonparametric models][nonparametric] will do just fine using only correlation. For many practical use-cases this point of view seems to be acceptable but surely not for all.

Consider for instance you are managing an advertisement campaign with a budget allowing you to send discount vouchers to 10,000 of your customer base. Obviously, you want to maximise the outcome of the campaign, meaning you want to focus on those customers that buy just because they received a voucher. If $y_{1i}$ resp. $y_{0i}$ is the amount of money spent by a customer $i$ described by $x_i$ having received resp. not having received a voucher, we want to find a subset $I'\subset{}I$ where $I$ is the set of all former customers with $|I'|=10,000$ so that $$\sum_{i\in I'}y_{1i} - y_{0i}$$ is maximised. Another example would be estimating the effect of additional booking options in an online market place. A common use-case of an online car marketplace is to give an estimation to a seller what the effect of an additional booking option (e.g. highlighting, top of page etc.) has on the selling time of a given car.

These two examples outline the need for methods to estimate the actual causal effect of a controllable covariate onto the response. Before we dip our toes into the deep sea of causal inference, a few general aspects are to be considered. The first and maybe not so obvious point when coming from a supervised learning background is that causal inference is thinking about *what did not happen*. That means that we don't know for instance how much a customer that received a voucher would have ordered if he had not received one. This fundamental problem basically renders it an unsupervised learning problem. More interesting aspects of causal inference are summarized in the blog post [10 Things to Know About Causal Inference][10things] by Macartan Humphreys. The question remains how can we estimate the causal effect of a controllable covariate?

## Strongly ignorable

The answer to this question lies in another important and actually the original use-case of causal inference which is the analysis of therapy effects. In a best-case scenario the effect of a therapy can be determined in a randomized trial comparing the response of a treatment group to a control group. In a randomized trial the allocation of the participants to the test or control group is random and thus independent of any covariates $X$. Following the original paper of Rosenbaum & Rubin [^rose], in a randomized trial the treatment assignment $Z$ and the (unobservable) potential outcome ${Y_1, Y_0}$ are conditionally independent given the covariates $X$, i.e. $${Y_1, Y_0} ⫫ Z \mid X.$$ Furthermore, we assume that each participant in the experiment has a chance to receive each treatment, i.e. $0 < p(Z=1|x) < 1$. The treatment assignment is said to be *strongly ignorable* if those two conditions hold for our observed covariates $x$. As already mentioned, in a randomized experiment the treatment assignment is strongly ignorable. 
 
## Causal effect in a randomized trial
 
In a randomized trial, the strong ignorability of $Z$ allows us to estimate the effect of the treatment by comparing the response of the treatment group with the one of the control group. The obvious approach for estimating the individual effect of for instance an additional booking option on a car's selling time with machine learning methods like a random forest is as following:

1. train the model with the covariates $X$ and $Z$ as feature and response $Y$ as target,
2. predict for a given $x$ the response $\hat{y}_1$ with $Z=1$ and $\hat{y}_0$ with $Z=0$,
3. calculate the effect with $\hat{y}_1 - \hat{y}_0$ or $\frac{\hat{y}_1}{\hat{y}_0}$.

In a real-world, big data problem we often have no control over the experimental setup, we are just left with the data. This happens for instance in observational studies. Imagine for instance the treatment with an experimental drug having strong side-effects that might cure a life-threatening disease. A controlled, randomized experiment where the control groups gets a placebo might be impracticable or even unethical. 

In such a nonrandomized experiment, things are really bad since there is no proper mathematical way to check if the treatment is strongly ignorable[^pearl1]. According to Pearl[^pearl2] by now assuming strong ignorability we are basically assuming that our covariate set $X$ is *admissible*, i.e. $p(y|\mathrm{do}(z))=\sum_{x}p(y|x,z)p(x)$. Here, the Pearl's $\mathrm{do}$-notation $p(y|\mathrm{do}(z))$ denotes the “causal effect” of $Z$ on $Y$, i.e. the distribution of $Y$ after setting variable $X$ to a constant $X = x$ by external intervention. In practice the assumption of admissibility of $X$ is often used in order to estimate a causal effect. This also led to wrong results in some studies as well as controversies[^pearl1] and therefore one should always be at least aware that the whole causal analysis stands and falls with it. 
 
 So far we have not only assumed admissibility of $X$ but also a randomized trial for our approach. Therefore we can should check beforehand that $X ⫫ Z$, which is a necessary but not sufficient condition, before applying the former approach. This can be done for instance by using $X$ to predict $Z$. If this is not possible and thus $p(z|x) = p(z)$ the former approach is a viable way. But what if $Z$ is not independent of $X$ as it happens quite often in real-world data. For instance car dealers of expensive car brands might be more willing to spend money and therefore tend to use more booking options. The data from the last marketing campaign presumingly includes a bias induced by the current strategy of the marketing department on how to pick customers that get a voucher. In these cases, we have to isolate the effect of $Z$ from our covariates $X$.
 
## Propensity score
 
By predicting $Z$ based on $X$ without even knowing we have estimated the *propensity score*, i.e. $p(Z=1|x)$. This of course assumes that we have used some classification methods that returns probabilities for the classes $Z=1$ and $Z=0$. Let $e_i=p(Z=1|x_i)$ be the propensity score of the $i$-th observation, i.e. the propensity of the $i$-th participant getting the treatment ($Z=1$). We can make use of the propensity score to define weights $w_i$ in order to create a synthetic sample in which the distribution of measured baseline covariates is independent of treatment assignment[^austin], i.e. $$w_i=\frac{z_i}{e_i}+\frac{1-z_i}{1-e_i},$$ where $z_i$ indicates if the $i$-th subject was treated. The covariates from our data sample $x_i$ are then weighted by $w_i$ to eliminate the correlation between $X$ and $Z$ which is a technique known as *inverse probability of treatment weighting* (IPTW). Taken as a whole this allows us to define the following approach to estimate the causal effect:
 
 1. train a model with covariates $X$ in order to predict $Z$,
 2. calculate the propensities scores $e_i$ by applying the trained model to all $x_i$,
 3. train a second model with covariates $X$ and $Z$ as features and response $Y$ as target by using $w_i$ as sample weight for the $i$-th observation,
 4. use this model to predict the causal effect like in the approach of the randomized trial.
 
There is actually a quite simple intuition behind IPTW. First of all we note that in case of a randomized trial the propensity score would be equal for all patients, i.e. $e_i=\frac{1}{2}$ and thus $w_i=2$. In a nonrandomized trial we would assign low weights to samples where the assignment of treatment is according to our expectation and height weights in opposite cases. By doing so we draw the attention of the machine learning algorithm to the observations where the effect of treatment is most dominant.
 
## Python implementation
 
In order to demonstrate and evaluate this method with the help of Python and Scikit-Learn we set up a synthetic experiment. The reason for a synthetic experiment is due to the fundamental problem of causal inference described above. With real data we just don't know what would have happened if we had not treated someone, sent a voucher or not booked that additional option and vice versa. Therefore, we come up with a model that describes the relationship of $X$ on $Y$ else well as the effect of $Z$ on $Y$. We use this model to generate observational data where for each sample $x_i$ we either have $Z=1$ or $Z=0$ and thus incomplete information in our data. Our task is now to estimate the effect of $Z$ with the help of the generated data.
   
The remaining part of this blog post is a Jupyter notebook and can also be downloaded [here]({filename}/notebooks/causal_inference_propensity_score.ipynb).

{% notebook causal_inference_propensity_score.ipynb %}

## Final notes

With great power comes great responsibility, so they say, and IPTW surely is a weapon of math destruction that needs to be handled carefully. Keep in mind that a controlled randomized experiment remains the gold standard to estimate a causal effect and should always be preferred. Furthermore, the demonstrated techniques relies on several things. Firstly, $X$ needs to be *admissible* which cannot practically be practically checked. What can be done though is an investigation on how the treatment was assigned in our data. In our campaign example, that would mean to ask the marketing department about their strategy when sending vouchers to customers. Then one would make sure that the data at hand includes all factors on wich their strategy relied on. Secondly, for this method to work a way to accurately estimate the propensity scores is needed. For the sake of simplicity in our demonstration we did not even check the prediction quality of our machine learning models on a test set which would be quite sloppy in a real application. It should go unsaid that the application of machine learning models should always encompass training, validation, test splits and a proper cost functional. That being said, this technique might be quite handy at times. Further improvements are achieved by first using only the covariates to estimate the recovery time followed by a residual training with the treatment and the sample weighting to further guide the machine learning algorithm by isolating the causal effect of the treatment but this is beyond the scope of this post.


## References

[^stuart]: E. Stuart; [The why, when, and how of propensity score methods for
estimating causal effects](http://www.preventionresearch.org/wp-content/uploads/2011/07/SPR-Propensity-pc-workshop-slides.pdf); Johns Hopkins Bloomberg School of Public Health, 2011 
[^rose]: Paul R. Rosenbaum, Donald B. Rubin; "The Central Role of the Propensity Score in Observational Studies for Causal Effects"; Biometrika, Vol. 70, No. 1., Apr., 1983, [pp. 41-55](http://www.stat.cmu.edu/~ryantibs/journalclub/rosenbaum_1983.pdf)
[^pearl1]: Judea Pearl; "CAUSALITY - Models, Reasoning and Inference"; 2nd Edition, 2009, [pp. 348-352](http://bayes.cs.ucla.edu/BOOK-09/ch11-3-5-final.pdf)
[^pearl2]: Judea Pearl; "CAUSALITY - Models, Reasoning and Inference"; 2nd Edition, 2009, [pp. 341-344](http://bayes.cs.ucla.edu/BOOK-09/ch11-3-2-final.pdf)
[^austin]: Peter C. Austin; "An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies"; Multivariate Behav Res. 2011 May; 46(3): [pp. 399–424](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/)

[nonparametric]: https://en.wikipedia.org/wiki/Nonparametric_statistics
[10things]: http://egap.org/methods-guides/10-things-you-need-know-about-causal-inference