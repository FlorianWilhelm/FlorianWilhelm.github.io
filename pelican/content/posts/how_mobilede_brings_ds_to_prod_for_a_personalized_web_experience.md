---
title: How mobile.de brings Data Science to Production for a Personalized Web Experience
date: 2018-07-07 18:00
modified: 2018-07-07 18:00
category: talk
tags: python, data science, production, spark, big data
authors: Florian Wilhelm, Markus Schüler
status: published
---


As Germany's biggest online car marketplace, mobile.de provides a personalized web experience. Our Data Team leverages the interactions of our users to infer their preferences. For this tasks we often apply Python and Spark to wrangle massive amounts of data. In this talk, we are going to present our personalization use-cases as well as the application of PySpark in production.

After a short introduction we will present various data use cases that where tackled by mobile.de, Germany's largest vehicle marketplace online, in the last two years. In a combined endeavour inovex, an IT project house with a strong focus on digitalization, has supported mobile.de on this voyage.

Personalized web experience is a commonly used term in e-commerce that is hard to grasp. Thus, we illustrate how mobile.de understands personalized web experience and outline its features as well as opportunities. In more detail, we will elaborate on the Bayesian framework that we use to approximate user preferences. Furthermore, we discuss the modelling of user intent and how this can be used to understand the buyer's journey.

Besides the data science and modelling aspects of the use cases we will also dive into the technical details and how we solved them with the help of Python and Spark (PySpark). In more detail we will address the implementation of efficient User-Defined-(Aggregation)-Functions with Pandas in PySpark as well as the management of isolated environments and dependencies with PySpark.

We will conclude our talk with the benefits of a personalized web experience for the users of mobile.de which was achieved with the help of Python and PySpark in production.


This talk was presented at [PyData 2018 Berlin][] together with my colleague Markus Schüler from [mobile.de][] and the slides are available on [SlideShare][].

{% youtube fFiI6HABW-0 800 500 %}

[PyData 2018 Berlin]: https://pydata.org/berlin2018/schedule/presentation/59/
[SlideShare]: https://www.slideshare.net/FlorianWilhelm2/how-mobilede-brings-data-science-to-production-for-a-personalized-web-experience
[mobile.de]: https://www.mobile.de/
