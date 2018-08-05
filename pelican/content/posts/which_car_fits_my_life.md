---
title: “Which car fits my life?” - mobile.de’s approach to recommendations
date: 2017-07-02 18:00
modified: 2017-07-02 18:00
category: talk
tags: machine-learning, python, recommender systems
authors: Florian Wilhelm, Arnab Dutta
status: published
---

At mobile.de, Germany’s biggest car marketplace, a dedicated team of data engineers and scientists, supported by the IT project house inovex is responsible for creating intelligent data products. Driven by our company slogan “Find the car that fits your life”, we focus on personalised recommendations to address several user needs. Thereby we improve customer experience during browsing as well as finding the perfect offering. In an introduction to recommendation systems, we briefly mention the traditional approaches for recommendation engines, thereby motivating the need for sophisticated approaches. In particular, we explain the different concepts including collaborative and content-based filtering as well as hybrid approaches and general matrix factorisation methods. This is followed by a deep dive into the implementation and architecture at mobile.de that comprises ElasticSearch, Cassandra and Mahout. We explain how Python and Java is used simultaneously to create and serve recommendations.

By presenting our car-model recommender that suggests similar car models of different brands as a concrete use-case, we reiterate on key-aspects during modelling and implementation. In particular, we present a matrix factorisation library that we used and share our experiences with it. We conclude by a brief demonstration of our results and discuss the improvements we achieved in terms of key performance indicators. Furthermore, we use our use case to exemplify the usage of deep learning for recommendations, comparing it with other traditional approaches and hence providing a brief account of the future of recommendation engines.

This talk was presented at [PyData 2017 Berlin][] and [code.talks 2017][]. The slides are available on [SlideShare][].

{% youtube v7MBunqwBSY 800 500 %}

[code.talks 2017]: https://www.codetalks.de/de/2017/programm/which-car-fits-my-life-mobile-de-s-approach-to-recommendations
[PyData 2017 Berlin]: https://pydata.org/berlin2017/schedule/presentation/33/
[SlideShare]: https://www.slideshare.net/FlorianWilhelm2/which-car-fits-my-life-pydata-berlin-2017
