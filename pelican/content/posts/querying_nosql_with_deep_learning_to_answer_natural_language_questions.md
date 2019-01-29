---
title: Querying NoSQL with Deep Learning to Answer Natural Language Questions
date: 2019-01-29 11:30
modified: 2019-01-29 11:30
category: talk
tags: data science, nlp
authors: Florian Wilhelm
status: published
---

Almost all of today’s knowledge is stored in databases and thus can only be accessed with the help of domain specific query languages, strongly limiting the number of people which can access the data. In our work, we demonstrate an end-to-end trainable question answering (QA) system that allows a user to query an external NoSQL database by using natural language. A major challenge of such a system is the non-differentiability of database operations which we overcome by applying policy-based reinforcement learning. We evaluate our approach on Facebook’s bAbI Movie Dialog dataset and achieve a competitive score of 84.2% compared to several benchmark models. We conclude that our approach excels with regard to real-world scenarios where knowledge resides in external databases and intermediate labels are too costly to gather for non-end-to-end trainable QA systems.

Our paper *Querying NoSQL with Deep Learning to Answer Natural Language Questions* was presented at [IAAI-19] and can be downloaded on [aaai.org]. There is also a simplified blog post about the same topic on the [inovex blog] by my colleague Sebastian Blank.

[IAAI-19]: https://aaai.org/Conferences/AAAI-19/
[aaai.org]: https://aaai.org/Papers/AAAI/2019/IAAI-BlankS.88.pdf
[inovex blog]: https://www.inovex.de/blog/seqpolicynet-nlp-elasticsearch/
