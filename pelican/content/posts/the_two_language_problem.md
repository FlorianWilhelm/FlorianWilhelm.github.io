---
title: The two language problem in Data Science
date: 2017-04-15 18:00
modified: 2017-06-17 18:00
category: post
tags: python, data science, production
authors: Florian Wilhelm
status: draft
---

python, data science, production


<figure>
<p align="center">
<img class="noZoom" src="/images/industry_vs_science.png" alt="Industry vs. Science">
<figcaption><strong>Figure:</strong> Executing <em>spark-submit</em> uploads our <em>activate_env.py</em> module and starts a Spark driver process. Thereafter, <em>activate_env.py</em> is executed within the driver and bootstraps our <em>venv</em> environment on the Spark driver as well as on the executors. Finally, <em>activate_env.py</em> relinquishes control to <em>my_pyspark_app</em>.</figcaption>
</p>
</figure>

[PMML]: http://dmg.org/pmml/v4-3/GeneralStructure.html
[PFA]: http://dmg.org/pfa/index.html
[ONNX]: https://onnx.ai/