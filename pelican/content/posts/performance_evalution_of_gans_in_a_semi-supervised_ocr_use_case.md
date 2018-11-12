---
title: Performance evaluation of GANs in a semi-supervised OCR use case
date: 2018-10-24 18:00
modified: 2018-10-24 18:00
category: talk
tags: machine-learning, python, GANs, semi-supervised
authors: Florian Wilhelm
status: published
summary: Even in the age of big data labelled data is a scarce resource in many machine learning use cases. We evaluate generative adversarial networks (GANs) at the task of extracting information from vehicle registrations under a varying amount of labelled data and compare the performance with supervised learning techniques. Using unlabelled data shows a significant improvement.
---

Online vehicle marketplaces are embracing artificial intelligence to ease the process of selling a vehicle on their platform. The tedious work of copying information from the vehicle registration document into some web form can be automated with the help of smart text spotting systems. The seller takes a picture of the document and the necessary information is extracted automatically.

We introduce the components of a text spotting system including the subtasks of object detection and character object recognition (OCR). In view of our use case, we elaborate on the challenges of OCR in documents with various distortions and artefacts which rule out off-the-shelve products for this task.

After an introduction of semi-supervised learning based on generative adversarial networks (GANs), we evaluate the performance gains of this method compared to supervised learning. More specifically, for a varying amount of labelled data the accuracy of a convolution neural network (CNN) is compared to a GAN which uses additionally unlabelled data during the training phase.

We conclude that GANs significantly outperform classical CNNs in use cases with a lack of labelled data. Regarding our use case of extracting information from vehicle registration documents, we show that our text spotting system easily exceeds an accuracy of 99.5% thus making it applicable in a real-world use case.

This talk was presented at [O'Reilly AI Conference 2018][] and [PyCon.de 2018 Karlsruhe][]. The slides are available on [SlideShare][].

{% youtube XniwzOCWi2c 800 500 %}

[PyCon.de 2018 Karlsruhe]: https://de.pycon.org/schedule/talks/performance-evaluation-of-gans-in-a-semi-supervised-ocr-use-case/
[O'Reilly AI Conference 2018]: https://conferences.oreilly.com/artificial-intelligence/ai-eu-2018/public/schedule/detail/70158
[SlideShare]: https://de.slideshare.net/FlorianWilhelm2/performance-evaluation-of-gans-in-a-semisupervised-ocr-use-case
