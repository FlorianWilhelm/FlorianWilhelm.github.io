---
title: Declarative Thinking and Programming
date: 2016-07-22 11:15
modified: 2016-07-22 11:15
category: post
tags: python, programming
authors: Florian Wilhelm
status: draft
summary:
---

Before we actually dive into this topic, imagine the following: 
You just moved to a new place and the time is ripe for a little house-warming dinner with your best friends Alice and Bob.
Since Alice is really tech-savvy you just send her a digital invitation with date, time and of course your new address
that she can add to her calendar with a single click. With good old Bob is a bit more difficult, he is having a real 
struggle with modern IT. That's why you decide to send him an e-mail not only including the time and location but also 
a suggestion which train to take to your city, details about the trams, stops, the right street and so. 

So let's discuss the differences of how you interacted with Alice and Bob but let's first recall what your task was. 
Your task was to organize a house-warming dinner with your friends. Therefore, your level of abstraction for that 
particular task should be on an organizing level. By sending the invitation message to Alice you *declared* that you want
her to be at your place at a certain time not caring about how she is actually gonna accomplish this. With Bob it's 
a completely different story. You not only declared your intentions but also provided lots of instructions on how to
accomplish these. By doing so you left the level of organizing a dinner and went down to the task of planning a journey
and this is not even the worst part. You also made a lot of error-prone assumptions that for instance Bob is gonna 
take the train to get to your city and not by some online ridesharing community which might have been a lot
cheaper and faster.

If we now transfer this example to programming we would describe the way of interaction with Alice as *declarative* and
the interaction with Bob as *imperative*. So in a nutshell, *declarative* programming is writing code in a way that
describes *what* you want to do and not *how* you want to do it. Sounds simple enough, right? But actually the difference
between *what* and *how* is often not that clear. Is telling someone to take a particular train always telling someone
how to do something and therefore imperative? No, actually not, it really depends on your actual task and the level of
abstraction that comes with it. 

Let's say we want to do some linear algebra, in particular we want to sum up two n-dimensional
vectors $a$ and $b$. Since we are in the domain of linear algebra using some CAS, we would expect to be able
to just declare what we want $c = a + b$. Calculating $c$ with the help of a loop would clearly by imperative in the
given context. The downsides of using a loop for this are manifold. Firstly, we would define an implicit order of which
elements to sum up first. This removes the possibility of our CAS software to choose a native [SIMD][] CPU operation
due to our over-specification of how to do it. Secondly, our code becomes much less readable and we are violating the
[single-level of abstraction principle][] which is highly connected to declarative programming and thinking. 
If, on the other hand though, our task is to solve a given linear optimization problem, starting to implement an 
Simplex algorithm on our own in order to solve it can be considered imperative. 
 
Having realized the importance of the level of abstraction resp. the domain our task lives in, the obvious question is 
the following one: How do languages like SQL and Prolog fit into our picture since they are always stated as being declarative? 
The first thing to note actually is that both languages are domain-specific languages (DSLs). 
 
[SIMD]: https://en.wikipedia.org/wiki/SIMD 
[single-level of abstraction principle]: http://principles-wiki.net/principles:single_level_of_abstraction
[]: http://www.zeit.de/2016/52/spiele-logelei-52
[topological sorting]: https://en.wikipedia.org/wiki/Topological_sorting