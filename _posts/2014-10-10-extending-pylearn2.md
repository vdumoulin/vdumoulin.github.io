---
layout: post-no-feature
title: Your models in Pylearn2
description: "A tutorial on the minimal effort required to develop a new model in Pylearn2"
categories: articles
comments: true
published: false
date: 2014-10-10
---

# Who should read this

This tutorial is designed for pretty much anyone working with Theano who's tired
of writing the same old boilerplate code over and over again. You have SGD
implementations scattered in pretty much every experiment file? This tutorial is
for you.

In my opinion, Pylearn2 is great for two things:

* It allows you to experiment with new ideas without much implementation
  overhead. The library was built to be modular, and it aims to be usable
  without an extensive knowledge of the codebase. Writing a new model from
  scratch is usually pretty fast once you know what to do and where to look.
* It has an interface (YAML) that allows to decouple implementation from
  experimental choices, which allows experiments to be constructed in a light
  and readable fashion.

# Toy example: logistic regression

You just read the
[deeplearning.net tutorial](http://www.deeplearning.net/tutorial/logreg.html#logreg)
on how to do logistic regression on MNIST digits using Theano.
