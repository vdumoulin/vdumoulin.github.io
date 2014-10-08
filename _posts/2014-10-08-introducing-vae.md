---
layout: post-no-feature
title: Introducing the VAE framework in Pylearn2
description: "Combining VAEs with 'Not your grandfather's machine learning library'"
categories: articles
comments: true
published: true
date: 2014-10-08
---

_After quite some time spent on the pull request, I'm proud to announce that
the VAE model is now integrated in Pylearn2. In this post, I'll go over the
main features of the VAE framework and how to extend it. I will assume the
reader is familiar with the VAE model. If not, have a look at my [VAE demo
webpage]({{ site.url }}/articles/vae-demo) as well as the
[(Kingma and Welling)](http://arxiv.org/abs/1312.6114) and [(Rezende _et
al._)](http://arxiv.org/abs/1401.4082) papers._


# The model

A VAE come with three moving parts:

* the prior distribution \\(p\\_\\theta(\\mathbf{z})\\) on latent vector
  \\(\\mathbf{z}\\)
* the conditional distribution \\(p\\_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\)
  on observed vector \\(\\mathbf{x}\\)
* the approximate posterior distribution \\(q\\_\\phi(\\mathbf{z} \\mid
  \\mathbf{x})\\) on latent vector \\(\\mathbf{z}\\)

The parameters \\(\\phi\\) and \\(\\theta\\) are arbitrary functions of
\\(\\mathbf{x}\\) and \\(\\mathbf{z}\\) respectively.

The model is trained to minimize the expected reconstruction loss of
\\(\\mathbf{x}\\) under \\(\\q\\_\\phi(\\mathbf{z} \\mid \\mathbf{x})\\) and the
KL-divergence between the prior and posterior distributions on \\(\\mathbf{z}\\)
at the same time.

In order to backpropagate the gradient on the reconstruction loss through the
function mapping \\(\\mathbf{x}\\) to parameters \\(\\phi\\), the
reparametrization trick is used, which allows sampling from \\(\\mathbf{z}\\) by
considering it as a deterministic function of \\(\\mathbf{x}\\) and some noise
\\(\\mathbf{\\epsilon}\\).

# The VAE framework

## Overview

### `pylearn2.models.vae.VAE`

The VAE model is represented in Pylearn2 by the `VAE` class. It is responsible
for high-level computation, such as computing the log-likelihood lower bound
known as the _VAE criterion_ or an importance sampling estimate of the
log-likelihood, both using the reparametrization trick, and acts as the
interface between the model and other parts of Pylearn2 such as the
VAE-associated costs (`pylearn2.costs.vae.VAECriterion` or
`pylearn2.costs.vae.ImportanceSamplingCriterion`) or the user.

It delegates much of its functionality to three objects:

* `pylearn2.models.vae.conditional.Conditional`
* `pylearn2.models.vae.prior.Prior`
* `pylearn2.models.vae.kl.KLIntegrator`

### `pylearn2.models.vae.conditional.Conditional`

`Conditional` is used to represent conditional distributions in the VAE
framework (namely \\(p\\_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\)
and \\(q\\_\\phi(\\mathbf{z} \\mid \\mathbf{x})\\)). It is responsible for
mapping its input to parameters of the conditional distribution it represents,
sampling from the conditional distribution with or without the reparametrization
trick and computing the conditional log-likelihood of the distribution it
represents given some samples.

Internally, the mapping from input to parameters of the conditional distribution
is done via an `MLP` instance. This allows users familiar with the MLP framework
to easily switch between different architectures for the encoding and
decoding networks.

### `pylearn2.models.vae.prior.Prior`

`Prior` is used to represent the prior distribution on \\(\\mathbf{z}\\) in the
VAE framework. It is responsible for sampling from the prior distribution and
computing the log-likelihood of the distribution it represents given some
samples.

### `pylearn2.models.vae.kl.KLIntegrator`

Some combination of prior and posterior distributions (e.g. a gaussian prior
with diagonal covariance matrix and a gaussian posterior with diagonal
covariance matrix) allow the analytic integration of the KL term in the VAE
criterion. `KLIntegrator` is responsible for representing this analytic
expression and optionally representing it as a sum of elementwise KL terms, when
such decomposition is allowed by the choice of prior and posterior
distributions.

This allows the VAE framework to be more modular: otherwise, the analytical
computation of the KL term would require that the prior and the posterior
distributions are defined in the same class.

Subclasses of `KLIntegrator` define one subclass of `Prior` and one subclass of
`Conditional` as class attributes and can carry out the analytic computation of
the KL term **for these two subclasses only**. The `pylearn2.models.vae.kl`
module also contains a method which can automatically infer which subclass of
`KLIntegrator` is compatible with the current choice of prior and posterior, and
`VAE` automatically falls back to a stochastic approximation of the KL term when
the analytical computation is not possible.

### `pylearn2.costs.vae.{VAE,ImportanceSampling}Criterion`

Two `Cost` objects are compatible with the VAE framework: `VAECriterion` and
`ImportanceSamplingCriterion`. `VAECriterion` represent the VAE criterion as
defined in [(Kingma and Welling)](http://arxiv.org/abs/1312.6114), while
`ImportanceSamplingCriterion` defines a cost based on the importance sampling
approximation of the marginal log-likelihood which allows backpropagation
through \\(\\q\\_\\phi(\\mathbf{z} \\mid \\mathbf{x})\\) via the
reparametrization trick.
