---
layout: post-no-feature
title: Introducing the VAE framework in Pylearn2
description: "Combining variational autoencoders with 'Not your grandfather's machine learning library'"
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
[(Kingma and Welling)](http://arxiv.org/abs/1312.6114) and [(Rezende et
al.)](http://arxiv.org/abs/1401.4082) papers._


# The model

A VAE comes with three moving parts:

* the prior distribution \\(p\_\\theta(\\mathbf{z})\\) on latent vector
  \\(\\mathbf{z}\\)
* the conditional distribution \\(p\_\\theta(\\mathbf{x} \\mid \\mathbf{z})\\)
  on observed vector \\(\\mathbf{x}\\)
* the approximate posterior distribution \\(q\_\\phi(\\mathbf{z} \\mid
  \\mathbf{x})\\) on latent vector \\(\\mathbf{z}\\)

The parameters \\(\\phi\\) and \\(\\theta\\) are arbitrary functions of
\\(\\mathbf{x}\\) and \\(\\mathbf{z}\\) respectively.

The model is trained to minimize the expected reconstruction loss of
\\(\\mathbf{x}\\) under \\(q\_\\phi(\\mathbf{z} \\mid \\mathbf{x})\\) and the
KL-divergence between the prior and posterior distributions on \\(\\mathbf{z}\\)
at the same time.

In order to backpropagate the gradient on the reconstruction loss through the
function mapping \\(\\mathbf{x}\\) to parameters \\(\\phi\\), the
reparametrization trick is used, which allows sampling from \\(\\mathbf{z}\\) by
considering it as a deterministic function of \\(\\mathbf{x}\\) and some noise
\\(\\mathbf{\\epsilon}\\).

# The VAE framework

## Overview

### pylearn2.models.vae.VAE

The VAE model is represented in Pylearn2 by the `VAE` class. It is responsible
for high-level computation, such as computing the log-likelihood lower bound
or an importance sampling estimate of the log-likelihood, and acts as the
interface between the model and other parts of Pylearn2.

It delegates much of its functionality to three objects:

* `pylearn2.models.vae.conditional.Conditional`
* `pylearn2.models.vae.prior.Prior`
* `pylearn2.models.vae.kl.KLIntegrator`

### pylearn2.models.vae.conditional.Conditional

`Conditional` is used to represent conditional distributions in the VAE
framework (namely the approximate posterior on \\(\\mathbf{z}\\) and the
conditional on \\(\\mathbf{x}\\)). It is responsible for mapping its input to
parameters of the conditional distribution it represents, sampling from the
conditional distribution with or without the reparametrization trick and
computing the conditional log-likelihood of the distribution it represents given
some samples.

Internally, the mapping from input to parameters of the conditional distribution
is done via an `MLP` instance. This allows users familiar with the MLP framework
to easily switch between different architectures for the encoding and
decoding networks.

### pylearn2.models.vae.prior.Prior

`Prior` is used to represent the prior distribution on \\(\\mathbf{z}\\) in the
VAE framework. It is responsible for sampling from the prior distribution and
computing the log-likelihood of the distribution it represents given some
samples.

### pylearn2.models.vae.kl.KLIntegrator

Some combinations of prior and posterior distributions (e.g. a gaussian prior
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

### pylearn2.costs.vae.{VAE,ImportanceSampling}Criterion

Two `Cost` objects are compatible with the VAE framework: `VAECriterion` and
`ImportanceSamplingCriterion`. `VAECriterion` represent the VAE criterion as
defined in [(Kingma and Welling)](http://arxiv.org/abs/1312.6114), while
`ImportanceSamplingCriterion` defines a cost based on the importance sampling
approximation of the marginal log-likelihood which allows backpropagation
through \\(q\_\\phi(\\mathbf{z} \\mid \\mathbf{x})\\) via the
reparametrization trick.

## Using the framework

### Training the example model

Let's go over a small example on how to train a VAE on MNIST digits.

In this example I'll be using
[Salakhutdinov and Murray](http://www.mit.edu/~rsalakhu/papers/dbn_ais.pdf)'s
binarized version of the MNIST dataset. Make sure the `PYLEARN2_DATA_PATH`
environment variable is set properly, and download the data using 

{% highlight bash %}
python pylearn2/scripts/datasets/download_binarized_mnist.py
{% endhighlight %}

Here's the YAML file we'll be using for the example:

{% highlight text %}
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.binarized_mnist.BinarizedMNIST {
        which_set: 'train',
    },
    model: !obj:pylearn2.models.vae.VAE {
        nvis: &nvis 784,
        nhid: &nhid 100,
        prior: !obj:pylearn2.models.vae.prior.DiagonalGaussianPrior {},
        conditional: !obj:pylearn2.models.vae.conditional.BernoulliVector {
            name: 'conditional',
            mlp: !obj:pylearn2.models.mlp.MLP {
                layers: [
                    !obj:pylearn2.models.mlp.RectifiedLinear {
                        layer_name: 'h_1',
                        dim: 200,
                        irange: 0.001,
                    },
                    !obj:pylearn2.models.mlp.RectifiedLinear {
                        layer_name: 'h_2',
                        dim: 200,
                        irange: 0.001,
                    },
                ],
            },
        },
        posterior: !obj:pylearn2.models.vae.conditional.DiagonalGaussian {
            name: 'posterior',
            mlp: !obj:pylearn2.models.mlp.MLP {
                layers: [
                    !obj:pylearn2.models.mlp.RectifiedLinear {
                        layer_name: 'h_1',
                        dim: 200,
                        irange: 0.001,
                    },
                ],
            },
        },
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 200,
        learning_rate: 1e-3,
        learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
            init_momentum: 0.05,
        },
        monitoring_dataset: {
            'train' : *train,
            'valid' : !obj:pylearn2.datasets.binarized_mnist.BinarizedMNIST {
                which_set: 'valid',
            },
            'test' : !obj:pylearn2.datasets.binarized_mnist.BinarizedMNIST {
                which_set: 'test',
            },
        },
        cost: !obj:pylearn2.costs.vae.VAECriterion {
            num_samples: 1,
        },
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs: 150
        },
        update_callbacks: [
            !obj:pylearn2.training_algorithms.sgd.ExponentialDecay {
                decay_factor: 1.00005,
                min_lr:       0.00001
            },
        ],
    },
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
            channel_name: 'valid_objective',
            save_path: "${PYLEARN2_TRAIN_FILE_FULL_STEM}_best.pkl",
        },
        !obj:pylearn2.training_algorithms.learning_rule.MomentumAdjustor {
            final_momentum: .95,
            start: 5,
            saturate: 6
        },
    ],
}
{% endhighlight %}

Give it a try:

{% highlight bash %}
# Assuming your YAML file is called ${YOUR_FILE_NAME}.yaml
python pylearn2/scripts/train.py ${YOUR_FILE_NAME}.yaml
{% endhighlight %}

This might take a while, but you can accelerate things using the appropriate
Theano flags to train using a GPU.

You'll see a couple things being monitored while the model learns:

* **{train,valid,test}_objective** tracks the value of the VAE criterion for
  the training, validation and test sets.
* **{train,valid,test}_expectation_term** tracks the expected reconstruction
  of the input under the posterior distribution averaged across the training,
  validation and test sets.
* **{train,valid,test}_kl_divergence_term** tracks the KL-divergence between
  the posterior and the prior distributions averaged across the training,
  validation and test sets.

### Evaluating the trained model

__N.B.: At the moment of writing this post, there are no scripts in Pylearn2 to
evaluate trained models by looking at samples or computing an approximate NLL.
This is definitely something that will be included in the future, but for the
moment here are some workarounds taken from my personal scripts.__

When training is complete, you can look at samples from the model by running the
following bit of Python code:

{% highlight python %}
import argparse
import numpy
import theano
from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.utils import serial


def show(vis_batch, dataset, mapback, pv, rows, cols):
    # Random selection of a subset of vis_batch to display
    index_array = numpy.arange(vis_batch.shape[0])
    vis_batch_subset = vis_batch[index_array[:rows * cols]]

    display_batch = dataset.adjust_for_viewer(vis_batch_subset)
    if display_batch.ndim == 2:
        display_batch = dataset.get_topological_view(display_batch)
    display_batch = display_batch.transpose(tuple(
        dataset.X_topo_space.axes.index(axis) for axis in ('b', 0, 1, 'c')
    ))
    if mapback:
        design_vis_batch = vis_batch_subset
        if design_vis_batch.ndim != 2:
            design_vis_batch = dataset.get_design_matrix(design_vis_batch)
        mapped_batch_design = dataset.mapback_for_viewer(design_vis_batch)
        mapped_batch = dataset.get_topological_view(mapped_batch_design)
    for i in xrange(rows):
        row_start = cols * i
        for j in xrange(cols):
            pv.add_patch(display_batch[row_start+j, :, :, :],
                         rescale=False)
            if mapback:
                pv.add_patch(mapped_batch[row_start+j, :, :, :],
                             rescale=False)
    pv.show()


def show_samples(model):
    num_samples = 100
    rows = 10
    cols = 10

    dataset_yaml_src = model.dataset_yaml_src
    dataset = yaml_parse.load(dataset_yaml_src)

    vis_batch = dataset.get_batch_topo(num_samples)
    rval = tuple(vis_batch.shape[dataset.X_topo_space.axes.index(axis)]
                 for axis in ('b', 0, 1, 'c'))
    _, patch_rows, patch_cols, channels = rval
    mapback = hasattr(dataset, 'mapback_for_viewer')
    pv = PatchViewer((rows, cols*(1+mapback)),
                     (patch_rows, patch_cols),
                     is_color=(channels == 3))

    samples, expectations = model.sample(num_samples)
    f = theano.function(inputs=[], outputs=expectations)
    samples_batch = f()
    show(samples_batch, dataset, mapback, pv, rows, cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to the pickled model")
    args = parser.parse_args()

    model_path = args.model_path
    model = serial.load(model_path)

    show_samples(model)
{% endhighlight %}

Look at samples by typing

{% highlight bash %}
# Assuming your YAML file is called ${YOUR_FILE_NAME}.yaml and your sampling
# script is named ${SAMPLING_SCRIPT}.py
python ${SAMPLING_SCRIPT}.py ${YOUR_FILE_NAME}.yaml
{% endhighlight %}

You can also make use of `VAE.log_likelihood_approximation` to compute
approximate NLL performance measures of the trained model:

{% highlight python %}
import argparse
import numpy
import theano
import theano.tensor as T
from pylearn2.config import yaml_parse
from pylearn2.utils import as_floatX, serial


def print_nll(model):
    dataset_yaml_src = model.dataset_yaml_src
    train_set = yaml_parse.load(dataset_yaml_src)
    valid_set = yaml_parse.load(dataset_yaml_src.replace("train", "valid"))
    test_set = yaml_parse.load(dataset_yaml_src.replace("train", "test"))

    X = T.matrix('X')
    importance_sampling_nll = -model.log_likelihood_approximation(
        X=X,
        num_samples=20,
    ).mean()
    f = theano.function(inputs=[X], outputs=importance_sampling_nll)

    batch_size = 100

    # Train
    train_importance_sampling_nll_list = []
    numpy_X = as_floatX(train_set.get_design_matrix())
    for i in xrange(numpy_X.shape[0] / batch_size):
        numpy_X_batch = numpy_X[batch_size * i: batch_size * (i + 1)]
        train_importance_sampling_nll_list.append(f(numpy_X_batch))
    # Valid
    valid_importance_sampling_nll_list = []
    numpy_X = as_floatX(valid_set.get_design_matrix())
    for i in xrange(numpy_X.shape[0] / batch_size):
        numpy_X_batch = numpy_X[batch_size * i: batch_size * (i + 1)]
        valid_importance_sampling_nll_list.append(f(numpy_X_batch))
    # Test
    test_importance_sampling_nll_list = []
    numpy_X = as_floatX(test_set.get_design_matrix())
    for i in xrange(numpy_X.shape[0] / batch_size):
        numpy_X_batch = numpy_X[batch_size * i: batch_size * (i + 1)]
        test_importance_sampling_nll_list.append(f(numpy_X_batch))

    print "Train NLL approximation: " + \
          str(numpy.mean(train_importance_sampling_nll_list))
    print "Valid NLL approximation: " + \
          str(numpy.mean(valid_importance_sampling_nll_list))
    print " Test NLL approximation: " + \
          str(numpy.mean(test_importance_sampling_nll_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to the pickled model")
    args = parser.parse_args()

    model_path = args.model_path
    model = serial.load(model_path)

    print_nll(model)
{% endhighlight %}

All you have to do is type

{% highlight bash %}
# Assuming your YAML file is called ${YOUR_FILE_NAME}.yaml and your sampling
# script is named ${NLL_SCRIPT}.py
python ${NLL_SCRIPT}.py ${YOUR_FILE_NAME}.yaml
{% endhighlight %}

### More details

Let's concentrate on this part of the YAML file:

{% highlight text %}
model: !obj:pylearn2.models.vae.VAE {
    nvis: &nvis 784,
    nhid: &nhid 100,
    prior: !obj:pylearn2.models.vae.prior.DiagonalGaussianPrior {},
    conditional: !obj:pylearn2.models.vae.conditional.BernoulliVector {
        name: 'conditional',
        mlp: !obj:pylearn2.models.mlp.MLP {
            layers: [
                !obj:pylearn2.models.mlp.RectifiedLinear {
                    layer_name: 'h_1',
                    dim: 200,
                    irange: 0.001,
                },
                !obj:pylearn2.models.mlp.RectifiedLinear {
                    layer_name: 'h_2',
                    dim: 200,
                    irange: 0.001,
                },
            ],
        },
    },
    posterior: !obj:pylearn2.models.vae.conditional.DiagonalGaussian {
        name: 'posterior',
        mlp: !obj:pylearn2.models.mlp.MLP {
            layers: [
                !obj:pylearn2.models.mlp.RectifiedLinear {
                    layer_name: 'h_1',
                    dim: 200,
                    irange: 0.001,
                },
            ],
        },
    },
}
{% endhighlight %}

We define the dimensionality of \\(\\mathbf{x}\\) through `nvis` and the
dimensionality of \\(\\mathbf{z}\\) through `nhid`.

At a high level, the form of the prior, posterior and conditional distributions
is selected through the choice of which subclasses to instantiate. Here we chose
a gaussian prior with diagonal covariance matrix, a gaussian posterior with
diagonal covariance matrix and a product of bernoulli as the conditional for
\\(\\mathbf{x}\\).

Note that we did not explicitly tell the model how to integrate the KL: it was
able to find it on its own by calling
`pylearn2.models.vae.kl.find_integrator_for`, which searched
`pylearn2.models.vae.kl` for a match and returned an instance of
`DiagonalGaussianPriorPosteriorKL`. If you were to explicitly tell the model how
to integrate the KL term (for instance, if you have defined a new prior and a
new `KLIntegrator` subclass to go with it), you would need to add

{% highlight text %}
kl_integrator: !obj:pylearn2.models.vae.kl.DiagonalGaussianPriorPosteriorKL {}
{% endhighlight %}

as a parameter to `VAE`'s constructor.

`Conditional` instances (passed as `conditional` and `posterior` parameters)
need a name upon instantiation. This is to avoid key collisions in the
monitoring channels.

They're also given **nested** `MLP` instance. Why this is needed will become
clear soon. Notice how the last layers' dimensionality does
not match either `nhid` or `nvis`. This is because they represent the last
hidden representation from which the conditional parameters will be computed.
You did not have to specify the layer mapping the last hidden representation to
the conditional parameters because it was automatically inferred: after
everything is instantiated, `VAE` calls `initialize_parameters` on `prior`,
`conditional` and `posterior` and gives them relevant information about their
input and output spaces. At that point, `Conditional` has enough information to
infer how the last layer should look like. It calls its private
`_get_default_output_layer` method, which returns a sane default output layer,
and adds it to its MLP's list of layers. This is why a nested MLP is required:
this allows `Conditional` to delay the initialization of the MLP's input space
in order to add a layer to it in a clean fashion.

Naturally, you may want to decide on your own how parameters should be computed
based on the last hidden representation. This can be done through
`Conditional`'s `output_layer_required` constructor parameter. It is set to
`True` by default, but you can switch it off and explicitly put the last layer
in the MLP. For instance, you could decide that the gaussian posterior's
\\(\\log \\sigma\\) should not be too big or too small and want to force it to
be between -1 and 1 by using a _tanh_ non-linearity. It can be done like so:

{% highlight text %}
posterior: !obj:pylearn2.models.vae.conditional.DiagonalGaussian {
    name: 'posterior',
    output_layer_required: 0,
    mlp: !obj:pylearn2.models.mlp.MLP {
        layers: [
            !obj:pylearn2.models.mlp.RectifiedLinear {
                layer_name: 'h_1',
                dim: 200,
                irange: 0.001,
            },
            !obj:pylearn2.models.mlp.CompositeLayer {
                layer_name: 'phi',
                layers: [
                    !obj:pylearn2.models.mlp.Linear {
                        layer_name: 'mu',
                        dim: *nhid,
                        irange: 0.001,
                    },
                    !obj:pylearn2.models.mlp.Tanh {
                        layer_name: 'log_sigma',
                        dim: *nhid,
                        irange: 0.001,
                    },
                ],
            },
        ],
    },
}
{% endhighlight %}

There are safeguards in place to make sure your code won't crash without
explanation in the case of a mistake: `Conditional` will verify that the custom
output layer you put in MLP has the same output space as what it expects and
raise an exception otherwise. Every `Conditional` subclass need to define how
should the conditional parameters look like through a private
`_get_required_mlp_output_space` method, and you should make sure that your
custom output layer has the right output space by looking at the code. Moreover,
you should have a look at the subclass' `_get_default_output_space`
implementation to see what is the nature and order of the conditional parameters
being computed.

## Extending the VAE framework

_This post will be updated soon with more information on how to write your own
subclasses of `Prior`, `Conditional` and `KLIntegrator`._
