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
implementations scattered in pretty much every experiment file? Pylearn2 looks
attractive but you think porting your Theano code to it is too much of an
investment? This tutorial is for you.

In my opinion, Pylearn2 is great for two things:

* It allows you to experiment with new ideas without much implementation
  overhead. The library was built to be modular, and it aims to be usable
  without an extensive knowledge of the codebase. Writing a new model from
  scratch is usually pretty fast once you know what to do and where to look.
* It has an interface (YAML) that allows to decouple implementation from
  experimental choices, which allows experiments to be constructed in a light
  and readable fashion.

Obviously, there is always a trade-off between being user-friendly and being
flexible, and Pylearn2 is no exception. For instance, users looking for a way to
work with sequential data might have a harder time getting started (although
this is something that's being worked on).

In this post, I'll assume that you have built a regression or classification
model with Theano and that the data it is trained on can be cast into two
matrices, one for training examples and one for training targets. People with
other use cases may need to work a little more (e.g. by figuring out how to put
their data inside Pylearn2), but I think the use case discussed here contains
useful information for anyone interested in porting a model to Pylearn2.

# How I work with Pylearn2

I do my research exclusively using Pylearn2, but that doesn't mean I use
or know everything in Pylearn2. In fact, I prototype new models in a very
Theano-like fashion: I write my model as a big monolithic block of hard coded
Theano expressions, and I wrap that up in the minimal amount of code necessary
to be able to plug my model in Pylearn2. **This bare minimum is what I intend to
teach here.**

Sure, every little change to the model is a pain, but it works, right? As I
explore new ideas and change the code, I gradually make it more flexible:
a hard coded input dimension gets factored out as a constructor argument,
functions being composed are separated into layers, etc.

The [VAE framework]({{ site.url }}/articles/introducing-vae) didn't start out
like it is now: all I did is reproduce what Joost van Amersfoort wrote in Theano
(see his code [here](https://github.com/y0ast/Variational-Autoencoder/blob/master/Theano/VariationalAutoencoder.py))
to reproduce the experiments in
[(Kingma and Welling)](http://arxiv.org/abs/1312.6114). Over time, I made the
code more modular and started reusing elements of the MLP framework, and at some
point it got to a state where I felt that it could be useful for other people.

I guess what I'm trying to convey here is that **it's alright to stick to the
bare minimum when developing a model for Pylearn2**. Your code probably won't
satisfy any other use cases than yours, but this is something that you can
change gradually as you go. There's no need to make things any more complicated
than they should be when you start.

# The bare minimum

Let's look at that _bare minimum_. It involves writing exactly two subclasses:

* One subclass of `pylearn2.costs.cost.Cost`
* One subclass of `pylearn2.models.model.Model`

No more than that? Nope. That's it! Let's have a look.

## It all starts with a cost expression

In the scenario I'm describing, your model maps an input to an output, the
output is compared with some ground truth using some measure of dissimilarity,
and the parameters of the model are changed to reduce this measure using
gradient information.

It is therefore natural that the object that interfaces between the model and
the training algorithm represents a cost. The base class for this object is
`pylearn2.costs.cost.Cost` and does three main things:

* It describes what data it needs to perform its duty and how it should be
* formatted.
* It computes the cost expression by feeding the input to the model and
  receiving its output.
* It differentiates the cost expression with respect to the model parameter and
  returns the gradients to the training algorithm.

What's nice about `Cost` is if you follow the guidelines I'm about to describe,
you only have to worry about the cost expression; the gradient part is all
handled by the `Cost` base class, and a very useful `DefaultDataSpecsMixin`
mixin subclass is defined to handle the data description part (more about that
when we look at the `Model` subclass).

Let's look at how the subclass should look like:

{% highlight python %}
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin


class MyCostSubclass(Cost, DefaultDataSpecsMixin):
    # Here it is assumed that we are doing supervised learning
    supervised = True

    def expr(model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)
        
        inputs, targets = data
        outputs = model.some_method_for_outputs(inputs)
        loss = # some loss measure involving outputs and targets
        return loss
{% endhighlight %}

The `supervised` class attribute is used by `DefaultDataSpecsMixin` to know how
to specify the data requirements. If it is set to `True`, the cost will expect
to receive inputs and targets, and if it is set to `False`, the cost will expect
to recive inputs only. In the example, it is assumed that we are doing
supervised learning, so we set `supervised` to `True`.

The first two lines of `expr` do some basic input checking and should always be
included at the beginning of your `expr` method. Without going too much into
details, `space.validate(data)` will make sure that the data you get is the data
you requested (e.g. if you do supervised learning you need an input tensor
variable and a target tensor variable). How "what you need" is decided will be
covered when we look at the `Model` subclass.

In that case, `data` is a tuple containing the inputs as the first element and
the targets as the second element (once again, bear with me everything isn't
completely clear for the moment, you'll get to understand soon enough).

We then get the model output by calling its `some_method_for_outputs` method,
whose name and behaviour is really for you to decide, as long as you `Cost`
subclass knows which method to call on the model.

Finally, we compute some loss measure on `outputs` and `targets` and return that
as the cost expression.

Note that things don't have to be _exactly_ like this. For instance, you could
want the model to have a method that takes inputs and targets as argument and
returns the loss directly, and that would be perfectly fine. All you need is
some way to make your `Model` and `Cost` subclasses to work together to produce
a cost expression in the end.

## Defining the model

Now it's time to make things more concrete by writing the model itself. The
model will be a subclass of `pylearn2.models.model.Model`, which is responsible
for the following:

* Defining what its parameters are
* Defining what its data requirements are
* Doing something with the input to produce an output

Like for `Cost`, the `Model` base class does lots of useful things on its own,
provided you set the appropriate instance attributes. Let's have a look at a
subclass example:

{% highlight python %}
from pylearn2.models.model import Model

class MyModelSubclass(Model):
    def __init__(self, *args, **kwargs):
        super(MyModelSubclass, self).__init__()

        # Some parameter initialization using *args and **kwargs
        # ...
        self._params = [
            # List of all the model parameters
        ]

        self.input_space = # Some `pylearn2.space.Space` subclass
        # This one is necessary only for supervised learning
        self.output_space = # Some `pylearn2.space.Space` subclass

    def some_method_for_outputs(inputs):
        # Some computation involving the inputs
{% endhighlight %}

The first thing you should do if you're overriding the constructor is call the
the superclass' constructor. Pylearn2 checks for that and will scold you if you
don't.

You should then initialize you model parameters **as shared variables**:
Pylearn2 will build an updates dictionary for your model variables using
gradients returned by your cost. _Protip: the `pylearn2.utils.sharedX` method
initializes a shared variable with the value and an optional name you provide.
This allows your code to be GPU-compatible without putting too much thought into
it._ For instance, a weights matrix can be initialized this way:

{% highlight python %}
import numpy
from pylearn2.utils import sharedX

self.W = sharedX(numpy.random.normal(size=(size1, size2)), 'W')
{% endhighlight %}

Put all your parameters in a list as the `_params` instance attribute. The
`Model` superclass defines a `get_params` method which returns `self._params`
for you, and that is method that is called to get the model parameters when
`Cost` is computing the gradients.

Your `Model` subclass should also describe the data format it expects as input
(`self.input_space`) and the data format of the model's output
(`self.output_space`, which is required only if you're doing supervised
learning). These attributes should be instances of `pylearn2.space.Space` (and
generally are instances of `pylearn2.space.VectorSpace`, a subclass of
`pylearn2.space.Space` used to represent batches of vectors). Without getting
too much into details, this mechanism allows for automatic conversion between
different data formats (e.g. if your targets are stored as integer indexes in
the dataset but are required to be one-hot encoded by the model).

The `some_method_for_outputs` method is really where all the magic happens. Like
I said before, the name of the method doesn't really matter, as long as your
`Cost` subclass knows that it's the one it has to call. This method expects a
tensor variable as input and returns a symbolic expression involving the input
and its parameters. What happens in between is up to you, and this is where you
can put all the Theano code you could possibly hope for. Just like you would do
in pure Theano scripts.

# Show me examples

So far we've only been handwaiving. Let's put these ideas to use by writing two
models, one which does supervised learning and one which does unsupervised
learning.

The data you train these models on is up to you, as long as it's represented in
a matrix of features (each row being an example) and a matrix of targets (each
row being a target for an example, obviously only required if you're doing
supervised learning). Note that it's not the only way to get data into Pylearn2,
but that's the one we'll be using as it's likely to be most people's use case.

For the purpose of this tutorial, we'll be training models on the venerable
MNIST dataset, which you can download as follows:

{% highlight bash %}
wget http://deeplearning.net/data/mnist/mnist.pkl.gz
{% endhighlight %}

Unzip it, and you're ready to go.

## Supervised learning using logistic regression

Let's keep things simple by porting to Pylearn2 what's pretty much the _Hello
World!_ of supervised learning: logistic regression. If you haven't already, go
read

Let's assume you have a logistic regression model you're very proud of that
strangely resembles the one from the
[deeplearning.net tutorial](http://www.deeplearning.net/tutorial/logreg.html#logreg)
and you would like to be able to use it in Pylearn2.

# What have we gained?

TODO: talk about why replacing boilerplate code with THIS boilerplate code

# What's next?

TODO: talk about exploring Pylearn2 to avoid reinventing the wheel for nothing.
