---
layout: post-no-feature
title: Your models in Pylearn2
description: "A tutorial on the minimal effort required to develop a new model in Pylearn2"
categories: articles
comments: true
published: true
date: 2014-10-10
---

# Who should read this

This tutorial is designed for pretty much anyone working with Theano who's tired
of writing the same old boilerplate code over and over again. You have SGD
implementations scattered in pretty much every experiment file? Pylearn2 looks
attractive but you think porting your Theano code to it is too much of an
investment? This tutorial is for you.

Having played with Pylearn2 and looked at some of the tutorials is stongly
recommended. If you're completely new to Pylearn2, have a look at the 
[softmax regression tutorial](http://nbviewer.ipython.org/github/lisa-lab/pylearn2/blob/master/pylearn2/scripts/tutorials/softmax_regression/softmax_regression.ipynb).

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
like it is now: all I did is port what Joost van Amersfoort wrote in Theano
(see his code [here](https://github.com/y0ast/Variational-Autoencoder/blob/master/Theano/VariationalAutoencoder.py))
to Pylearn2 in order to reproduce the experiments in
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
  formatted.
* It computes the cost expression by feeding the input to the model and
  receiving its output.
* It differentiates the cost expression with respect to the model parameter and
  returns the gradients to the training algorithm.

What's nice about `Cost` is if you follow the guidelines I'm about to describe,
you only have to worry about the cost expression; the gradient part is all
handled by the `Cost` base class, and a very useful `DefaultDataSpecsMixin`
mixin subclass is defined to handle the data description part (more about that
when we look at the `Model` subclass).

Let's look at how the subclass should look:

{% highlight python %}
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin


class MyCostSubclass(DefaultDataSpecsMixin, Cost):
    # Here it is assumed that we are doing supervised learning
    supervised = True

    def expr(self, model, data, **kwargs):
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
the targets as the second element (once again, bear with me if everything isn't
completely clear for the moment, you'll understand soon enough).

We then get the model output by calling its `some_method_for_outputs` method,
whose name and behaviour is really for you to decide, as long as your `Cost`
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

    def some_method_for_outputs(self, inputs):
        # Some computation involving the inputs
{% endhighlight %}

The first thing you should do if you're overriding the constructor is call the
the superclass' constructor. Pylearn2 checks for that and will scold you if you
don't.

You should then initialize you model parameters **as shared variables**:
Pylearn2 will build an updates dictionary for your model variables using
gradients returned by your cost. _**Protip: the `pylearn2.utils.sharedX` method
initializes a shared variable with the value and an optional name you provide.
This allows your code to be GPU-compatible without putting too much thought into
it.**_ For instance, a weights matrix can be initialized this way:

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
can put all the Theano code you could possibly hope for, just like you would do
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

To make things easier to manipulate, we'll unzip that file into six different
files:

{% highlight bash %}
python -c "from pylearn2.utils import serial; \
           data = serial.load('mnist.pkl'); \
           serial.save('mnist_train_X.pkl', data[0][0]); \
           serial.save('mnist_train_y.pkl', data[0][1].reshape((-1, 1))); \
           serial.save('mnist_valid_X.pkl', data[1][0]); \
           serial.save('mnist_valid_y.pkl', data[1][1].reshape((-1, 1))); \
           serial.save('mnist_test_X.pkl', data[2][0]); \
           serial.save('mnist_test_y.pkl', data[2][1].reshape((-1, 1)))"
{% endhighlight %}

## Supervised learning using logistic regression

Let's keep things simple by porting to Pylearn2 what's pretty much the _Hello
World!_ of supervised learning: logistic regression. If you haven't already, go
read the [deeplearning.net tutorial](http://www.deeplearning.net/tutorial/logreg.html#logreg)
on logistic regression. Here's what we have to do:

* Implement the negative log-likelihood (NLL) loss in our `Cost` subclass
* Initialize the model parameters W and b
* Implement the model's logistic regression output

Let's start by the `Cost` subclass:

{% highlight python %}
import theano.tensor as T
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin


class LogisticRegressionCost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)
        
        inputs, targets = data
        outputs = model.logistic_regression(inputs)
        loss = -(targets * T.log(outputs)).sum(axis=1)
        return loss.mean()
{% endhighlight %}

Easy enough. We assumed our model has a `logistic_regression` method which
accepts a batch of examples and computes the logistic regression output. We will
implement that method in just a moment. We also computed the loss as the average
negative log-likelihood of the targets given the logistic regression output, as
described in the deeplearning.net tutorial. Also, notice how we set `supervised`
to `True`.

Now for the `Model` subclass:

{% highlight python %}
import numpy
import theano.tensor as T
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX


class LogisticRegression(Model):
    def __init__(self, nvis, nclasses):
        super(LogisticRegression, self).__init__()

        self.nvis = nvis
        self.nclasses = nclasses

        W_value = numpy.random.uniform(size=(self.nvis, self.nclasses))
        self.W = sharedX(W_value, 'W')
        b_value = numpy.zeros(self.nclasses)
        self.b = sharedX(b_value, 'b')
        self._params = [self.W, self.b]

        self.input_space = VectorSpace(dim=self.nvis)
        self.output_space = VectorSpace(dim=self.nclasses)

    def logistic_regression(self, inputs):
        return T.nnet.softmax(T.dot(inputs, self.W) + self.b)
{% endhighlight %}

The model's constructor receives the dimensionality of the input and the number
of classes. It initializes the weights matrix and the bias vector with
`sharedX`. It also sets its input space to an instance of `VectorSpace` of
the dimensionality of the input (meaning it expects the input to be a batch of
examples which are all vectors of size `nvis`) and its output space to an
instance of `VectorSpace` of dimension `nclasses` (meaning it produces an output
corresponding to a batch of probability vectors, one element for each possible
class).

The `logistic_regression` method does pretty much what you would expect: it
returns a linear transformation of the input followed by a softmax
non-linearity.

How about we give it a try? Save those two code snippets in a single file (e.g.
`log_reg.py` and save the following in `log_reg.yaml`:

{% highlight text %}
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.dense_design_matrix.DenseDesignMatrix {
        X: !pkl: 'mnist_train_X.pkl',
        y: !pkl: 'mnist_train_y.pkl',
        y_labels: 10,
    },
    model: !obj:log_reg.LogisticRegression {
        nvis: 784,
        nclasses: 10,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 200,
        learning_rate: 1e-3,
        monitoring_dataset: {
            'train' : *train,
            'valid' : !obj:pylearn2.datasets.dense_design_matrix.DenseDesignMatrix {
                X: !pkl: 'mnist_valid_X.pkl',
                y: !pkl: 'mnist_valid_y.pkl',
                y_labels: 10,
            },
            'test' : !obj:pylearn2.datasets.dense_design_matrix.DenseDesignMatrix {
                X: !pkl: 'mnist_test_X.pkl',
                y: !pkl: 'mnist_test_y.pkl',
                y_labels: 10,
            },
        },
        cost: !obj:log_reg.LogisticRegressionCost {},
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs: 15
        },
    },
}
{% endhighlight %}

Run the following command:

{% highlight bash %}
python -c "from pylearn2.utils import serial; \
           train_obj = serial.load_train_file('log_reg.yaml'); \
           train_obj.main_loop()"
{% endhighlight %}

Congratulations, you just implemented your first model in Pylearn2!

*(By the way, the targets you used to initialize `DenseDesignMatrix` instances
were column matrices, yet your model expects to receive one-hot encoded vectors.
The reason why you can do that is because Pylearn2 does the conversion for you
via the `data_specs` mechanism. That's why specifying the model's `input_space`
and `output_space` is important.*


## Unsupervised learning using an autoencoder

Let's now have a look at an unsupervised learning example: an autoencoder with
tied weights. Once again, having read [deeplearning.net tutorial](http://www.deeplearning.net/tutorial/logreg.html#logreg)
on the subject is recommended. Here's what we'll do:

* Implement the binary cross-entropy reconstruction loss in our `Cost` subclass
* Initialize the model parameters W and b
* Implement the model's reconstruction logic

Let's start again by the `Cost` subclass:

{% highlight python %}
import theano.tensor as T
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin


class AutoencoderCost(DefaultDataSpecsMixin, Cost):
    supervised = False

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)
        
        X = data
        X_hat = model.reconstruct(X)
        loss = -(X * T.log(X_hat) + (1 - X) * T.log(1 - X_hat)).sum(axis=1)
        return loss.mean()
{% endhighlight %}

We assumed our model has a `reconstruction` method which encodes and decodes its
input. We also computed the loss as the average binary cross-entropy between the
input and its reconstruction. This time, however, we set `supervised` to
`False`.

Now for the `Model` subclass:

{% highlight python %}
import numpy
import theano.tensor as T
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX


class Autoencoder(Model):
    def __init__(self, nvis, nhid):
        super(Autoencoder, self).__init__()

        self.nvis = nvis
        self.nhid = nhid

        W_value = numpy.random.uniform(size=(self.nvis, self.nhid))
        self.W = sharedX(W_value, 'W')
        b_value = numpy.zeros(self.nhid)
        self.b = sharedX(b_value, 'b')
        c_value = numpy.zeros(self.nvis)
        self.c = sharedX(c_value, 'c')
        self._params = [self.W, self.b, self.c]

        self.input_space = VectorSpace(dim=self.nvis)

    def reconstruct(self, X):
        h = T.tanh(T.dot(X, self.W) + self.b)
        return T.nnet.sigmoid(T.dot(h, self.W.T) + self.c)
{% endhighlight %}

The constructor looks a lot like for the logistic regression example, except
that this time we don't need to specify the model's output space.

The `reconstruct` method simply encodes and decodes its input.

Let's try to train it. Save the two code snippets in a single file (e.g.
`autoencoder.py` and save the following in `autoencoder.yaml`:

{% highlight text %}
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.dense_design_matrix.DenseDesignMatrix {
        X: !pkl: 'mnist_train_X.pkl',
    },
    model: !obj:autoencoder.Autoencoder {
        nvis: 784,
        nhid: 200,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 200,
        learning_rate: 1e-3,
        monitoring_dataset: {
            'train' : *train,
            'valid' : !obj:pylearn2.datasets.dense_design_matrix.DenseDesignMatrix {
                X: !pkl: 'mnist_valid_X.pkl',
            },
            'test' : !obj:pylearn2.datasets.dense_design_matrix.DenseDesignMatrix {
                X: !pkl: 'mnist_test_X.pkl',
            },
        },
        cost: !obj:autoencoder.AutoencoderCost {},
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs: 15
        },
    },
}
{% endhighlight %}

Run the following command:

{% highlight bash %}
python -c "from pylearn2.utils import serial; \
           train_obj = serial.load_train_file('autoencoder.yaml'); \
           train_obj.main_loop()"
{% endhighlight %}

# What have we gained?

At this point you might be thinking _"There's still boilerplate code to write;
what have we gained?"_

The answer is we gained access to a plethora of scripts, model parts, costs and
training algorithms all built into Pylearn2. You don't have to re-invent the
wheel anymore when you wish to train using SGD and momentum. You want to switch
from SGD to BGD? In Pylearn2 this is as simple as changing the training
algorithm description in your YAML file.

Like I said earlier, what I'm showing is the **bare minimum** needed to
implement a model in Pylearn2. Nothing prevents you from digging deeper in the
codebase and overriding some methods to gain new functionalities.

Here's an example of how a few more lines of code can do a lot for you in
Pylearn2.

## Monitoring various quantities during training

Let's monitor the classification error of our logistic regression classifier.

To do so, you'll have to override `Model`'s `get_monitoring_data_specs` and
`get_monitoring_channels` methods. The former specifies what the model needs for
its monitoring, and in which format they should be provided. The latter does the
actual monitoring by returning an `OrderedDict` mapping string identifiers to
their quantities.

Let's look at how it's done. Add the following to `LogisticRegression`:

{% highlight python %}
# Keeps things compatible for Python 2.6
from theano.compat.python2x import OrderedDict
from pylearn2.space import CompositeSpace


class LogisticRegression(Model):
    # (Your previous code)

    def get_monitoring_data_specs(self):
        space = CompositeSpace([self.get_input_space(),
                                self.get_target_space()])
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)

    def get_monitoring_channels(self, data):
        space, source = self.get_monitoring_data_specs()
        space.validate(data)

        X, y = data
        y_hat = self.logistic_regression(X)
        error = T.neq(y.argmax(axis=1), y_hat.argmax(axis=1)).mean()

        return OrderedDict([('error', error)])
{% endhighlight %}

The content of `get_monitoring_data_specs` may look cryptic at first.
Documentation for data specs can be found
[here](http://deeplearning.net/software/pylearn2/internal/data_specs.html), but
all you have to know is that this is the standard way in Pylearn2 to request a
tuple whose first element represents features and second element represents
targets.

The content of `get_monitoring_channels` should more familiar. We start by
checking `data` just as in `Cost` subclasses' implementation of `expr`, and we
separate `data` into features and targets. We then get predictions by
calling `logistic_regression` and compute the average error the standard way.
We return an `OrderedDict` mapping `'error'` to the Theano expression for the
classification error.

Launch training again using

{% highlight bash %}
python -c "from pylearn2.utils import serial; \
           train_obj = serial.load_train_file('log_reg.yaml'); \
           train_obj.main_loop()"
{% endhighlight %}

and you'll see the classification error being displayed with other monitored
quantities.

# What's next?

The examples given in this tutorial are obviously very simplistic and could be
easily replaced by existing parts of Pylearn2. They do, however, show the path
one needs to take to implement arbitrary ideas in Pylearn2.

In order not to reinvent the wheel, it is oftentimes useful to dig into
Pylearn2's codebase to see what's implemented. For instance, the VAE framework
I wrote relies on the MLP framework to represent the mapping from inputs to
conditional distribution parameters.

Although code reuse is desirable, the ease with which it can be acomplished
depends a lot on the level of familiarity you have with Pylearn2 and how
different your model is from what's already in there. You should never feel
ashamed to dump a bunch of Theano code inside `Model` subclass' method like I
showed here if that's what works for you. Modularity and code reuse can be
brought to your code gradually and at your own pace, and in the meantime you can
still benefit from Pylearn2's features, like human-readable experiment
descriptions, automatic monitoring of various quantities, easily-interchangeable
training algorithms and so on.
