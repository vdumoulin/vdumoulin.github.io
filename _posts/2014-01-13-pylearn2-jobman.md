---
layout: post-no-feature
title: Integrating Pylearn2 and Jobman
categories: articles
comments: true
date: 2014-01-13
---

_This post is adapted from an iPython Notebook I wrote which is part of a pull
request to be added to the Pylearn2 documentation. I assume the reader is
familiar with Pylearn2 (mostly its YAML file framework for describing
experiments) and with [Jobman](http://deeplearning.net/software/jobman/), a tool
to launch and manage experiments._

### The problem

Suppose you have a YAML file describing an experiment which looks like that:

{% highlight text %}
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.mnist.MNIST {
        which_set: 'train',
        one_hot: 1,
    start: 0,
        stop: 50000
    },
    model: !obj:pylearn2.models.mlp.MLP {
        layers: [
            !obj:pylearn2.models.mlp.Sigmoid {
                layer_name: 'h0',
                dim: 500,
                sparse_init: 15,
            }, !obj:pylearn2.models.mlp.Softmax {
                layer_name: 'y',
                n_classes: 10,
                irange: 0.
            }
        ],
        nvis: 784,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 100,
        learning_rate: 1e-3,
        learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
            init_momentum: 0.5,
        },
        monitoring_batches: 10,
        monitoring_dataset : *train,
        termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs: 50
        },
    },
    save_path: "mlp.pkl",
    save_freq : 5
}
{% endhighlight %}

You're not sure if the learning rate and the momentum coefficient are optimal,
though, and you'd like to try different hyperparameter values to see if you can
come up with something better.

One (painful) way to do it would be to create multiple copies of the YAML file
and, for each copy, manually change the value of the learning rate and the
momentum coefficient. You'd then call the `train` script on each of these
copies. This solution is not satisfying for multiple reasons:

* This is long and tedious
* There's lot of code duplication going on
* You'd better be sure there are no errors in the original YAML file, or else
  you're in for a nice editing ride (been there)

Ideally, the solution should involve a single YAML file and some way of
specifying how hyperparameter should be handled. One such solution exists,
thanks to Pylearn2 and Jobman.

### Solution overview

Pylearn2 can instantiate a `Train` object specified by a YAML string via the
`pylearn2.config.yaml_parse.load` method; using this method and Python's string
substitution syntax, we can "fill the blanks" of a template YAML string based
on our original YAML file and run the experiment described by that string.

In order to to that, we'll need a dictionary mapping hyperparameter names to
their value. This is where Jobman will prove useful: Jobman accepts
configuration files describing a job's parameters, and its syntax allows to
initialize parameters by calling an external Python method. This way, we can
randomly sample hyperparameters for our experiment.

To summarize it all, we will

1. Adapt the YAML file by replacing hyperparameter values with string
   substitution statements
2. Write a configuration file specifying how to initialize the hyperparameter
   dictionary
1. Read the YAML file into a string
2. Fill in hyperparameter values using string substitution with the
   hyperparameter dictionary
3. Instantiate a `Train` object with the YAML string by calling
   `pylearn2.config.yaml_parse.load`
4. Call the `Train` object's `main_loop` method
5. Extract results from the trained model

Let's break it down.

### Adapting the YAML file

This step is pretty straightforward. Looking back to our example, the only lines
we have to replace are

{% highlight text %}
        learning_rate: 1e-3,
        learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
            init_momentum: 0.5,
        },
{% endhighlight %}

Using string subsitution syntax, they become

{% highlight text %}
        learning_rate: %(learning_rate)f,
        learning_rule: !obj:pylearn2.training_algorithms.learning_rule.Momentum {
            init_momentum: %(init_momentum)f,
        },
{% endhighlight %}

### String substitution and training logic

The next step, assuming we already have a dictionary mapping hyperparameters
to their values, would be to build a method which

1. takes the YAML string and the hyperparameter dictionary as inputs,
2. does string substitution on the YAML string,
3. calls the `pylearn2.config.yaml_parse.load` method to instantiate a `Train`
   object and calls its `main_loop` method and
4. extracts and returns results after the model is trained.

Luckily for us, one such method already exists:
`pylearn2.scripts.jobman.experiment.train_experiment`.

This method integrates with Jobman: it expects `state` and `channel`
arguments as input and returns `channel.COMPLETE` at the end of training.
Here's the method's full implementation:

{% highlight python %}
def train_experiment(state, channel):
    """
    Train a model specified in state, and extract required results.

    This function builds a YAML string from ``state.yaml_template``, taking
    the values of hyper-parameters from ``state.hyper_parameters``, creates
    the corresponding object and trains it (like train.py), then run the
    function in ``state.extract_results`` on it, and store the returned values
    into ``state.results``.

    To know how to use this function, you can check the example in tester.py
    (in the same directory).
    """
    yaml_template = state.yaml_template

    # Convert nested DD into nested ydict.
    hyper_parameters = expand(flatten(state.hyper_parameters), dict_type=ydict)

    # This will be the complete yaml string that should be executed
    final_yaml_str = yaml_template % hyper_parameters

    # Instantiate an object from YAML string
    train_obj = pylearn2.config.yaml_parse.load(final_yaml_str)

    try:
        iter(train_obj)
        iterable = True
    except TypeError:
        iterable = False
    if iterable:
        raise NotImplementedError(
                ('Current implementation does not support running multiple '
                 'models in one yaml string.  Please change the yaml template '
                 'and parameters to contain only one single model.'))
    else:
        # print "Executing the model."
        train_obj.main_loop()
        # This line will call a function defined by the user and pass train_obj
        # to it.
        state.results = jobman.tools.resolve(state.extract_results)(train_obj)
        return channel.COMPLETE
{% endhighlight %}

As you can see, it builds a dictionary out of state.hyper_parameters and uses
it to do string substitution on state.yaml_template.

It then instantiates the `Train` object as described in the YAML string and
calls its `main_loop` method.

Finally, when the method returns, it calls the method referenced in the
`state.extract_results` string by passing it the `Train` object as argument.
This method is responsible to extract any relevant results from the `Train`
object and returning them, either as is or in a `DD` object. The return value
is stored in `state.results`.

## Writing the extraction method

Your extraction method should accept a `Train` object instance and return
either a single value (`float`, `int`, `str`, etc.) or a `DD` object containing
your values.

For the purpose of this tutorial, let's write a simple method which extracts
the misclassification rate and the NLL from the model's monitor:

{% highlight python %}
from jobman.tools import DD

def results_extractor(train_obj):
    channels = train_obj.model.monitor.channels
    train_y_misclass = channels['y_misclass'].val_record[-1]
    train_y_nll = channels['y_nll'].val_record[-1]

    return DD(train_y_misclass=train_y_misclass, train_y_nll=train_y_nll)
{% endhighlight %}

Here we extract misclassification rate and NLL values at the last training
epoch from their respective channels of the model's monitor and return a `DD`
object containing those values.

### Building the hyperparameter dictionary

Let's now focus on the last piece of the puzzle: the Jobman configuration file.
Your configuration file should contain

* `yaml_template`: a YAML string representing your experiment
* `hyper_parameters.[name]`: the value of the `[name]` hyperparameter.
  You must have at least one such item, but you can have as many as you want.
* `extract_results`: a string written in `module.method` form representing the
  result extraction method which is to be used

Here's how a configuration file could look for our experiment:

{% highlight text %}
yaml_template:=@__builtin__.open('mlp.yaml').read()

hyper_parameters.learning_rate:=@utils.log_uniform(1e-5, 1e-1)
hyper_parameters.init_momentum:=@utils.log_uniform(0.5, 1.0)

extract_results = "utils.results_extractor"
{% endhighlight %}

Notice how we're using the `key:=@method` statement. This serves two purposes:

1. We don't have to copy the yaml file to the configuration file as a long,
   hard to edit string.
2. We don't have to hard-code hyperparameter values, which means every time
   Jobman is called with this configuration file, it'll receive different
   hyperparameters.

For reference, here's `utils.log_uniform`'s implementation:

{% highlight python %}
def log_uniform(low, high):
    """
    Generates a number that's uniformly distributed in the log-space between
    `low` and `high`

    Parameters
    ----------
    low : float
        Lower bound of the randomly generated number
    high : float
        Upper bound of the randomly generated number

    Returns
    -------
    rval : float
        Random number uniformly distributed in the log-space specified by `low`
        and `high`
    """
    log_low = numpy.log(low)
    log_high = numpy.log(high)
    
    log_rval = numpy.random.uniform(log_low, log_high)
    rval = float(numpy.exp(log_rval))

    return rval
{% endhighlight %}

### Running the whole thing

Here's how you would train your model:

{% highlight bash %}
$ jobman cmdline pylearn2.scripts.jobman.experiment.train_experiment mlp.conf
{% endhighlight %}

Alternatively, you can chain jobs using `jobdispatch`:
{% highlight bash %}
$ jobdispatch --local --repeat_jobs=10 jobman cmdline \
  pylearn2.scripts.jobman.experiment.train_experiment mlp.conf
{% endhighlight %}
