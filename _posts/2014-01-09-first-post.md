---
layout: post-no-feature
title: First Post
description: "My very first post"
categories: articles
date: 2014-01-09
---

# Hi!

Just a rubbish post used to test code highlighting features.

This function builds a YAML string from ``state.yaml_template``, taking the
values of hyper-parameters from ``state.hyper_parameters``, creates the
corresponding object and trains it (like train.py), then run the function in
``state.extract_results`` on it, and store the returned values into
``state.results``.

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
