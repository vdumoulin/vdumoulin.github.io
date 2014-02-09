---
layout: post-no-feature
title: State of TIMIT dataset in Pylearn2
description: "An unfinished business"
categories: articles
comments: true
date: 2014-02-09
---

This is a small post just to let you know the current state of the TIMIT dataset
in Pylearn2. You can find the source code
[here](https://github.com/vdumoulin/research/blob/master/code/pylearn2/datasets/timit.py).

I'm mostly done working on the initialization, thanks to Laurent Dinh's
[code](https://github.com/laurent-dinh/mumbler/blob/master/dataset/timit.py).

The dataset is able to load all relevant files, but only the acoustic samples
are used. For now I won't bother including phones/phonemes and auxiliary speaker
information, as I have already plenty to manage with the acoustic samples
already.

The biggest problem I'm facing is the lack of support for variable-length
sequences in Pylearn2. The library is mostly built around the assumption that
your data will be a matrix of training examples (with examples being stored in
the matrix's rows) and a matrix of training targets.

One way to circumvent that is to transform the dataset into a matrix of training
examples each containing a sequence of k frames and a matrix of training targets
each containing the next frame after its corresponding sequence. The problem is
it causes lots of duplication in memory.

Another solution would be to keep the dataset as an array of variable-length
sequences and maintain a _visiting order_ list of tuples containing the index
of a sequence and the index of the starting frame in the sequence. This is where
I'm currently headed. One problem with this solution is that no iterator built
in Pylearn2 is suited to working with the _visiting order_ list. I'll have to
write one on my own, which might take some time, as I'm not fully fluent with
the whole _data specs_ framework used in Pylearn2.

Conclusion: if you're waiting for me to finish the TIMIT dataset implementation
in Pylearn2, this might take some time; you'd be better off working directly in
Theano with Laurent's TIMIT class for now.
