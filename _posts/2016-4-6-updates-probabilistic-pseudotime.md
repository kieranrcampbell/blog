---
layout: post
title: "Probabilistic models for single-cell pseudotime"
tags:
    - pseudotime
    - Bayesian inference
---

I spent last autumn working on probabilistic methods for pseudotime inference and in particular examining the effects of uncertainty on pseudotime trajectories and calling differential expression. We then got a little distracted by some other projects, so this post is to wrap up several things that came out of that work. Our preprint for this is [now on biorxiv](http://biorxiv.org/content/early/2016/04/05/047365) and below I mention some other developments.

## Talk at NIPS Machine learning in computational biology

I presented our work at the [C1omics](http://radiant-project.eu/C1omics2015.html) conference in Manchester in November then at the [NIPS machine learning in computational biology workshop](http://mlcb.org/) in Montreal a couple of weeks later in December. The slides for this can be found below.

<iframe src="//www.slideshare.net/slideshow/embed_code/key/3AcUiXWIXwTXuu" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/KieranCampbell4/nips-machine-learning-in-computational-biology-presentation" title="NIPS machine learning in computational biology presentation" target="_blank">NIPS machine learning in computational biology presentation</a> </strong> from <strong><a href="//www.slideshare.net/KieranCampbell4" target="_blank">Kieran Campbell</a></strong> </div>

## pseudogp R package

We also created the pseudogp R package [available on github](http://github.com/kieranrcampbell/pseudogp/). It uses the [Stan probabilistic programming language](http://mc-stan.org) to fit two-dimensional probabilistic trajectories to single-cell data, achieving similar results to existing pseudotime inference algorithms but with full posterior uncertainty information for each cell.

## A hint of what's next

We became interested in the Bayesian approach to single-cell expression analysis and in particular using informative priors to introduce useful biological information. Single-cell RNA-seq is notoriously noisy and so the addition of information can help with robust statistical inference. As a result we've been working on Bayesian methods for learning pseudotimes from a small panel of marker genes where some knowledge of gene dynamics is known a priori. Our package for this is [on github](http://github.com/kieranrcampbell/bnlfa/) along with [a vignette](http://kieranrcampbell.github.io/bnlfa) detailing inference and model choice. Any feedback is appreciated.