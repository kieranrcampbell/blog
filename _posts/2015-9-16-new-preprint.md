---
layout: post
title: "New preprint: Gaussian Processes for pseudotime inference in single-cell RNA-seq data"
tags:
    - Gaussian processes
    - scRNA-seq
    - Bayesian inference
---

Today my supervisor [Chris Yau](http://www.ndm.ox.ac.uk/principal-investigators/researcher/christopher-yau) and I submitted our [new preprint](http://biorxiv.org/content/early/2015/09/15/026872) on using Gaussian Process Latent Variable Models (GP-LVM) for the inference of pseudotime from single-cell RNA-seq data. This work allowed me to explore some pretty cool areas, not least [Gaussian Processes](https://en.wikipedia.org/wiki/Gaussian_process) and the [Julia programming language](http://julialang.org/). All our code is available on [github](http://www.github.com/kieranrcampbell/gpseudotime) though predictably there's a lot of R to handle all the plotting.

## Single-cell RNA-seq and pseudotime

Single-cell RNA-seq is coming of age as a method to quantify gene expression in single cells. One exciting result of this is the idea of pseudotime - cells asynchronously develop through some biological process (differentiation, cell cycle, apoptosis) and we want to assign a surrogate measure of their progression. The main method of choice currently is [Monocle](http://www.nature.com/nbt/journal/v32/n4/full/nbt.2859.html) and we've previously worked on a [novel method](http://www.github.com/kieranrcampbell/embeddr)  too using Laplacian Eigenmaps and Principal Curves.

However, one enduring characteristic of this data is how noisy it is. We realised early on that slight changes in pseudotime assignment would have drastic effects on which genes were called as differentially expressed. All methods to date are purely algorithmic in the pseudotime fitting and resulting methods that could be used to quantify uncertainty (e.g. using boostrap samples) are computationally expensive.

## Gaussian Processes for pseudotime

Our main approach is to use a probabilistic model to infer pseudotimes so we can quantify the posterior uncertainty by sampling from $$ p( \mathrm{pseudotime} \| \mathrm{data}) $$. One way of doing this is to use Gaussian Process Latent Variable Models (for an overview see e.g. [here](http://papers.nips.cc/paper/2540-gaussian-process-latent-variable-models-for-visualisation-of-high-dimensional-data.pdf)). GPs essentially assume points are generated from a multivariate normal distribution that is entirely defined by its covariance function. The basic idea behind this for pseudotime ordering is if two points have similar latent pseudotimes they have a large covariance value and thus appear close together in gene expression space. In our paper we use the squared exponential kernel so the covariance between two cells is given by $$ K(t, t') = \exp( -\lambda (t-t')^2 ) $$.


![GP for pseudotime]({{ site.baseurl }}/img/monocle_all.png)
*Our GPLVM inference method applied to the Monocle dataset. The large width of the 95% credible interval in (D) highlights the need for uncertainty in pseudotime analyses.*

One particular issue we faced was how to define the prior $$\pi(\mathbf{t}) $$. Since pseudotime is an artificial measure all pseudotimes are equivalent up to scale, translation and parity transformations (so a pseudotime running from 0 to 1 is just as good as one running from 13 to 17 or $$-  \pi $$ to $$ -e $$). The problem with this is it's completely ill-defined with resepect to any probability density so the question becomes how to force the pseudotimes into a particular interval and occupy it well. We found a [recently published](http://arxiv.org/abs/1506.03768) method (accepted to NIPS 2015) that models each point (here a cell) as repelling charged particles which then 'fill out' over the interval of choice. One interesting result of using a Bayesian approach is we can discover the ordering without any prior knowledge as opposed to the original paper where the points are always stuck in the initial ordering due to the maximum-likelihood approach.

A final advantage of our method is because it works in a reduced dimension representation of the data (we use Laplacian Eigenmaps, but if you choose your genes right PCA will work too) it's very easy to check the posterior is correct. Both our likelihood and prior are notoriously multimodal, meaning if the parameters aren't quite right it's very easy to get stuck in a local maximum. Interestingly, these have geometrically intuitive interpretations, such as the pseudotimes curving back on themselves. 

Our initial results show that the 95% credible interval of pseudotimes can be as high as 0.5 (see figure), which is half the entire pseudotime window. This has real consequences when talking about pseudotime and in particular differential expression, where statistical tests always assume inputs with zero measurement error. 

## Some thoughts on Julia

Starting a new project from scratch allowed me to test drive the [Julia](http://www.julialang.org) programming language. Julia is a fairly new addition to the language arena with an emphasis on fast numerical computation and a simple matlab-style array syntax.

Overall, Julia has been pretty awesome (waiting for the MCMC traces in Python lead to more reddit-procrastination than usual). Though there's a slight lag for the JIT compiler at the start, running long MCMC traces is much faster than usual and the native array support avoids the pain of np.everything in python. Because the syntax shares elements of Python, Matlab & R it will destroy your syntax-error-free coding when you return to those languages. The only downside is a lot of the packages are underdeveloped (particularly plotting compared to R) due to the relative youth of the lagnauge, though hopefully this will improve. 

