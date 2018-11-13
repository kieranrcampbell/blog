---
layout: post
title: "Using R and Tensorflow to understand gene expression in single cancer cells"
---

<h4>Table of Contents</h4>
* this unordered seed list will be replaced by toc as unordered list  
{:toc}


## Introduction

Massively parallel sequencing of nucelic acids in single cells using technologies such as 
[10X genomics](https://www.10xgenomics.com/solutions/single-cell/) for 
[RNA-seq](https://en.wikipedia.org/wiki/RNA-Seq) and 
[Direct Library Preparation (DLP)](https://www.ncbi.nlm.nih.gov/pubmed/28068316) for single-cell whole genome sequencing lets us understand the molecular properties of cancer cells. Ideally, we would like to understand how the sets of mutations in a tumour cell impact its gene expression, possibly causing it to become resistant to chemotherapy. However, the majority of scalable single-cell assays measure  gene expression and mutations in different cells, albeit from the same tumour.

To solve this problem and link a tumour's gene expression to its genomic profile, we developed [clonealign](https://www.github.com/kieranrcampbell/clonealign), a probabilistic model that assigns each cell measured using single-cell RNA-sequencing to a genomic clone (a set of cells with similar mutational profiles) measured using single-cell whole genome sequencing. This post introduces the clonealign model, stochastic gradient variational Bayesian inference, and sketches out how we used 
[Tensorflow](https://tensorflow.rstudio.com/)
 to implement these as an R package for integration with other genomics tools.

## clonealign: linking scRNA-seq to scDNA-seq

The key behind assigning cells measured using scRNAseq to those with scDNAseq is to make an assumption that links the copy number from the DNA-seq to the expression profile from the RNA-seq. The copy number can simply be thought of as the number of copies of each gene - in healthy cells this should be 2 (diploid), but in many cancers errors in DNA repair result in losses (copy number number less than 2) and amplifications (copy number greater than 2). Crucially, we look at regions of the genome where the copy number is _clone specific_ and assume this has an impact on gene expression - genes in cells in a clone with copy number $$3$$ will have $$3/2$$ times more expression than those in a clone of copy number $$2$$:


<img src="{{ site.baseurl }}/img/clonealign/cnv-expression.png" width="60%"/>


Since we know the copy number of each clone in advance we can write this as a generative probabilistic model that looks similar to a mixture model. Essentially, the expected counts coming from a gene in a cell _given that cell is assigned to a clone_ consists of some base expression of that gene multiplied by the copy number of that gene in that clone. We further model fixed effects (e.g. batch or sample specific covariates) and random effects that allow more gene expression variation to be explained than copy number alone. Given this probabilistic model, we can perform inference on the posterior distributions of the clonal assignments (detailed below), using a negative binomial likelihood model as is common for RNA-seq. Altogether, this gives us the clonealign model:

<img src="{{ site.baseurl }}/img/clonealign/clonealign-overview.png" width="80%"/>


## Stochastic gradient variational Bayes in R and Tensorflow

### Why be variational?

We would like to perform inference on 
$$p(z_n | Y, \Lambda)$$ 
-- the posterior distribution of the clone assignments $$z_n$$ given the gene expression data $$Y$$ and copy number data $$\Lambda$$, marginalizing out all the model parameters $$\Theta$$ such as mean expression of each gene and any covariate coefficients. However, this requires performing high-dimensional integration over the clone assignments and model parameters and is exceptionally hard to compute.

Instead of computing 
$$p(z_n, \Theta  | Y, \Lambda)$$ 
exactly, we turn to Variational Inference that seeks a _variational distribution_ 
$$ q(z, \Theta | \eta) $$ 
of a tractable parametric family that approximates the full posterior, i.e. 
$$q(z, \Theta | \eta) \approx p(z_n, \Theta|Y, \Lambda) $$.
The quantities $$\eta$$ are known as the _variational parameters_, and the trick of variational inference is to tweak these parameters so that the variational distribution looks as close to the posterior as possible. To do this we attempt to minimize a measure of divergence between the two distributions, known as the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) defined as

$$ \text{KL}\left[ q(z, \Theta | \eta) \|  p(z, \Theta|Y, \Lambda) \right]
= \text{E}_{q(z, \Theta | \eta)}\left[ \log q(z, \Theta | \eta) - \log p(z, \Theta|Y, \Lambda) \right].$$

While this still requires computing 
$$p(z|Y, \Lambda)$$ 
-- which we can't do in the first place -- the above objective is equivalent to minimizing

$$ \text{E}_{q(z, \Theta | \eta)}\left[ \log q(z, \Theta | \eta) - \log p(Y, \Lambda | z, \Theta) - \log p(z, \Theta) \right] $$

where we've applied Bayes rule to 
$$ p(z, \Theta | Y, \Lambda) $$ 
and the term 
$$p(Y, \Lambda)$$ 
falls out of the objective as a constant (the real trick behind variational inference!).

### Constructing variational distributions using the reparametrization trick

In practice we make a mean field assumption on the approximating distributions, so 
$$ q(z, \Theta | \eta) = q(z | \eta_z) q(\Theta | \eta_\Theta) $$. 
As we will see shortly, the natural approximating distribution of $$z$$ is categorical, allowing us to take the expectation above analytically. This means $$q(z_n = c) = \psi_{nc}$$ - in other words, the probability that cell $$ n $$ is assigned to clone $$c$$ takes the value $$ \psi_{nc} $$, where $$ \psi $$ is a variational parameter to be optimized with the obvious constraint that $$ \sum_c \psi_{nc} = 1.$$ In Tensorflow we write this as

{% highlight R %}
library(tensorflow)
psi <- tf$nn$softmax(tf$Variable(tf$zeros(shape = shape(N,C)))) # N cells and C clones
{% endhighlight %}

Here we have created a <code>Variable</code> to be optimized (the variational parameter), instantiated to all zeros as an $$N \times C$$ matrix (to match the shape of $$\psi$$), and pass it through the [softmax function](https://en.wikipedia.org/wiki/Softmax_function) to ensure the probabilities for each cell sum to 1. 

We next look at the variational distributions for the model parameters, using $$\mu_g$$ (the base expression of each gene) as an example. We can no longer take this bound analytically so must resort to a Monte Carlo approximation of the loss. To do this we use a method that originated with [Variational Autoencoders](https://arxiv.org/abs/1312.6114)  known as the _reparametrization trick_ that uncouples the randomness in the expression from the variational parameters. To do this, we must be able to write

$$ \mu_g = g(\eta_{\mu_g}, \epsilon) $$ 

where $$\eta_\mu$$ are the variational parameters, $$\epsilon$$ is a random variable independent of $$\eta_\mu$$ and $$g$$ is an invertible differentiable function. For our base expression parameters, we know that they must be positive (since gene expression is count data), and have unknown mean and variance, so we posit a transform of the form

$$ \mu_g = \text{Softplus}(\nu_g + \sigma_g \epsilon) $$

where $$(\nu_g, \sigma_g) \equiv \eta_{\mu_g}$$ are the variational parameters analogous to the location and scale parameters in a transformed Gaussian  model, we sample $$\epsilon \sim \mathcal{N}(0,1)$$ where $$ \mathcal{N}(0,1) $$ is the standard normal distribution, and $$ \text{Softplus}(x) = \log(1 + \exp(x))$$. Then if $$f(\mu)$$ represents the joint probability of the parameters and data (we drop the dependency on other parameters for simplicity) then we can evaluate the loss as 

$$ \text{E}_{q(\mu)} \left[ \log q(\mu) - \log f(\mu) \right] =
\text{E}_{\epsilon \sim \mathcal{N}(0,1)}
\left[ \log \left( \mathcal{N}(\epsilon | 0, 1) \frac{dg^{-1}}{d\eta_{\mu}\;\;\;}(  g(\eta_{\mu_g}, \epsilon) ) \right) - \log f( g(\eta_{\mu_g}, \epsilon) )\right],
$$ 

where the expectation comes from $$S$$ Monte Carlo samples of $$\epsilon$$. While this looks complicated, it's equivalent to the bound above where quanties are passed through $$g$$ (to compute mu) or the inverse (to ensure the density of $$ q(\mu)$$ is correct). The good news is Tensorflow's computation graph essentially handles this for us, thanks to some nice engineering on the part of the [Tensorflow distributions](https://arxiv.org/abs/1711.10604) library. Therefore, we can implement it as

{% highlight R %}
tfd <- tf$contrib$distributions
tfb <- tfd$bijectors
qmu <- tfd$TransformedDistribution(
    bijector = tfb$Softplus(),
    distribution = tfd$Normal(loc = tfd$Variable(tf$zeros(G)), 
                              scale = tf$exp(tf$Variable(tf$zeros(G)))),
    name = "qmu"
)
mu_samples <- qmu$sample(S)
{% endhighlight %}

The different parts of this implementation break down as follows:
* <code>bijector</code> specifies the softplus function we pass the variational parameters and $$\epsilon$$ through - we could have also have specified $$\nu_g + \sigma_g \epsilon$$ but this is taken care of by the normal distribution below
* <code>distribution</code> specifies the base distribution that is Gaussian with mean $$\nu_g$$ and variance $$\sigma_g$$ - since Tensorflow _defines_ a Gaussian distribution with this mean and variance as the transform $$\nu_g + \sigma_g \epsilon$$ with $$\epsilon \sim \mathcal{N}(0,1)$$ this takes care of the inner statements in $$g$$. Note we pass the <code>Variable</code> for the scale (i.e. standard deviation) through an <code>exp</code> transformation to ensure it is positive. Both the mean and standard deviation are of length <code>G</code> for $$G$$ genes -- independent distributions for each gene
* <code>mu_samples</code> draws $$S$$ sample of $$\mu$$ to compute the lower bound

### Computing the lower bound

We can now use the quantities so far to compute our lower bound. Firstly, the _entropy_ term
$$ \text{E}_{q(\mu, z)} \log q(\mu, z) $$ becomes


{% highlight R %}
E_log_q <- tf$reduce_sum(psi * tf$log(psi)) + tf$reduce_sum(qmu$log_prob(mu_samples)) / S
{% endhighlight %}
where the first term (for $$\psi$$) is computed analytically while the second term (for $$\mu$$) is a stochastic estimate using Monte Carlo samples.

We next sketch how to compute 
$$ \text{E}_{q(\mu, z)} \log p(Y, \Lambda | \mu, z) $$
 -- for the full code please see [clonealign on github](https://www.gihub.com/kieranrcampbell/clonealign). First, we construct the expectation for the expression of Monte Carlo sample $$S$$, for clone $$C$$, and gene $$G$$:

 {% highlight R %}
L <- tf$placeholder(shape=shape(G,C)) # Copy number of gene g in clone c
s <- tf$placeholder(shape = N) # Size factors for N cells
expression_mean <- tf$einsum('sg,gc->scg', mu_samples, L)
expression_mean <- tf$einsum('n,scg->scng')
{% endhighlight %}

and then compute the log likelihood for the observed gene expression data <code>Y</code> in each Monte Carlo sample, clone, gene, and cell after converting to a different parameterization of the Negative Binomial distribution:
 {% highlight R %}
Y <- tf$placeholder(shape=shape(N,G)) # cell by gene placeholder for raw counts
dispersion <- tf$ones(1)
p <- expression_mean / (expression_mean + dispersion)
log_prob <- tfd$NegativeBinomial(probs = p, total_count = dispersion)$log_prob(Y)
{% endhighlight %}
where we have set the dispersion parameters to 1 for convenience but in practice model this using spline functions.

We are now in a position to compute the expectations with respect to $$z$$ and $$\mu$$. As mentioned above, we can take the expectation over $$z$$ analytically, giving

 {% highlight R %}
E_log_p <- tf$einsum('nc,scng->sg',psi,log_prob) 
{% endhighlight %}

followed by the expectation over $$\mu$$ using an MC estimate:
 {% highlight R %}
E_log_p <- tf$reduce_sum(E_log_p) / S
{% endhighlight %}

Finally, we implement a similar strategy for the priors, before wrapping it all up into a tensor named <code>elbo</code> (short for Evidence Lower BOund), which we then maximize using the Adam optimizer built in to Tensorflow.



## Clone-specific gene expression in triple-negative breast cancers

By applying the above inference framework to single-cell RNA-seq and single-cell DNA-seq from triple negative breast cancer patient-derived xenografts (PDXs) we were able to assign gene expression states to cancer clones. Crucially, from [previous work](https://www.nature.com/articles/nature13952) we knew that one of the clones (clone A) had higher _fitness_ and expanded over time. By applying clonealign we were able to associate gene expression patterns with this expansion:

<img src="{{ site.baseurl }}/img/clonealign/de_mhc-class-i.png" width="50%"/>

Remarkably, the entire MHC class-i complex is downregulated in clone A, a mechanism by which cells can hide proteins from cytotoxic T-cells known as _immune escape_. Therefore, the clonealign statistical framework enables us to associate the fitness landscape of tumour cells with functional changes to their transcriptional profile.

Clonealign is available as [an R package](https://www.gihub.com/kieranrcampbell/clonealign) and a (slightly outdated) [preprint](https://biorxiv.org/content/early/2018/06/11/344309).


