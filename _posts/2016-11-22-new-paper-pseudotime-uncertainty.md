---
layout: post
title: "New paper: Uncertainty in single-cell pseudotime"
tags:
    - Gaussian processes
    - scRNA-seq
    - Bayesian inference
---

Today our paper examining the effects of uncertainty in single-cell pseudotime was [published in PLOS Computational Biology](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005212). When single cells have their gene expression measured they are often undergoing some process of interest, such as differentiation, that ideally we would like to track over time to help us understand the dynamic cellular activity. However, the destructive nature of RNA sequencing means we can only ever measure each cell once. Crucially, because the cells develop asynchronously, the exact time of each cell is unknown. Several methods have been proposed to computationally re-infer these times as a _pseudotime_. However, we can never recover the true ordering of cells but only estimate an ordering that is consistent with the observed gene expression data.


Such methods to re-infer pseudotimes from single-cell gene expression data - also known as trajectory inference algorithms - have become increasingly popular with [over 20 different methods](https://docs.google.com/spreadsheets/d/1n3hXzzhrHZgClLD8P3cyIrK_6YgdjtdGlLswNAuoKSI/edit) developed at time of writing. Many trajectory inference algorithms  (such as [Monocle](http://www.nature.com/nbt/journal/v32/n4/abs/nbt.2859.html), [TSCAN](http://nar.oxfordjournals.org/content/early/2016/05/13/nar.gkw430.abstract) and [Waterfall](http://www.cell.com/cell-stem-cell/abstract/S1934-5909(15)00312-4)) begin with dimensionality reduction on the full gene expression matrix. This aims to compress the information contained across several thousand genes into two or three “components”, which is possible due to large correlations in gene behaviour across the transcriptome.  These algorithms then proceed to fit a one-dimensional “trajectory” in the reduced space using methods such as minimum spanning trees or principal curves. With the exception of [DeLorean](https://academic.oup.com/bioinformatics/article/32/19/2973/2196633/Pseudotime-estimation-deconfounding-single-cell?searchresult=1), the vast majority of these methods can be thought of as _deterministic_: no uncertainty is associated with each estimate of pseudotime and there is no discussion that such estimates are derived from inherently noisy data and are therefore themselves uncertain.


![Probabilistic pseudotime]({{ site.baseurl }}/img/pseudogp/fig2.png)

*Probabilistic pseudotime inference. Starting with a compressed representation of the data, we fit a Bayesian Gaussian Process Latent Variable Model and propagate the uncertainty to downstream analysis.*


To investigate the effects of such uncertainty, we replaced the one-dimensional trajectory reconstruction step with probabilistic curves using [Bayesian Gaussian Process Latent Variable Models (B-GPLVM)](http://jmlr.csail.mit.edu/papers/volume6/lawrence05a/lawrence05a.pdf). B-GPLVM is a probabilistic, nonlinear dimensionality reduction method that in our case learns a one-dimensional trajectory through the two-dimensional representation of the expression data giving the full posterior distribution of the pseudotimes for each cell.  Pseudotimes are estimated from noisy data and are therefore random variables themselves but are often treated as fixed quantities in analyses. This is problematic as a cell could be assigned to half way along a trajectory, but if we have no handle on the uncertainty of this estimate in reality it could be anywhere! We were then able to propagate this uncertainty through to standard downstream analyses, such as differential-expression-across-pseudotime. 

![Pseudotime differential expression FDR]({{ site.baseurl }}/img/pseudogp/fig6c.png){:  width="350px"}

*Taking uncertainty in pseudotime into account reveals approximately half of genes found as differentially expressed are 'unstable' to the underlying pseudotime ordering.*


We found significant posterior uncertainty in pseudotime estimates, with the 95% credible interval (the Bayesian version of confidence intervals) covering up to a quarter of the trajectory. By using posterior pseudotime traces from the MCMC output we fit multiple differential expression models for each gene which allowed us to identify which genes were ‘robust’ and which were ‘unstable’ to the uncertainty in the pseudotimes. We found that in the three datasets studied approximately half the genes were unstable in their differential expression across pseudotime. In other words, not accounting for uncertainty in pseudotime vastly increases the false discovery rate! This means that if differential-expression-across-pseudotime analysis is performed with no regard to the uncertainty then up to half the genes found to be significant may not actually be involved in the process of interest.


{: .center}
![Switch-like expression across pseudotime]({{ site.baseurl }}/img/pseudogp/example_sigmoid.png){:  width="400px"}

*Switch-like differential expression over pseudotime allows us to infer switching times and strengths, implemented in the switchde package.*


Finally, we introduced a separate differential expression model that identifies “switch-like” behaviour in genes across pseudotime. This model returns an estimate of where in the trajectory a given gene is up or down regulated. If this is fit across multiple MCMC traces, we can then infer the posterior distribution with uncertainty of when a gene is regulated, which allows us to say with some confidence that one gene is regulated before another. This differential expression model is implemented in the R package switchde, available on both [github](https://www.github.com/kieranrcampbell/switchde) and on [Bioconductor](https://bioconductor.org/packages/release/bioc/html/switchde.html). Such ideas are expanded upon in our method [Ouija](https://github.com/kieranrcampbell/ouija/) where we directly derive (in a fully probabilistic way) the pseudotime estimates from the switching behaviour of the genes.


<font color="grey"> All images are licenced under the Creative Commons CC-BY licence. Figures 1 & 2 can be found on the <a href="http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005212">PLOS website</a>.
</font>


