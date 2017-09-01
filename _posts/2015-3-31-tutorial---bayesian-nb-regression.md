---
layout: post
title: "Tutorial - Bayesian negative binomial regression from scratch in python"
tags:
    - python
    - notebook
---
The [negative binomial
distribution](http://en.wikipedia.org/wiki/Negative_binomial_distribution) crops
up a lot in computational biology, and in particular RNA-sequencing analysis. In
an ideal world we might expect the distribution of RNA-seq reads to be poisson,
where the variance equals the mean and the only error comes from sampling alone.
However, RNA-seq reads typically display more dispersion than this, which makes
the negative binomial (NB) distribution a good choice since it can be thought of
as an overdispersed poisson (or a poisson with a gamma-distributed rate).

Here we consider Bayesian negative binomial regression of the form

$$ y_i \, \; | \, \; \beta_0, \beta_1, x_i, r \sim NB(\mu_i = \beta_0 + \beta_1
x_i, r) $$

where $$r$$ is known as the _dispersion_ parameter, since the variance may be
written as $$ Var(y_i) = \mu_i + \frac{\mu_i^2}{r} $$. We will implement
Bayesian inference using metropolis-hastings with the parameters $$\beta_0 =
10$$, $$\beta_1 = 5$$ and $$r = 10 $$.

**In [1]:**

{% highlight python %}
""" some setup """

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.stats import truncnorm
import seaborn as sns
%matplotlib inline

sns.set_palette("hls", desat=.6)
sns.set_context(rc={"figure.figsize": (6, 4)})
np.random.seed(123)
{% endhighlight %}

### A brief recap of Metropolis-Hastings and approximate inference

Parameter inference in a Bayesian setting is often concerned with estimating
some posterior density $$p(x)$$. However, if the prior and likelihood are not
conjugate to each other then there is no closed-form solution for the posterior
as the normalisation factor is intractable. Here the posterior distribution
takes the form $$p(x) = p^\star(x)/Z$$, where we can easily compute $$
p^\star(x) $$ but not $$Z$$.

In such cases we can use the _Metropolis-Hastings_ algorithm. This builds up a
Markov-chain of the variable $$x$$, the samples of which converge towards the
posterior distribution of $$p(x)$$. It works as follows:

 1. Sample $$x'$$ from the proposal distribution $$ q(x' | x) $$. This is
typically a Gaussian distribution centred around $$x$$.
 2. Compute the _acceptance ratio_
$$ \alpha = \frac{p(x')q(x | x')}{p(x)q(x'|x)} $$
which is essentially the ratio of the probabilities of $$x'$$ to $$x$$
normalised by the probability of being there in the first place (i.e. the ratio
of $$q$$). This last point is important as it is easy to get this the wrong
way round.
 3. Accept $$x'$$ as the next sample in the Markov chain with probability $$
\mathrm{min}(1, \alpha ) $$, otherwise use $$x$$. In other words, if $$x'$$
is more probable then always accept it, and even if it is less probable still
accept it with probability $$ \alpha $$.


First we can generate 150 data points and visualise them:

**In [2]:**

{% highlight python %}
""" simulate some data """
beta_0 = 10
beta_1 = 5

N = 150
x = np.random.randint(0, 50, N)

true_mu = beta_0 + beta_1 * x
true_r = 10
p = 1 - true_mu / (float(true_r) + true_mu)

y = np.random.negative_binomial(n = true_r, p = p, size = N)

plt.scatter(x, y, color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.title('150 points generated')
{% endhighlight %}

 


    <matplotlib.text.Text at 0x1154d39d0>




![png]({{ site.baseurl}}/img/negbin/negbin1.png)


### Define acceptance ratio and proposal distributions:

In order to implement [metropolis-
hastings](http://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) we
need to define proposal distributions for our parameters and compute the
acceptance ratio for each move. The proposal distributions for $$\beta$$ is
just a normal centred around the current value, while that for $$r$$ is a
truncated normal centred around the current value. Since these distributions are
symmetric (i.e. $$ N(x\,|\,x') = N(x' \, | \, x) $$ ) they do not factor into
the acceptance ratio.

For now we place uninformative priors on all the variables, though these can
easily be implemented by modifying the acceptance ratio and including hyper-
parameters. The regression can also easily be extended to more than one
independent variable.

Two parameterizations of the negative binomial distribution exist: $$ (n,p) $$
as per wikipedia and the mean version used for regression with $$ (\mu,r) $$.
The equivalence is that $$ r=n $$ and $$ \mu = \frac{pr}{1-p} $$ (though
careful how $$p$$ is defined). It can be shown that the log acceptance ratio
is given by

$$
\begin{array}{r}
\alpha = \sum_i \left[ r'\ln\left(\frac{r'}{r' + \mu_i'}\right)  -
r\ln\left(\frac{r}{r + \mu_i}\right) + \ln\Gamma(r' + x_i) -\ln\Gamma(r + x_i) +
x_i\ln\left(\frac{\mu_i'}{\mu_i}\frac{\mu_i + r}{\mu'_i + r'}\right) \right] +
\\
N \left(\ln\Gamma(r) -\ln\Gamma(r')\right)
\end{array}
$$

where

$$ \mu_i = \beta_0 + \beta_1 x_i $$

We also need to define the sampling distributions $$q(x'|x) $$. Since the
parameters are independent we can factorise the expression to

$$ q(r', \beta_0', \beta_1' \, | r, \beta_0, \beta_1) = q_r(r' \, | r)
q_b(\beta_0' \, | \beta_0) q_b(\beta_1' \, | \beta_1)$$

For $$q_b$$ we just choose a normal distribution centred around the current
value (i.e. $$ q_b \sim N(\beta, \sigma) $$ ), while for $$q_r$$ we choose a
normal distribution truncated at 0, since the dispersion parameter is strictly
positive.

The art of Metropolis-hastings MCMC comes from choosing the variances of the
proposal distributions. Too high and the probability of the proposed point will
be very small and almost never accepted. Too low and the proposed point will
almost always be accepted and it takes far too long for the random walk to fill
out the posterior. What MCMC needs is the goldilocks zone - getting the
variances just right.

There is a huge body of literature into _adaptive_ MCMC algorithms - those that
find the optimal parameters automatically (see e.g.
[here](http://probability.ca/jeff/ftpdir/adaptex.pdf)). However, for small
problems such as this it can be easier just to play around with the parameters
until we get an _acceptance rate_ (proportion of proposed points accepted) to
the desired value. Somewhat amazingly [it has been
shown](http://projecteuclid.org/euclid.aoap/1034625254) that the optimal
acceptance ratio is around 0.44 in a one-parameter problem, and asymptotically
approaches 0.234 as the number of parameters increases. Therefore, in our case
an acceptance ratio in that region would be desirable.

**In [3]:**

{% highlight python %}
def nb_acceptance_ratio(theta, theta_p, y, N):
    """ theta = (mu, r), y is data, N = len(x) """
    mu, r = theta
    mu_p, r_p = theta_p
    
    term1 =  r_p * np.log(r_p / (r_p + mu_p))
    term2 = -r * np.log(r / (r + mu))

    term3 = y * np.log(mu_p / mu * (mu + r)/(mu_p + r_p))
    
    term4 = gammaln(r_p + y)
    term5 = - gammaln(r + y)
    
    term6 = N * (gammaln(r) - gammaln(r_p))
    
    return (term1 + term2 + term3 + term4 + term5).sum() + term6

""" proposal from previous blog post """
def truncnorm_prop(x, sigma): # proposal for r (non-negative)
    return truncnorm.rvs(-x / sigma, np.Inf, loc=x, scale=sigma)
{% endhighlight %}

Before we're ready to run the sampling loop there are two further parameters we
need to consider: the burn in of the trace and any thinning we'd like. The burn
in is the number of samples to discard at the start of an MCMC run. While
typically taken to be half of the samples in total, graphs of autocorrelation
can be used to assess the point at which the autocorrelation of the markov chain
reaches zero. The second parameter is the thinning, which tells us the
proportion of samples to remove from the trace, so if _thin_ = 5 then only every
fifth sample is used, which again removes autocorrelation.

#### MCMC loop:

**In [4]:**

{% highlight python %}
n_iter = 10000
burn_in = 4000

sigma_beta_0 = 0.5
sigma_beta_1 = 0.7
sigma_r = 0.5

def calculate_mu(beta, x):
    return beta[0] + beta[1] * x

def metropolis_hastings(n_iter, burn_in, thin=5):
    
    trace = np.zeros((n_iter, 3)) # ordered beta_0 beta_1 r
    trace[0,:] = np.array([5.,5.,1.]) 
    acceptance_rate = np.zeros(n_iter)
    # store previous mu to avoid calculating each time 
    mu = calculate_mu(trace[0,0:2], y) 
    
    for i in range(1, n_iter):
        theta = trace[i-1,:] # theta = (beta_0, beta_1, r)
        
        theta_p = np.array([np.random.normal(theta[0], sigma_beta_0), 
                            np.random.normal(theta[1], sigma_beta_1), 
                            truncnorm_prop(theta[2], sigma_r)]) 
        
        mu_p = calculate_mu(theta_p[0:2], x)
        
        if np.any(mu <= 0):
            print "mu == 0 on iteration %d" % i

        alpha = nb_acceptance_ratio((mu, theta[2]), 
                                    (mu_p, theta_p[2]), y, N)
        
        u = np.log(np.random.uniform(0., 1.))

        if u < alpha:
            trace[i,:] = theta_p
            mu = mu_p
            acceptance_rate[i-1] = 1
        else:
            trace[i,:] = theta
        
            
    print "Acceptance rate: %.2f" % acceptance_rate[burn_in:].mean()
    return trace[burn_in::thin,:]

trace = metropolis_hastings(n_iter, burn_in)
{% endhighlight %}

    Acceptance rate: 0.24


#### Plot traces:

**In [5]:**

{% highlight python %}
import pylab 
pylab.rcParams['figure.figsize'] = (12., 8.)

plt.subplot(3,2,1)
plt.plot(trace[:,0])
plt.title("beta_0")

plt.subplot(3,2,2)
plt.hist(trace[:,0])

plt.subplot(3,2,3)
plt.plot(trace[:,1])
plt.title("beta_1")

plt.subplot(3,2,4)
plt.hist(trace[:,1])

plt.subplot(3,2,5)
plt.plot(trace[:,2])
plt.title("r")

plt.subplot(3,2,6)
plt.hist(trace[:,2])
{% endhighlight %}

 


    (array([   4.,   23.,   97.,  169.,  276.,  209.,  187.,  135.,   79.,   21.]),
     array([  7.07102198,   7.77737246,   8.48372294,   9.19007342,
              9.8964239 ,  10.60277438,  11.30912486,  12.01547534,
             12.72182582,  13.4281763 ,  14.13452678]),
     <a list of 10 Patch objects>)




![png]({{ site.baseurl}}/img/negbin/negbin2.png)


On the left hand side we can see the traces of the Markov-chain, while on the
right we can see the histogram of the posterior. So it seems the posterior
captures the true values quite well, given the noise.

The full IPython notebook for this tutorial can be found
[here](http://nbviewer.ipython.org/github/kieranrcampbell/blog-notebooks/blob/ma
ster/Tutorial%20-%20bayesian%20negative%20binomial%20regression%20from%20scratch
%20in%20python.ipynb).
