---
title: "GMM -- An EM Example"
date: 2021-02-23T21:53:28Z
draft: false
markup: pandoc
math: true
---

## Gaussian Mixture Model ##

Mixture models are one of the most simple examples of a latent variable model (by which I mean a model where some random variable, commonly refered to as $Z$, is not observable).
A _Gaussian Mixture Model_ (GMM) is one where our quantity of interest can be considered to follow a Normal distribution (which can be multivariate).
There's obviously something more going on -- the mixture part comes from the fact that the parameters for this Normal distribution aren't fixed for every observation we might make -- they are in fact drawn from a finite set.

More formally, we say that $X \sim GMM(p, \{ \mu_k \}, \{ \Sigma_k \})$ if:

\begin{align}
    Z &\sim \text{Categorical}(p) \\
    X &\sim \text{Normal}(\mu_Z, \Sigma_Z)
\end{align}

The intermediate variable $Z$ allows us to build multimodal distributions (something that would be very useful) from unimodal Normal distributions (something that is very well studied).
The only issue comes when we want to actually fit the thing.

Putting aside priors on parameters for the moment, let's assume that we want maximum likelihood estimates for our parameters $p, \mu_k, \Sigma_k$.
(We can do this by setting our prior term in our $Q$ function to a constant, and then realising that we can then just drop it for the purposes of maximisation.)

## EM + GMM ##

So we take a look at the form of the likelihood for a collection of data $\mathbf{X} = \left\{X_i : i \in \{1, \dots, n \}\right\}$, with associated latent variables $\mathbf{Z} = \left\{Z_i : i \in \{1, \dots, n \}\right\}$ and $\theta = \left\{ p_{ij}, \mu_j, \Sigma_j : i \in \{1, \dots, n\}, j \in \{ 1, \dots, k\}\right\}$:

\begin{align}
    f( \mathbf{X} \vert \theta, \mathbf{Z} ) &= \prod_i f_{\mathcal{N}}(x \vert \mu_{Z_i}, \Sigma_{Z_i}) \\
    f( \mathbf{Z} \vert \theta) &= \prod_i p_{iZ_i}
\end{align}

We then obtain our $Q$ function by taking the expectation of the sum of the logs of these two densities, where the expectation is with respect to $Z_i \vert \varphi, X_i$ for each $i$.
This is one of the most crucial steps of the algorithm, and it's really important to be clear about exactly what this distribution is.
In this case, as $Z$ can only take a finite number of values we have a simple expression for this from Bayes' rule:

\begin{align}
    \mathbb{P}( Z = j \vert X, \varphi)
    &= 
    \frac{ \mathbb{P}( Z = j, X, \varphi) }{ \mathbb{P}( X, \varphi) }\\
    &=
    \frac{ \mathbb{P}( X \vert Z =j, \varphi) \mathbb{P}( Z = j \vert \varphi) }
         { \sum_{m} \mathbb{P}( X \vert Z = m, \varphi) \mathbb{P}( Z = m \vert \varphi) }\\
\end{align}

We're going to call these reweightings (which are _not_ the same as the $p^{\varphi}_{ij})$ $q^{\varphi}_{ij}$. The above densities are simply normal and categorical from the definition of the model.

\begin{align}
Q(\theta, \varphi) 
&=
\mathbb{E}_{\mathbf{Z} \sim Z \vert \varphi, X}
\left[
\sum_i 
    - \frac{1}{2} \log \left( (2 \pi)^d \det\left(\Sigma^\theta_{Z_i}\right) \right)
    - \frac{1}{2} (x_i - \mu^\theta_{Z_i})^T \Sigma^{{-1}^{\theta}}_{Z_i} (x_i - \mu^\theta_{Z_i})
    + \log p^\theta_{iZ_i}
\right]
\\
&=
\sum_i
\sum_j
    - q^\varphi_{ij} \frac{1}{2} \log \left( (2 \pi)^d \det\left(\Sigma^\theta_j\right) \right)
    - q^\varphi_{ij} \frac{1}{2} (x_i - \mu^\theta_j)^T \Sigma^{{-1}^\theta}_j (x_i - \mu^\theta_j)
    + q^\varphi_{ij} \log p^\theta_{ij}
\end{align}

I've add superscripts to each of the variables so that we can see which parameter estimate $\theta$ or $\varphi$ it is associated with.
The task is now to maximise $Q$ with respect to the $\theta$ parameters, which happily we can do analytically[^analytic-max] (using $\hat\cdot$ to indicate the maximum).

[^analytic-max]: We find the maximum of this using exactly the same techniques that we exploit to find maximum likelihood estimates -- namely differentiation, log-concavity and the use of a Lagrange multiplier to enforce the sum-to-one constraint on the $p_{ij}$.

\begin{align}
\hat{\mu}^\theta_j &= \frac{\sum_i p^\varphi_{ij} x_i}{\sum_i p^\varphi_{ij}} \\
\hat{\Sigma}^\theta_j &= \frac{\sum_i p^\varphi_{ij} ( x_i - \hat{\mu}^\theta_j ) (x_i - \hat{\mu}^\theta_j)^T  }{ \sum_i p^\varphi_{ij} } \\
\hat{p}^{\theta}_{ij} &= q^{\varphi}_{ij}
\end{align}

Now at each iteration of our algorithm, we can use the parameters of the current step to generate _better_ parameters (in the sense of maximising the likelihood) according to the update step above -- if we say $\varphi = \theta_t$, then $\theta_{t+1} = \hat{\theta}$ in the above relations.

There is a neat interpretation to this update step where we're performing a sort of weighted maximum likelihood estimate for the parameters of the normal distribution based on the cluster assignments of the previous step, and then updating our cluster assignments for the new step based on the old weighted cluster assignment -- there's a sort of lock step where the two sets of parameters are constantly playing catch up with each other to best fit the data.

## What else? ##

This example covered a classic "latent variable" type use of the EM algorithm, but it also works really well in situations where you have partially or unobserved data -- in this case the $Z$ values become a sort of obscured $X$ value, and can be really useful in cases where your measurement of a system imposes some kind of censoring on the values.