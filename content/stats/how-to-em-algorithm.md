---
title: "GMM -- An EM Example"
date: 2021-02-23T21:53:28Z
draft: true
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

Putting aside our Bayesian hats for the moment, let's go for step one of model fitting -- getting a MAP estimate (or even an MLE if the thought of a prior fills you with dread).
So we take a look at the form of the likelihood:

\begin{equation}
    f( x \vert \theta ) = \sum_k p_k f_{\mathcal{N}}(x \vert \mu_k, \Sigma_k)
\end{equation}

## EM + GMM ##

## 

