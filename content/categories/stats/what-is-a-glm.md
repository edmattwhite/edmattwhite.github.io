---
title: "What is a Generalised Linear Model?"
date: 2021-11-08T21:38:24Z
draft: false
markup: pandoc
math: true
---

Generalised Linear Models (GLMs) are a natural extension of simple normal linear regression to scenarios involving different kinds of responses and different assumptions.

## What is being extended? ##

Linear Regression (or more precisely, the Normal Linear Model) is a very elegant tool for predicting real valued responses given some covariates, or features, that you believe have some ability to explain the trends that you see in your response.
The actual model for your outcomes $Y \in \mathbb{R}^n$ given features $X \in \mathbb{R}^{n \times p}$ can be concisely expressed as
\begin{equation}
Y \sim \text{Normal}\left(X\beta, \sigma^2 I_n \right),
\end{equation}
where $I_n$ is the $n$-dimensional identity matrix, $\sigma^2$ is some unknown variance parameter, and $\beta$ is an unknown parameter defining how your data relates to the expectation of $Y$ (hint: _linearly_).

This is a really powerful way of constructing models, and can get a _very long way_ to providing excellent solutions to complex problems involving large datasets today.
However, you may have noticed that there is a slight problem here -- the use of a normal distribution encodes an assumption about the range of the random variable $Y$ and the distribution of our errors around the expected response values.
Using the model we can produce continuous estimates for $Y_i$ in the whole of $\mathbb{R}$.
This clearly makes no sense if we want to predict a proportion ($Y_i \in [0, 1]$), a count ($Y_i \in \mathbb{Z}$) or a length ($Y_i \in \mathbb{R^+}$).

## What does a GLM do? ##

The Generalised Linear Model extends the Normal Linear Model in two specific ways:

1. Stochastic: relaxes the assumption that the stochastic component of the model should always be a Normal distribution -- instead it allows us to use any member of the exponential family[^1] of distributions (both continuous and discrete).
1. Deterministic: the mean of our outcome no longer depends just on our linear term $X\beta$, it is now allowed to be a monotonic, differentiable function of this quantity.

[^1]: _Generalized Linear Models with R_, Peter K. Dunn & Gordon K. Smyth, Springer, 2018

What I found very confusing when first reading about this concept was disentangling the fact that these are two separate assumptions.
We will see that whilst the family of distributions for the stochastic part of the model give a very natural choice for the function that gives the mean response, it is _not_ fixed to be this, and keeping them separate will make picking up the theory a little easier to begin with.

## The Stochastic Part ##

Each element of the response, $Y_i$, is independent of the other components and has distribution in the same family (i.e. the same "type" of random variable, with a different parametrisation given by the associated observed data).
This family must be an exponential family, i.e. its density or mass function takes on the form [^2]:
\begin{equation}
f(y_i \vert \theta_i, \varphi) = \exp \left( \frac{y\theta_i - b(\theta_i)}{a(\varphi)} + c(\theta_i, \varphi) \right)
\end{equation}
$a$, $b$ and $c$ are simply real valued functions.
We refer to $\varphi$ as the dispersion parameter, and $\theta_i$ as the canonical or natural parameter.
It is normal to assume that $\varphi$ is a nuisance parameter shared across different observations (in the same way that a Normal linear model has a shared variance for each of the error terms), but we allow the $\theta_i$ to be different for each observation.

This assumption on the form of the density might look unusual and quite restrictive at first, but in actual fact the Normal distribution, Poisson, Binomial, Gamma, etc. can be coerced to this form.
This may require fixing something that we previously took as a parameter to be constant, like the number of trials for the Binomial.

[^2]: There are a few different formulations of an exponential family distribution -- this one makes the set up for GLMs easy. It also leaves out the condition that the sample space can't depend on parameters, but that's a bit in the weeds if you just want to get started with GLMs.

## The Deterministic Part ##

This part is usually described using a parameter $\eta$, where $\eta_i = X_{ij}\beta_j$ (just a linear combination of observations with parameters $\beta$).
It's not immediately obvious how to relate our probabilistic component to our $\eta_i$ -- in the Normal linear model, we do this through the mean parameter $\mu_i = \eta_i$.
This doesn't work if we're trying to perform regression on observations only taking on values in $[0, 1]$, as we described previously.
To get around this, we introduce a link function $g: \mathcal{Y} \rightarrow \mathbb{R}$, where $\mathcal{Y}$ is the domain of our data, so that:
\begin{equation}
g(\mu_i) = \eta_i
\end{equation}
We require that $g$ is monotonic and differentiable in order for the model that we put together to work.

## Is that it? ##

Yes, it's as simple as that -- defining a probability distribution and a link function fully describes a Generalised Linear Model.

## What is a canonical link function? ##

The canonical link function is one that identifies the linear combination of data and parameters $\eta_i$, with the canonical or natural parameter $\theta_i$:
\begin{equation}
\eta_i = \theta_i
\end{equation}
There is also a much less obvious (but entirely equivalent statement), that the canonical link function is given by:
\begin{equation}
g\left(b'(\theta)\right) = \theta
\end{equation}
The canonical link function is often a sensible choice -- a default that you can choose to override, but should really think hard about why it is that you're doing so before you do it.

## How do I fit one? ##

If you're lucky enough to be working in _R_, GLMs come baked into the standard library, and can be fit [^3] like this:
```R
glm(y ~ x, family = poisson(link = "log"))
```
This even lets you specify a link function of your own!
It's worth noting that unlike a Normal linear model, the GLM has to be fit with an approximate, iterative procedure called Iterated Weighted Least Squares (IWLS), which is essentially Newton's method.
This procedure actually makes use of fitting lots of linear models with least squares under the hood, and is quite simple to implement once you have your model written down, making GLMs very approachable even if your language of choice doesn't have them by default!
If you want to find out how to do this, I'd really recommend Dunn and Smyth, which you'll find as a reference below.

[^3]: This fit is maximum likelihood estimation of the parameters $\beta$ -- the link between the least squares estimate and the MLE is broken for the general exponential family.