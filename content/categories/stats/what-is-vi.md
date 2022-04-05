---
title: "What is Variational Inference?"
date: 2022-03-29T15:51:29+01:00
draft: false
markup: pandoc
math: true
---

Lots of problems have the property of being harder to solve in one direction than another -- just imagine being asked to find the square root of a random integer compared with being asked to square one. 
We can also think about integration and differentiation as being a forward/inverse pair with an imbalance in their evaluation.
In general, constructing a derivative can be done very easily using just a handful of simple rules.
Integration, on the other hand, is much harder, and you may not even be able to find a closed form solution in what looks like a relatively simple problem.

This has a bearing on how we talk about probabilistic modelling -- marginalisation is an integration operation, and is therefore difficult.
Bayesian analysis has to grapple with this problem of being presented with integrals that are generally speaking intractable.
Suppose that we have a latent variable $Z$ that we want to calculate the posterior for given data $X$, and that we have densities $p$, then we see:

\begin{equation}
 p(z \vert x) 
 = \frac{p(x | z)p(z)}{p(x)}
 = \frac{p(x | z)p(z)}{\int p(x | z')p(z') dz'}.
\end{equation}

Our posterior calculation involves an integral over the entirety of the domain of $Z$ -- which even for fairly low dimensional problems can be infeasible.
Note that this is _similar_ to the [EM algorithm]({{% ref "categories/stats/what-is-em-algorithm" %}}), in that our initial approach has been blocked by a difficult integral.

## Variational Inference to the Rescue ##

At its core, variational inference is a neat trick for turning a hard problem (integration) into an easy one (differentiation).
This trade has to come at a cost -- we move away from the world of exact inference, and accept that we will approximate a distribution of interest.

We do this by introducing a 'variational family' of distributions that we allow our approximation to be drawn from -- in practice this can be a reasonable assumption, but care has to be taken in situations where these approximations are fed in to other machinery.
The family that we use to approximate this distribution, the 'variational family' will have the feature that we can index it with some parameter $\varphi$, so that changing $\varphi$ allows us to smoothly traverse the entire set of distributions in that family.
The associated density is usually denoted $q_{\varphi}$ -- note that we've dropped the $x$ from the dependence on $x$ in the argument in the density, as in our approximation dependence on $X$ passes through $\varphi$.

If we want to now find an optimal $\varphi$ we need to find some useful objective.
We know the model evidence $p(x)$ is fixed, so we proceed by suggestively playing around with that quantity:

\begin{align}
\log p(x) 
&= \log p(x) + \log p(z \vert x) - \log p(z \vert x)\\
&= \log p(z, x) - \log p(z \vert x)\\
&= \log p(z, x) - \log q_\varphi(z) + \log q_\varphi(z) - \log p(z \vert x)
\end{align}

All we've done so far is throw in a load of quantities that are related to things that we're interested in.
The next step just involves taking the expectation with respect to the variational distribution $q_\varphi$, and using the definition of KL divergence to get:

\begin{align}
\log p(x) 
&= \mathbb{E}_{q_\varphi}\left[\log p(z, x) - \log q_\varphi(z) + \log q_\varphi(z) - \log p(z \vert x) \right] \\
&= \mathbb{E}_{q_\varphi}\left[\log p(z, x) - \log q_\varphi(z) \right] + D_{KL}\left( q_\varphi \Vert p(z \vert x) \right)
\end{align}

This is now a familiar story if you've seen the EM algorithm's derivation -- the left hand side is fixed, so we know that increasing one quantity on the right hand side will decrease the other and vice versa.
The strange quantity $\mathbb{E}_{q_\varphi}\left[\log p(z, x) - \log q_\varphi(z) \right]$ is called the ELBO, `short' for Evidence Lower BOund.
Maximising this quantity will minimise the divergence between our approximation of the posterior and the true posterior, and that is all that variational inference (in its various forms) does.