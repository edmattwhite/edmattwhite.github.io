---
title: "What is the Expectation Maximisation algorithm?"
date: 2021-02-22T20:33:41Z
draft: false
markup: pandoc
math: true
---

Powerful statistical models are often useful because they are very expressive.
The structure of the model can lend itself well to describing a real world process.
Unfortunately, real world processes are incredibly rude and often lead to intractable likelihoods.
The Expectation Maximisation (EM) algorithm offers us a way around this by giving an iterative procedure for find MAP (or maximum likelihood) estimates for parameters.

One thing to consider when reading this is that the EM algorithm is less an algorithm in itself, but more a method to find algorithms.

## What's the problem? ##

Let's say we have a statistical model for some observed data $X$, and that statistical model is characterised by densities $p$.
This statistical model involves some **latent** or unobserved random variables $Z$, and some parameters $\theta$, which as we're good Bayesians, we will suppose are also random variables.

We wish to find a MAP estimate for $\theta$. That is, we want to find $\hat{\theta}$ that satisfies:

\begin{align}
\hat{\theta} 
&= \text{argmax}_{\theta} \left\{ p \left( \theta \vert X \right) \right\} \\
&= \text{argmax}_{\theta} \left\{
     \frac
     {p \left( X \vert \theta \right) p \left( \theta \right)}
     {p \left( X \right) }
\right\} \\
&= \text{argmax}_{\theta} \left\{ p \left( X \vert \theta \right) p \left( \theta \right) \right\}
\end{align}

The last equality follows from the denominator having no dependence on $\theta$.
Having written down our expression for $\hat{\theta}$ we can see the problem straight away; the data likelihood term -- $p(X \vert \theta)$ -- has no dependence on $Z$.
We don't have expressions for this objects in our model, as we need explicit values for $Z$ to be able to calculate probabilities of our observed variable.
If we rewrite our expression for $\hat{\theta}$ in terms of quantities that we have access to:

\begin{align}
\hat{\theta} 
&= \text{argmax}_{\theta} \left\{
     {\int p \left( X, Z \vert \theta \right) p \left( \theta \right) dz }
\right\}
\end{align}

We've now got another problem -- integrating over the domain of $Z$ is going to be impossible for all but trivial $Z$.

## What's the solution? ##

If we've given up on directly calculating our MAP estimate, maybe it would be nice to have an iterative algorithm that gets us closer and closer to a good estimate for $\hat{\theta}$.
If we're going to create an algorithm that does this for us, we're going to want some way of assessing how close we are getting to the truth.
Considering that, let's try and at least write down an integral free expression with our posterior involving our latent variable:

\begin{align}
p( \theta \vert X )
&= \frac
        { p( X, \theta )}
        { p(X) }\\
&= \frac
        { p( X, \theta ) }
        { p(X) }
   \frac
        { p( X, Z, \theta ) }
        { p( X, Z, \theta ) }\\
&= \frac
        { p( Z, \theta \vert X) }
        { p( Z \vert \theta, X ) }
\end{align}

It's easy to lose track, but here our observable $X$ and parameters $\theta$ are held fixed, letting our Z randomly vary.
If we move this to log space, we just get:

\begin{equation}
\log p( \theta \vert X ) = \log p( Z, \theta \vert X) - \log p( Z \vert \theta, X )
\end{equation}

Now we've got our first trick of this algorithm -- we consider another value for our parameters, which we'll call $\varphi$.
If we were to fix this parameter value $\varphi$, then we could reasonably infer a probability distribution over $Z$ conditioned on this value and the observable data, which we'll denote with $Z \sim Z \vert \varphi, X$.
A bigger leap is that we can then take expectations of our last expression using this probability distribution, which is fine as it's just another constant.

\begin{align}
\mathbb{E}_{Z \sim Z \vert \varphi, X}\left[ \log p( \theta \vert X ) \right]
&=
\mathbb{E}_{Z \sim Z \vert \varphi, X}\left[ \log p( Z, \theta \vert X) \right]
-
\mathbb{E}_{Z \sim Z \vert \varphi, X}\left[ \log p( Z \vert \theta, X ) \right]
\\
\Rightarrow
\log p( \theta \vert X ) 
&=
\mathbb{E}_{Z \sim Z \vert \varphi, X}\left[ \log p( Z, \theta \vert X) \right]
-
\mathbb{E}_{Z \sim Z \vert \varphi, X}\left[ \log p( Z \vert \theta, X ) \right]
\end{align}

We've been able to simplify because the expectation is taken over a constant value.
The next trick is to notice that this expression is true **for any** $\theta$ and $\varphi$.
So we could set $\theta = \varphi$, and see what happens when we take the difference of that expression with one involving $\theta$ and $\varphi$.

\begin{align}
\log p( \theta \vert X )
-
\log p( \varphi \vert X )
&=
\mathbb{E}_{Z \sim Z \vert \varphi, X}\left[ \log p( Z, \theta \vert X) \right]
-
\mathbb{E}_{Z \sim Z \vert \varphi, X}\left[ \log p( Z, \varphi \vert X) \right]\\
&+
\mathbb{E}_{Z \sim Z \vert \varphi, X}\left[ \log p( Z \vert \varphi, X ) \right]
-
\mathbb{E}_{Z \sim Z \vert \varphi, X}\left[ \log p( Z \vert \theta, X ) \right]\\
&=
\mathbb{E}_{Z \sim Z \vert \varphi, X}\left[ \log p( Z, \theta \vert X) \right]
-
\mathbb{E}_{Z \sim Z \vert \varphi, X}\left[ \log p( Z, \varphi \vert X) \right]\\
&+
D_{KL} \left[ p( Z \vert \varphi, X ) \Vert p( Z \vert \theta, X ) \right]
\\
\end{align}

Where $D_{KL}$ is the Kullback-Leibler divergence.
Looking at this we're in a nice position to start making statements about how good our estimates might be, with a mixture of quantities we're interested in, quantities we can calculate and quantities that we can bound.
To make this a bit more explicit, let's make one more definition:

\begin{align}
Q(\theta, \varphi) 
&= \mathbb{E}_{Z \sim Z \vert \varphi, X}
\left[
         \log p( X \vert Z, \theta)
         +
         \log p( Z \vert \theta)
         +
         \log p(\theta)
\right] \\
&= \mathbb{E}_{Z \sim Z \vert \varphi, X}
\left[
         \log p( Z, \theta \vert X)
         + 
         \log p(X)
\right]
\end{align}

Our previous expression now simplifies down:

\begin{equation}
\log p( \theta \vert X )
-
\log p( \varphi \vert X )
=
Q(\theta, \varphi)
-
Q(\varphi, \varphi)
+
D_{KL} \left[ p( Z \vert \varphi, X ) \Vert p( Z \vert \theta, X ) \right]
\end{equation}

$D_{KL}(p \Vert q) > 0$ for any two probability distributions.
This gives us an equality linking our posteriors and $Q$:

\begin{equation}
\log p( \theta \vert X )
-
\log p( \varphi \vert X )
\geq
Q(\theta, \varphi)
-
Q(\varphi, \varphi)
\end{equation}

This one line gives us the bones of the EM algorithm.
Suppose that $\varphi$ is our current estimate for the MAP.
If we pick a new estimate $\theta$ that maximises $Q(\theta, \varphi)$, then we're guaranteed to increase the log posterior by **at least** as much as we increase $Q(\theta, \varphi)$ over $Q(\varphi, \varphi)$.
Our inequality tells us that we can keep improving our MAP estimate, but we can't be sure that we get to the true MAP.
We could get stuck in some local maxima, but that's often something that we'll be willing to accept.

## What are we missing? ##

I've really just asserted that $Q$ is going to be helpful, as at the moment the expectation that we're taking is still running over all possible states.
In real examples we can often eploit model structure and the linearity of expectation to turn these sums into something more tractable.
It'll be easier to understand this in [an example]({{% ref "categories/stats/how-to-em-algorithm" %}}).