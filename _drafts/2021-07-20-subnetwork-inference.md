---
layout:      post
title:       "Bayesian Deep Learning via Subnetwork Inference"
tags:        [bayesian inference, deep learning]
authors:
    - name: Erik Daxberger
      link: https://edaxberger.github.io
    - name: Eric Nalisnick
      link: https://enalisnick.github.io
comments:   true
image:      /assets/images/subnetwork-inference/d_prediction.png
excerpt: |
    This blog post describes <i>subnetwork inference</i>, a Bayesian deep learning method where inference is done over only a small, carefully selected subset of the model weights instead of all weights. This allows using expressive posterior approximations (<i>e.g.</i> full covariance Gaussian distributions) that would otherwise be intractable.
---


## Motivation: Bayesian deep learning is important but hard

Despite their successes, accurately quantifying uncertainty in the predictions of DNNs is notoriously hard, especially if there is a shift in the data distribution between train and test time.
In practice, this might often lead to overconfident predictions, which is particularly harmful in safety-critical applications such as healthcare and autonomous driving.
One principled approach to quantify the predictive uncertainty of a neural net is to use _Bayesian inference_.

The standard practice in deep learning is to estimate the parameters using just a _single point_ found through gradient-based optimisation. In contrast, in Bayesian deep learning (check out [this blog post](https://jorisbaan.nl/2021/03/02/introduction-to-bayesian-deep-learning.html) for an introduction to Bayesian deep learning), the goal is to infer a _full posterior distribution_ over the model’s weights. By capturing a distribution over weights, we capture a distribution over neural networks, which means that prediction essentially takes into account the predictions of (infinitely) many neural networks. Intuitively, on data points that are very distinct from the training data, these different neural nets will disagree on their predictions. This will result in high _predictive uncertainty_ on such data points and therefore reduce overconfidence.

The problem is that modern deep neural nets are so big that even trying to approximate this posterior is highly non-trivial. We are not even talking about humongous 100-billion-parameter models like OpenAI’s GPT-3 here ([Brown _et al._, 2020](https://arxiv.org/abs/2005.14165)) --- even for a neural net with more than just a few layers it’s hard to do good posterior inference! Therefore, it’s becoming more and more challenging to design approximate inference methods that actually scale.

To cope with this problem, many existing Bayesian deep learning methods make very strong and unrealistic approximations to the structure of the posterior. For example, the common _mean field approximation_ approximates the posterior by a distribution which fully factorises over individual weights. Unfortunately, recent papers ([Ovadia _et al._, 2019](https://arxiv.org/abs/1906.02530); [Foong _et al._, 2019](https://arxiv.org/abs/1906.11537)) have empirically demonstrated that such strong assumptions result in bad performance on downstream tasks such as uncertainty estimation. Can we do better than this?
$\newcommand{\vw}{\mathbf{w}}$
$\newcommand{\D}{\mathcal{D}}$
$\newcommand{\c}{\textsf{c}}$


## Idea: Do inference over only a small subset of the model parameters!

Most existing Bayesian deep learning methods try to do inference over all the weights of the neural net. But do we actually need to estimate a posterior distribution over all weights? 

It turns out that you often don’t need all those weights. In particular, recent research ([Cheng _et al._, 2017](https://arxiv.org/abs/1710.09282)) has shown that, since deep neural nets are so heavily overparametrised, it’s possible to find a small subnetwork within a neural net containing only a very small fraction of the weights, which, miraculously, can achieve the same accuracy as the full neural net. These subnetworks can be found by so-called pruning techniques. 

{% include image.html
   name="Figure 1"
   ref="pruning"
   alt="An illustration of neural network pruning (<a href=\"https://arxiv.org/abs/1506.02626\">Han <i>et al.</i>, 2015</a>)."
   src="subnetwork-inference/pruning.png"
   width=700
%}

As shown in [Figure 1](#figure-pruning), pruning techniques typically first train the neural net, and then, after training, remove certain weights or even entire neurons according to some criterion. There has been a lot of recent interest in this research direction; for example, the best paper award at ICLR 2019 went to Jonathan Frankle and Michael Carbin’s now famous work on the lottery ticket hypothesis ([Frankle and Carbin, 2018](https://arxiv.org/abs/1803.03635)), which showed that you can even retrain the pruned network from scratch and still achieve the same accuracy as the full network.

But how does this help us? We asked ourselves the exact same question about the model uncertainty: Can a full deep neural net's model uncertainty be well-preserved by a small subnetwork’s model uncertainty? It turns out that the answer is yes, and in the remainder of this blog post, you will learn about how we came to this conclusion.

## Our proposed approximation to the posterior 

Assume that we have divided the weights $\vw$ into two disjoint subsets: (1) the subnetwork $\vw_S$ and (2) the set of all remaining weights $\\{\vw\_r\\}\_{r \in S^\c}$. We will later describe how we select the subnetwork; for now, just assume that we have it already. We propose to approximate the posterior distribution over as follows:
\begin{equation}
   p(\vw \cond \D)
   \overset{\text{(i)}}{\approx}
      p(\vw_S \cond \D)
      \prod_{r \in S^\c} \delta(\vw\_r - \hat{\vw}\_r)
   \overset{\text{(ii)}}{\approx}
      q(\vw_S)
      \prod_{r \in S^\c} \delta(\vw\_r - \hat{\vw}\_r)
   =: q\_S(\vw).
\end{equation}
The first step (i) of our posterior approximation then involves a posterior distribution over just the subnetwork $\vw_S$, and delta functions over all remaining weights $\\{\vw\_r\\}\_{r \in S^\c}$. Put differently, we only treat the subnetwork $\vw_S$ in a probabilistic way, and assume that each remaining weight $\vw_r$ is deterministic and set to some fixed value $\hat\vw_r$. Unfortunately, exact inference over the subnetwork is still intractable, so, in the second step (ii) of our approximation, we introduce an approximate posterior $q$ over the subnetwork $\vw_S$. Importantly, as the subnetwork is much smaller than the full network, this allows us to use expressive posterior approximations that would otherwise be computationally intractable (_e.g._ full-covariance Gaussians). That’s it.

There are a few questions that we still need to answer:

{: class="parentheses" }
1. How do we choose and infer the subnetwork posterior $q(\vw_S)$? That is, what form does $q$ have, and how do we infer its parameters?
2. How do we set the fixed values $\hat\vw_r$ of all remaining weights $\\{\vw\_r\\}\_{r \in S^\c}$?
3. How do we select the subnetwork $\vw_S$ in the first place?
4. How do we make predictions with this approximate posterior?
5. How does subnetwork inference perform in practice?

Let’s start with question (1).


