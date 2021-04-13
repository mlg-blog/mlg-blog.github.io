---
layout:      post
title:       "What Keeps a Bayesian Awake At Night? Part 1: Day Time"
tags:        [theory, foundations]
authors:
    - name: Wessel Bruinsma
      link: https://wesselb.github.io
    - name: Andrew Y. K. Foong
      link: https://andrewfoongyk.github.io
    - name: Richard E. Turner
      link: http://cbl.eng.cam.ac.uk/Public/Turner/Turner
comments:    true
image:      /assets/images/what-keeps-a-bayesian-awake-at-night/day.jpg
image_attribution: 
    name: Raygar He
    link: https://unsplash.com/photos/bQDwE0_Q_e8
excerpt: |
    <i>The theory of probabilities is at bottom nothing but common sense reduced to calculus;
    it enables us to appreciate with exactness that which accurate minds feel with a sort of instinct for which ofttimes they are unable to account. <br>
    — Pierre-Simon Laplace (1749–1827)</i>
---

> *The theory of probabilities is at bottom nothing but common sense reduced to calculus;*
> *it enables us to appreciate with exactness that which accurate minds feel with a sort of instinct for which ofttimes they are unable to account.*
>
> --- *Pierre-Simon Laplace (1749--1827)*


The Cambridge Machine Learning Group is renowned for having drunk the Bayesian Kool-Aid. We evangelise about probabilistic approaches in our teaching and research as a principled and unifying view of machine learning and statistics. In this context, I (Rich) have found it striking that many of the most influential contributors to this brand of machine learning and statistics are more circumspect. From Michael Jordan remarking that "this place is far too Bayesian",[^1] to Geoff Hinton saying that, having listed the key properties of the problems he's interested in, Bayesian approaches just don't cut it. Even Zoubin Ghahramani confides that aspects of the Bayesian approach "keep him awake at night".[^2] 

So, as we kick-off this new blog, we thought we'd dig into these concerns and attempt to burst the Bayesian bubble. In order to do this, we've written two posts. In this first part, we present some of the standard sunny arguments for probabilistic inference and decision making. In the second part, we'll shoot some of these arguments down and face the demons lurking in the night.

## What is the probabilistic approach to inference and decision making?

The goal of an inference problem is to estimate an unknown quantity $X$ from known quantities $D$. Examples include inferring the mass of the Higgs boson ($X$) from collider data ($D$); estimating the prevalence of Covid 19 infections ($X$) from PCR test data ($D$); or reconstructing files ($X$) from corrupted versions stored on a damaged hard disk ($D$). 

The probabilistic approach to solving such problems proceeds in three stages: 

**Stage 1: probabilistic modelling.** The first stage is called *probabilistic modelling* and involves designing a probabilistic recipe that describes how all the variables, known variables $D$ and unknown variables $X$, are assumed to be produced. The model specifies a joint distribution over all these variables $p(X, D)$ and samples from it should reflect typical settings of the variables that you might have expected to encounter before seeing any data. 

**Stage 2: probabilistic inference.** The second stage is called *probabilistic inference*. In this stage, the sum and product rules of probability are used to manipulate the joint distribution over all variables into the conditional distribution of the unknown variables given the known variables:

\begin{equation} \label{eq:posterior}
    p( X \cond D ) = \frac{p( X, D)}{p( D )}.
\end{equation}

This distribution on the left hand side of this equation is known as the *posterior distribution*. It tells us how probable any setting of the unknown variables X is given the known variables $D$. In this way, it tells us not only what is the most likely setting of the unknown variables ($\hat X_{\text{MAP}} = \argmax_{X} p(X \cond D)$), but also our uncertainty about $X$ --- it summarises our *belief* about $X$ after seeing $D$. Equation \eqref{eq:posterior} follows from the product rule of probability;
another application of the product rule leads to Bayes' rule.

In the example of inferring the mass of the Higgs boson, the unknown X is a parameter.[^3] Inferring parameters using the sum and product rules to form the posterior distribution is called being *Bayesian*. 

**Stage 3: Bayesian decision theory.** In real-world problems, inferences are usually made to serve decision making. For example, inferences could inform the design of a new particle accelerator to pin down particle masses; could decide whether to implement another national lockdown; or could decide whether to prosecute someone for financial crimes based on data recovered from a hard drive. 

The probabilistic approach supports decision making in a third stage which goes by the grand name of *Bayesian decision theory*. Here the user provides a loss function $L(X, \delta)$ which specifies how unhappy they would be if they take decision $\delta$ when the unknown variables take a value $X$. We can compute how unhappy they will be on average with any decision by averaging the loss function over all possible settings of the unobserved variables, weighted by how probable they are under the posterior distribution $p( X \cond D )$:

\begin{equation}
    (\text{average unhappiness})(\delta) = \E_{p(X \cond D)}[L(X, \delta)].
\end{equation}

This quantity is called the *posterior expected loss*[^7]. We now pick the decision that we expect to be least unhappy about by minimising our expected unhappiness with respect to our decision $\delta$, and we're done!

**Summary.** This framework proposes a cleanly separated sequential three-step procedure: first, articulate your assumptions about the data via a probabilistic model; second, compute the posterior distribution over unknown variables; and third, select the decision which minimises the average loss under the posterior. 

## What's the formal justification for the probabilistic approach to inference and decision making?

Why does a Bayesian represent their beliefs with probabilities[^4], reason according to the sum and product rule, and select actions which minimise the posterior expected loss? We'll now review the most common theoretical arguments.

**(1) de Finetti's exchangeability theorem** justifies the use of model parameters $\theta$, conditional distributions $p(D_n \cond \theta)$ over data given parameters (also called the *likelihood of parameters*), and critically *prior distributions over parameters* $p(\theta)$ when specifying probabilistic models. The theorem says that if you believe the order in which the data $D = (D_n)\_{n=1}^N$ arrives is unimportant --- $p(D)$ is invariant to the order of $(D_n)\_{n=1}^N$, an idea called *exchangeability* --- then there exists a random variable $\theta$ with associated prior distribution such that the data $D$ are i.i.d. given $\theta$ and your belief is recovered by marginalising over $\theta$:

\begin{equation}
    p(D) = \int p(\theta) \prod_{n=1}^N p(D_n \cond \theta) \,\mathrm{d}\theta.
\end{equation}

The argument presupposes the use of probability distributions over data, but shows that this in combination with exchangeability entails the existence of parameters with associated prior and posterior distributions. This idea was important historically as there were schools of statistical thought that eschewed placing distributions over parameters, but were happy placing distributions over data.

**(2) Cox's theorem and coherence.** Cox's theorem ([Cox, 1945](https://aapt.scitation.org/doi/abs/10.1119/1.1990764); [Jaynes, 2003](https://bayes.wustl.edu/etj/prob/book.pdf)) justifies the use of a probabilistic model and application of the sum and the product rules to perform inference. The theorem starts out by listing a number of *desiderata* that any reasonable system of quantitative rules for inference should satisfy. One very important such desideratum is *consistency* or *coherence*[^5]: if there are multiple ways of arriving at an inference, they should all give the same answer. For example, updating our beliefs about an unknown variable X after observing data $D=(D_n)\_{n=1}^N$ should give the same result as incrementally updating our beliefs about $X$ one data point $D_n$ at a time. Cox's theorem is the conclusion that every system satisfying the desiderata must be probability theory. In particular, it identifies probability theory as the unique extension of propositional logic, where a proposition is either true or false, to varying degrees of plausibility.

**(3) The Dutch book argument** ([Ramsey, 1926](https://EconPapers.repec.org/RePEc:hay:hetcha:ramsey1926); [de Finetti, 1931](http://eudml.org/doc/212523)) is another argument for the optimality of modelling and inference, one which connects to decision making. The argument goes as follows: if you're willing to take bets on propositions with certain odds (in a way, these odds represent your beliefs), then, unless these odds are consistent with probability theory, you're willing to take a collection of bets that nets a sure loss (a *Dutch book*). 

**(4) Savage's theorem** ([Savage, 1945](https://doi.org/10.1002/nav.3800010316)) is used to argue that optimal decision making entails the three-step probabilistic approach. Like Cox's theorem, it takes an axiomatic approach, but here the axioms relate to decisions rather than inferences. The axioms include the idea that a decision maker is characterised by the ability to rank all decisions in some order of preference. It then lists properties that any reasonable ranking should have.[^6] Savage's theorem says that these properties entail that the decision maker's ranking is consistent with them acting according to Bayesian decision theory ([Karni, 2005](http://www.econ2.jhu.edu/people/Karni/savageseu.pdf)). 

The above arguments justify large parts of the probabilistic approach to inference and decision making. In contrast, the arguments we turn to next are only concerned with specific properties. One way to view them is as *unit tests* that the Bayesian framework passes.

Before we turn to them, you may have noticed that the formulation of the probabilistic approach and the justifications made so far do not include a notion of the *true model* or *true parameters*; rather, all that matters is your own personal beliefs about the world, how you update them as data arrive, and how decisions are made. However, it is perfectly reasonable to ask questions like "How does the posterior over parameters behave when the data were generated using some true underlying parameter value?" or "If I have the right model and apply probabilistic inference, will my predictions be good?" The next two results step out of the Bayesian framework to answer these questions:

**(5) Doob's consistency theorem** ([Doob, 1949](https://www.emis.de/journals/JEHPS/juin2009/Locker.pdf)) shows that Bayesian inference is *consistent*: very often, if the data were sampled from $p(D \cond X)$ for some true value for $X$, then, as the user collects more and more data, the posterior over the unknown variables $p(X \cond D)$ concentrates on this true value for $X$. This is a frequentist analysis of a Bayesian estimation procedure, and it shows that the two paradigms can live comfortably side by side: Bayesian methods provide estimation procedures; frequentist tools allow analysis of these procedures. This is one reason why Michael Jordan thought that any Bayesian-focussed research group worth its salt should be paying close attention to frequentist ideas.

**(6) Optimality of Bayesian predictions.** In many applications, such as those typically encountered in machine learning, predicting future data points is of central focus. What guarantees do we have on the quality of such estimates arising from the probabilistic approach? Well, if we use the Kullback--Leibler (KL) divergence to measure the distance between the true density $p(X \cond \theta)$ over the unknown $X$ and any data-dependent estimate of this density $q(X \cond D)$, then, when averaged over potential settings of the parameters and associated observed data $p(\theta)p(D\cond\theta)$, the estimate $q^*$ that minimises the average divergence is the Bayesian posterior predictive ([Aitchison, 1975](https://doi.org/10.1093/biomet/62.3.547)): 

\begin{equation}
    q^*(X \cond D) = \argmin_{q} \E_{p(\theta,D)}[ \operatorname{KL}( p(X \cond \theta) \\| q(X \cond D)) ] = p(X \cond D).
\end{equation}

This argument tells us that the Bayesian predictions coming from the "right model" are KL-optimal on average. Interestingly, this result connects to recent ideas in meta-learning ([Gordon *et al.*, 2019](https://arxiv.org/abs/1805.09921)).

**(7) Wald's theorem ([Wald, 1949](https://doi.org/10.1214/aoms/1177730030))** can be used to justify minimising an expected loss as a way of decision making. The theorem is concerned with *admissible* decision rules, which are rules that, for every other decision rule, achieve a better loss for at least *some* realisation of the unknown $X$. This condition is a very low bar: we'd hope that any reasonable decision rule would have this property. However, surprisingly, Wald's theorem says that the *only* rules which are admissible are essentially those derived from minimising the expected loss under some distribution ([Wald, 1949](https://doi.org/10.1214/aoms/1177730030); [Lehmann & Casella, 1998](https://doi.org/10.1007/b98854)).


## Conclusion

It is striking that a number of arguments based on a diversity of desirable properties --- including coherence, optimal betting strategies, specifying sensible preferences over actions, frequentist guarantees like consistency and optimal predictive accuracy --- all suggest that the probabilistic approach to inference is a reasonable one. But in the next post we'll ask whether, in the dead of night, everything is as rosy as it seems in daylight.

[^1]: Actually he was referring to the Gatsby unit in 2004 --- in many ways the mother of the Cambridge Machine Learning Group --- and his comment was a fair one.

[^2]: This was in the Approximate Inference Workshop at NeurIPS in 2017.

[^3]: Parameters are distinguished from variables by asking what happens as we see more data: variables get more numerous, parameters do not. 

[^7]: We previously incorrectly called the posterior expected loss the *Bayes risk*. Thanks to Corey Yanofsky for pointing out the mistake.

[^4]: That probabilities represent degrees of belief is only one interpretation of probability. For example, in *Probability, Statistics, and Truth* ([Von Mises, 1928](https://store.doverpublications.com/0486242145.html)), von Mises argues that probability concerns limiting frequencies of repeating events. In this view and contrary to the Bayesian view, it is meaningless to talk about the probability of a one-off event: it is not possible to repeatedly sample that one-off event, which means that it doesn't have a limiting frequency.

[^5]: Consistency also has another specific technical meaning in statistics, so we will use the term coherence in what follows.

[^6]:  The axioms are reminiscent of those used in [Arrow's impossibility theorem](https://en.wikipedia.org/wiki/Arrow%27s_impossibility_theorem) concerning *fair* voting systems.
