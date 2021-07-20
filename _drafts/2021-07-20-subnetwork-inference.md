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
image:      /assets/images/subnet-inference/d_prediction.png
excerpt: |
    This blog post describes <i>subnetwork inference</i>, a recently-proposed framework for improved Bayesian inference in deep neural networks. The high-level idea is to perform inference over only a small, carefully selected subset of the model parameters instead of all parameters. This allows using fairly expressive posterior approximations (<i>e.g.</i> full covariance Gaussian distributions) that would otherwise be intractable.
---
