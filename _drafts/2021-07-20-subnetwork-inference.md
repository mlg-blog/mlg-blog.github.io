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
