---
title: 'InvertibleNetworks.jl: A Julia package for scalable normalizing flows'
tags:
  - Julia
  - inverse problems
  - Bayesian inference
  - imaging
  - normalizing flows
authors:
  - name: Rafael Orozco
    affiliation: 1
  - name: Philipp Witte
    affiliation: 2
  - name: Mathias Louboutin
    affiliation: 3
  - name: Ali Siahkoohi
    affiliation: 4
  - name: Gabrio Rizzuti
    affiliation: 5
  - name: Bas Peters
    affiliation: 6
  - name: Felix J. Herrmann
    affiliation: 1
affiliations:
 - name: Georgia Institute of Technology (GT), USA
   index: 1
 - name: Microsoft Research, USA
   index: 2
 - name: Devito Codes, UK
   index: 3
 - name: Rice University, USA
   index: 4
 - name: Shearwater GeoServices, UK
   index: 5
 - name: Computational Geosciences Inc, Canada
   index: 6
date:  28 November 2023  
bibliography: paper.bib
---

# Summary

Normalizing flows is a density estimation method that provides efficient exact likelihood estimation and sampling [@dinh2014nice] from high-dimensional distributions. This method depends on the use of the change of variables formula, which requires an invertible transform. Thus normalizing flow architectures are built to be invertible by design [@dinh2014nice]. In theory, the invertibility of architectures constrains the expressiveness, but the use of coupling layers allows normalizing flows to exploit the power of arbitrary neural networks, which do not need to be invertible, [@dinh2016density] and layer invertibility means that, if properly implemented, many layers can be stacked to increase expressiveness without creating a training memory bottleneck.  

The package we present, InvertibleNetworks.jl, is a pure Julia [@bezanson2017julia] implementation of normalizing flows. We have implemented many relevant neural network layers, including GLOW 1x1 invertible convolutions [@kingma2018glow], affine/additive coupling layers [@dinh2014nice], Haar wavelet multiscale transforms [@haar1909theorie], and Hierarchical invertible neural transport (HINT) [@kruse2021hint], among others. These modular layers can be easily composed and modified to create different types of normalizing flows. As starting points, we have implemented RealNVP, GLOW, HINT, Hyperbolic networks [@lensink2022fully] and their conditional counterparts for users to quickly implement their individual applications. 

# Statement of need


This software package focuses on memory efficiency. The promise of neural networks is in learning high-dimensional distributions from data thus normalizing flow packages should allow application to large dimensional inputs such as images or 3D volumes. Interestingly, the invertibility of normalizing flows naturally alleviates memory concerns since intermediate network activations can be recomputed instead of saved in memory. This greatly reduces the memory needed during backpropagation. The problem is that directly implementing normalizing flows in automatic differentiation frameworks such as PyTorch [@paszke2019pytorch] will not automatically exploit this invertibility. The available packages for normalizing flows such as Bijectors.jl [@fjelde2020bijectors], NormalizingFlows.jl [@NormalizingFlows.jl], nflows [@nflows], normflows [@stimper2023normflows], and FrEIA [@freia] are built depending on automatic differentiation frameworks and thus do not exploit invertibility for memory efficiently. 

We chose to write this package in Julia since it was built with a commitment to facilitate interoperability with other packages for workflows in scientific machine learning [@louboutin2022accelerating]. The interoperability was facilitated by the multiple dispatch system of Julia. Our goal is to provide solutions for imaging problems with high degrees of freedom, where computational speed is crucial. We have found that this software significantly benefits from Julia's Just-In-Time compilation technology.

# Memory efficiency
By implementing gradients by hand, instead of depending completely on automatic differentiation, our layers are capable of scaling to large inputs. By scaling, we mean that these codes are not prone to out-of-memory errors when training on GPU accelerators. Indeed, previous literature has described memory problems when using normalizing flows as their invertibility requires the latent code to maintain the same dimensionality as the input [@khorashadizadeh2023conditional].

![Our InvertibleNetworks.jl package provides memory frugal implementations of normalizing flows. Here, we compare our implementation of GLOW with an equivalent implementation in a PyTorch package.  Using a 40GB A100 GPU, the PyTorch package can not train on image sizes larger than 480x480,  while our package can handle sizes larger than 1024x1024.
\label{fig:memory}](./figs/mem_used_new.png)

In \autoref{fig:memory}, we show the relation between input size and the memory required for a gradient calculation in a PyTorch normalizing flow package (normflows [@stimper2023normflows]) as compared to our package. The two tests were run with identical normalizing flow architectures. We note that the PyTorch implementation quickly increases the memory load and throws an out-of-memory error on the 40-GB A100 GPU at a spatial image size of 480x480, while our InvertibleNetworks.jl implementation still has not run out of memory at spatial size 1024x1024. Note that this is in the context of a typical learning routine, so the images include 3 channels (RGB) and we used a batch size of 8. 

![Due to the invertibility of the normalizing flow architecture, the memory consumption does not increase as we increase the depth of the network. Our package properly exploits this property and thus shows constant memory consumption, whereas the PyTorch package does not. 
\label{fig:memory-depth}](./figs/mem_used_new_depth.png)

Since traditional normalizing flow architectures need to be invertible they might be less expressive then their non-invertible counterparts. In order to increase their expressiveness, practitioners stack many invertible layers to increase the overall expressive power. Increasing the depth of a neural network would in most cases increase the memory consumption of the network but in this case, since normalizing flows are invertible, the memory consumption does not increase. Our package displays this phenomenon as shown in \autoref{fig:memory-depth} while the PyTorch (normflows) package, which has been implemented with automatic differentiation, does not display this constant memory phenomenon. 

# Ease of use
Although the normalizing flow layer gradients are hand-written, the package is fully compatibly with ChainRules [@frames_white_2023_10100624] in order to integrate with automatic differentiation frameworks in Julia such as Zygote [@innes2019differentiable]. This integration allows users to add arbitrary neural networks which will be differentiated by automatic differentiation while the memory bottleneck created by normalizing flow gradients will be dealt with by InvertibleNetworks.jl. The typical use case for this combination are the summary networks used in amortized variational inference such as BayesFlow [@radev2020bayesflow], which is also implemented in our package. 

All implemented layers are tested for invertibility and correctness of their gradients with continuous integration testing via GitHub actions.  There are many examples for layers, networks and  application workflows allowing new users to quickly build networks for a variety of applications. The ease of use is demonstrated by the publications that made use of the package.

Many publications have used InvertibleNetworks.jl for diverse applications including change point detection, [@peters2022point], acoustic data denoising [@kumar2021enabling], seismic imaging [@rizzuti2020parameterizing; @siahkoohi2021preconditioned; @siahkoohi2022wave;@siahkoohi2023reliable; @louboutin2023learned;@alemohammad2023self], fluid flow dynamics [@yin2023solving], medical imaging [@orozco2023adjoint;@orozco2023amortized; @orozco2021photoacoustic;@orozco2023refining], and monitoring CO2 for combating climate change [@gahlot2023inference].

# Future work
The neural network primitives (convolutions, non-linearities, pooling etc) are implemented in NNlib.jl abstractions, thus support for AMD, Intel, and Apple GPUs can be trivially extended. Also, while our package can currently handle 3D inputs and has been used on large volume-based medical imaging [@orozco2022memory], there are interesting avenues of research regarding the "channel explosion" seen in invertible down and upsampling used in invertible networks [@peters2019symmetric]. 

# Acknowledgements

The development of this package was carried out with the support of Georgia Research Alliance and partners of the ML4Seismic Center. This was also supported in part by the US National Science Foundation grant OAC 2203821 and the Department of Energy grant No. DE-SC0021515.

# References

::: {#refs}
:::

