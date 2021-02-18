# InvertibleNetworks.jl

![CI](https://github.com/slimgroup/InvertibleNetworks.jl/workflows/CI/badge.svg)
[![DOI](https://zenodo.org/badge/239018318.svg)](https://zenodo.org/badge/latestdoi/239018318)

Building blocks for invertible neural networks in the [Julia] programming language.

- Memory efficient building blocks for invertible neural networks 
- Hand-derived gradients, Jacobians $J$ , and $\log |J|$
- [Flux] integration 
- Support for [Zygote] and [ChainRules]
- GPU support
- Includes various examples of invertible neural networks, normalizing flows, variational inference, and uncertainty quantification


## Installation

```
] dev https://github.com/slimgroup/InvertibleNetworks.jl
```

## Papers

The following publications use [InvertibleNetworks.jl]:

- **[“Preconditioned training of normalizing flows for variational inference in inverse problems”]**
    - paper: [https://arxiv.org/abs/2101.03709](https://arxiv.org/abs/2101.03709)
    - [presentation](https://slim.gatech.edu/Publications/Public/Conferences/AABI/2021/siahkoohi2021AABIpto/siahkoohi2021AABIpto_pres.pdf)
    - code: [FastApproximateInference.jl]

- **["Parameterizing uncertainty by deep invertible networks, an application to reservoir characterization"]**
    - paper: [https://arxiv.org/abs/2004.07871](https://arxiv.org/abs/2004.07871)
    - [presentation](https://slim.gatech.edu/Publications/Public/Conferences/SEG/2020/rizzuti2020SEGuqavp/rizzuti2020SEGuqavp_pres.pdf)
    - code: [https://github.com/slimgroup/Software.SEG2020](https://github.com/slimgroup/Software.SEG2020)

- **["Generalized Minkowski sets for the regularization of inverse problems"]**
    - paper: [http://arxiv.org/abs/1903.03942](http://arxiv.org/abs/1903.03942)
    - code: [SetIntersectionProjection.jl]

## Building blocks 

- 1x1 Convolutions using Householder transformations ([example](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples/layers/layer_convolution_1x1.jl))

- Residual block ([example](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples/layers/layer_residual_block.jl))

- Invertible coupling layer from Dinh et al. (2017) ([example](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples/layers/layer_coupling_glow.jl))

- Invertible hyperbolic layer from Lensink et al. (2019) ([example](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples/layers/layer_coupling_hyperbolic.jl))

- Invertible coupling layer from Putzky and Welling (2019) ([example](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples/layers/layer_coupling_irim.jl))

- Invertible recursive coupling layer HINT from Kruse et al. (2020) ([example](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples/layers/layer_coupling_hint.jl))

- Activation normalization (Kingma and Dhariwal, 2018) ([example](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples/layers/layer_actnorm.jl))

- Various activation functions (Sigmoid, ReLU, leaky ReLU, GaLU)

- Objective and misfit functions (mean squared error, log-likelihood)

- Dimensionality manipulation: squeeze/unsqueeze (column, patch, checkerboard), split/cat

- Squeeze/unsqueeze using the wavelet transform


## Examples

- Invertible recurrent inference machines (Putzky and Welling, 2019) ([generic example](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples/networks/network_irim.jl))

- Generative models with maximum likelihood via the change of variable formula ([example](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples/applications/application_glow_banana_dist.jl))

- Glow: Generative flow with invertible 1x1 convolutions (Kingma and Dhariwal, 2018) ([generic example](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/examples/networks/network_glow.jl), [source](https://github.com/slimgroup/InvertibleNetworks.jl/tree/master/src/networks/invertible_network_glow.jl))

## GPU support

GPU support is supported via Flux/CuArray. To use the GPU, move the input and the network layer to GPU via `|> gpu`

```
using InvertibleNetworks, Flux

# Input
nx = 64
ny = 64
k = 10
batchsize = 4

# Input image: nx x ny x k x batchsize
X = randn(Float32, nx, ny, k, batchsize) |> gpu

# Activation normalization
AN = ActNorm(k; logdet=true) |> gpu

# Test invertibility
Y_, logdet = AN.forward(X)
```

## Acknowledgments

This package uses functions from [NNlib.jl](https://github.com/FluxML/NNlib.jl), [Flux.jl](https://github.com/FluxML/Flux.jl) and [Wavelets.jl](https://github.com/JuliaDSP/Wavelets.jl)


## References

 - Yann Dauphin, Angela Fan, Michael Auli and David Grangier, "Language modeling with gated convolutional networks", Proceedings of the 34th International Conference on Machine Learning, 2017. https://arxiv.org/pdf/1612.08083.pdf

 - Laurent Dinh, Jascha Sohl-Dickstein and Samy Bengio, "Density estimation using Real NVP",  International Conference on Learning Representations, 2017, https://arxiv.org/abs/1605.08803

 - Diederik P. Kingma and Prafulla Dhariwal, "Glow: Generative Flow with Invertible 1x1 Convolutions", Conference on Neural Information Processing Systems, 2018. https://arxiv.org/abs/1807.03039

 - Keegan Lensink, Eldad Haber and Bas Peters, "Fully Hyperbolic Convolutional Neural Networks", arXiv Computer Vision and Pattern Recognition, 2019. https://arxiv.org/abs/1905.10484

 - Patrick Putzky and Max Welling, "Invert to learn to invert", Advances in Neural Information Processing Systems, 2019. https://arxiv.org/abs/1911.10914

 - Jakob Kruse, Gianluca Detommaso, Robert Scheichl and Ullrich Köthe, "HINT: Hierarchical Invertible Neural Transport for Density Estimation and Bayesian Inference", arXiv Statistics and Machine Learning, 2020. https://arxiv.org/abs/1905.10687

## Authors

 - Philipp Witte, Microsoft Corporation (pwitte@microsoft.com)

 - Gabrio Rizzuti, Utrecht University

 - Mathias Louboutin, Georgia Institute of Technology

 - Ali Siahkoohi, Georgia Institute of Technology

[Flux]:https://fluxml.ai
[Julia]:https://julialang.org
[Zygote]:https://github.com/FluxML/Zygote.jl
[ChainRules]:https://github.com/JuliaDiff/ChainRules.jl
[InvertibleNetworks.jl]:https://github.com/slimgroup/InvertibleNetworks.jl
[“Preconditioned training of normalizing flows for variational inference in inverse problems”]:https://slim.gatech.edu/content/preconditioned-training-normalizing-flows-variational-inference-inverse-problems
[FastApproximateInference.jl]:https://github.com/slimgroup/Software.siahkoohi2021AABIpto
["Generalized Minkowski sets for the regularization of inverse problems"]:https://slim.gatech.edu/content/generalized-minkowski-sets-regularization-inverse-problems-1
[SetIntersectionProjection.jl]:https://github.com/slimgroup/SetIntersectionProjection.jl
["Parameterizing uncertainty by deep invertible networks, an application to reservoir characterization"]:https://slim.gatech.edu/content/parameterizing-uncertainty-deep-invertible-networks-application-reservoir-characterization