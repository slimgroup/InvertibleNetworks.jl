# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020
# Copyright: Georgia Institute of Technology, 2020

module InvertibleNetworks

import Base.size, Base.getindex, Flux.glorot_uniform
using LinearAlgebra, Random, NNlib, Flux, Statistics, Wavelets

export clear_grad!, glorot_uniform, get_params


# Getters for DenseConvDims fields 
# (need to redefine here as they are not public methods in NNlib)
input_size(c::DenseConvDims) = c.I
kernel_size(c::DenseConvDims{N,K,C_in,C_out,S,P,D,F}) where {N,K,C_in,C_out,S,P,D,F} = K
channels_in(c::DenseConvDims{N,K,C_in,C_out,S,P,D,F}) where {N,K,C_in,C_out,S,P,D,F} = C_in
channels_out(c::DenseConvDims{N,K,C_in,C_out,S,P,D,F}) where {N,K,C_in,C_out,S,P,D,F} = C_out

# Utils
include("utils/parameter.jl")
include("utils/objective_functions.jl")
include("utils/dimensionality_operations.jl")
include("utils/activation_functions.jl")
include("utils/test_distributions.jl")

# Single network layers (invertible and non-invertible)
include("conditional_layers/conditional_layer_residual_block.jl")
include("layers/layer_residual_block.jl")
include("layers/layer_affine.jl")
include("layers/invertible_layer_actnorm.jl")
include("layers/invertible_layer_conv1x1.jl")
include("layers/invertible_layer_basic.jl")
include("layers/invertible_layer_irim.jl")
include("layers/invertible_layer_glow.jl")
include("layers/invertible_layer_hyperbolic.jl")
include("layers/invertible_layer_hint.jl")
include("layers/invertible_layer_slim_additive.jl")
include("layers/invertible_layer_slim_affine.jl")
include("layers/invertible_layer_slim_learned.jl")

# Invertible network architectures
include("networks/invertible_network_irim.jl")  # i-RIM: Putzky and Welling (2019)
include("networks/invertible_network_glow.jl")  # Glow: Dinh et al. (2017), Kingma and Dhariwal (2018)
include("networks/invertible_network_hyperbolic.jl")    # Hyperbolic: Lensink et al. (2019)

# Conditional layers
include("conditional_layers/conditional_layer_hint.jl")
include("conditional_layers/conditional_layer_slim.jl")

end
