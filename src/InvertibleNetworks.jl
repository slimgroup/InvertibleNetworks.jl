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

# Basic building blocks
include("parameter.jl")
include("dimensionality_operations.jl")
include("activation_functions.jl")
include("residual_block.jl")

# Invertible layers
include("actnorm.jl")
include("conv1x1.jl")
include("invertible_layer_irim.jl")
include("invertible_layer_glow.jl")

# Invertible network architectures
include("invertible_network_irim.jl")   # Putzky and Welling (2019)
include("invertible_network_glow.jl")   # Dinh et al (2017), Kingma and Dhariwal (2018)

end