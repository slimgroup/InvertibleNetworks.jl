# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020
# Copyright: Georgia Institute of Technology, 2020

module InvertibleNetworks

import Base.size, Base.length, Base.getindex, Base.reverse, Base.reverse!#, Base.show
import Base.+, Base.*, Base.-, Base./
import LinearAlgebra.dot, LinearAlgebra.norm#, LinearAlgebra.adjoint
import JOLI.adjoint, JOLI.show
import Flux.glorot_uniform
import CUDA: CuArray

using LinearAlgebra, Random, NNlib, Flux, Statistics, Wavelets, Zygote, JOLI

export clear_grad!, glorot_uniform


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
include("utils/neuralnet.jl")
include("utils/invnet_linops.jl")
include("utils/jacobian.jl")
include("utils/invertible_network_sequential.jl")

# Single network layers (invertible and non-invertible)
include("conditional_layers/conditional_layer_residual_block.jl")
include("layers/layer_residual_block.jl")
include("layers/layer_flux_block.jl")
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

# Conditional layers and nets
include("conditional_layers/conditional_layer_hint.jl")
include("conditional_layers/conditional_layer_slim.jl")
include("networks/invertible_network_conditional_hint.jl")
include("networks/invertible_network_conditional_hint_multiscale.jl")

# gpu
include("utils/cuda.jl")

end
