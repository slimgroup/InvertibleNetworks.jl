# Tests for invertible neural network module
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, Test

test_suite = "layers"   # "all", "layers" or "networks"

if test_suite == "all" || test_suite == "layers"
    @testset "Test individual layers" begin
        include("test_objectives.jl")
        include("test_householder_convolution.jl")
        include("test_residual_block.jl")
        include("test_nnlib_convolution.jl")
        include("test_activations.jl")
        include("test_invertible_layer.jl")
        include("test_coupling_layer.jl")
        include("test_hyperbolic_layer.jl")
        include("test_actnorm.jl")
        include("test_squeeze.jl")
    end
end

if test_suite == "all" || test_suite == "networks"
    @testset "Test networks" begin
        include("test_unrolled_loop.jl")
        include("test_generator.jl")
        include("test_glow.jl")
    end
end