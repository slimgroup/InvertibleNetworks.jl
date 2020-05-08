# Tests for invertible neural network module
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, Test

test_suite = "all"   # "all", "layers" or "networks"

if test_suite == "all" || test_suite == "layers"
    @testset "Test individual layers" begin

        # Utils
        include("test_utils/test_objectives.jl")
        include("test_utils/test_nnlib_convolution.jl")
        include("test_utils/test_activations.jl")
        include("test_utils/test_squeeze.jl")

        # Layers
        include("test_layers/test_residual_block.jl")
        include("test_layers/test_householder_convolution.jl")
        include("test_layers/test_coupling_layer_basic.jl")
        include("test_layers/test_coupling_layer_basic_inverse.jl")
        include("test_layers/test_coupling_layer_irim.jl")
        include("test_layers/test_coupling_layer_glow.jl")
        include("test_layers/test_coupling_layer_hint.jl")
        include("test_layers/test_coupling_layer_slim.jl")
        include("test_layers/test_coupling_layer_slim_learned.jl")
        include("test_layers/test_conditional_layer_hint.jl")
        include("test_layers/test_conditional_layer_slim.jl")
        include("test_layers/test_conditional_res_block.jl")
	    include("test_layers/test_hyperbolic_layer.jl")
        include("test_layers/test_actnorm.jl")
    end
end

# Networks
if test_suite == "all" || test_suite == "networks"
    @testset "Test networks" begin
        include("test_networks/test_unrolled_loop.jl")
        include("test_networks/test_generator.jl")
        include("test_networks/test_glow.jl")
        include("test_networks/test_hyperbolic_network.jl")
        include("test_networks/test_conditional_hint_network.jl")
    end
end
