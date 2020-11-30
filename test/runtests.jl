# Tests for invertible neural network module
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020
using InvertibleNetworks, Test

const test_suite = get(ENV, "test_suite", "all") # "all", "basics", "layers" or "networks"

basics = ["test_utils/test_objectives.jl",
          "test_utils/test_sequential.jl",
          "test_utils/test_nnlib_convolution.jl",
          "test_utils/test_activations.jl",
          "test_utils/test_squeeze.jl",
          "test_utils/test_jacobian.jl"]

          # Layers
layers = ["test_layers/test_residual_block.jl",
          "test_layers/test_flux_block.jl",
          "test_layers/test_resnet.jl",
          "test_layers/test_layer_conv1x1.jl",
          "test_layers/test_coupling_layer_basic.jl",
          "test_layers/test_coupling_layer_irim.jl",
          "test_layers/test_coupling_layer_glow.jl",
          "test_layers/test_coupling_layer_hint.jl",
          "test_layers/test_coupling_layer_slim.jl",
          "test_layers/test_coupling_layer_slim_learned.jl",
          "test_layers/test_conditional_layer_hint.jl",
          "test_layers/test_conditional_layer_slim.jl",
          "test_layers/test_conditional_res_block.jl",
          "test_layers/test_hyperbolic_layer.jl",
          "test_layers/test_actnorm.jl",
          "test_layers/test_layer_affine.jl"]

networks = ["test_networks/test_unrolled_loop.jl",
            "test_networks/test_generator.jl",
            "test_networks/test_glow.jl",
            "test_networks/test_hyperbolic_network.jl",
            "test_networks/test_multiscale_hint_network.jl",
            "test_networks/test_conditional_hint_network.jl"]


if test_suite == "all" || test_suite == "basics"
    for t=basics
        @testset "Test $t" begin
            @time include(t)
        end
    end
end

if test_suite == "all" || test_suite == "layers"
    for t=layers
        @testset  "Test $t" begin
            @time include(t)
        end
    end
end

# Networks
if test_suite == "all" || test_suite == "networks"
    for t=networks
        @testset  "Test $t" begin
            @time include(t)
        end
    end
end
