# Tests for invertible neural network module
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020
using InvertibleNetworks, Test

using TimerOutputs: TimerOutputs, @timeit

# Collect timing and allocations information to show in a clear way.
const TIMEROUTPUT = TimerOutputs.TimerOutput()
timeit_include(path::AbstractString) = @timeit TIMEROUTPUT path include(path)

const test_suite = get(ENV, "test_suite", "all") # "all", "basics", "layers" or "networks"

basics = ["test_utils/test_objectives.jl",
          "test_utils/test_sequential.jl",
          "test_utils/test_nnlib_convolution.jl",
          "test_utils/test_activations.jl", 
          "test_utils/test_squeeze.jl",
          "test_utils/test_jacobian.jl",
          "test_utils/test_chainrules.jl"]

          # Layers
layers = ["test_layers/test_residual_block.jl",
          "test_layers/test_flux_block.jl",
          "test_layers/test_resnet.jl",
          "test_layers/test_layer_conv1x1.jl",
          "test_layers/test_coupling_layer_basic.jl",
          "test_layers/test_coupling_layer_irim.jl",
          "test_layers/test_coupling_layer_glow.jl",
          "test_layers/test_coupling_layer_hint.jl",
          "test_layers/test_conditional_layer_hint.jl",
          "test_layers/test_conditional_res_block.jl",
          "test_layers/test_hyperbolic_layer.jl",
          "test_layers/test_actnorm.jl",
          "test_layers/test_layer_affine.jl"]

networks = ["test_networks/test_unrolled_loop.jl",
            "test_networks/test_generator.jl",
            "test_networks/test_glow.jl",
            "test_networks/test_hyperbolic_network.jl",
            "test_networks/test_multiscale_hint_network.jl",
            "test_networks/test_multiscale_conditional_hint_network.jl",
            "test_networks/test_conditional_hint_network.jl"]


if test_suite == "all" || test_suite == "basics"
    @testset verbose = true "Basics" begin
        for t=basics
            @testset  "Test $t" begin
                @timeit TIMEROUTPUT "$t" begin include(t) end
            end
        end
    end
end

if test_suite == "all" || test_suite == "layers"
    @testset verbose = true "Layers" begin
        for t=layers
            @testset  "Test $t" begin
                @timeit TIMEROUTPUT "$t" begin include(t) end
            end
        end
    end
end

# Networks
if test_suite == "all" || test_suite == "networks"
    @testset verbose = true "Networks" begin
        for t=networks
            @testset  "Test $t" begin
                @timeit TIMEROUTPUT "$t" begin include(t) end
            end
        end
    end
end

show(TIMEROUTPUT; compact=true, sortby=:firstexec)