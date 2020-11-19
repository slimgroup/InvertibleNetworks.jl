using ChainRulesCore
import ChainRulesCore: frule, rrule


## Tape types and utilities

"""
Invertible global state type, it keeps track of invertible blocks of operations (each block being a sequence of contiguous invertible layers)
"""
mutable struct StateInvertibleOperations
    Y::Array{Any,1}
    layer_blocks::Array{Any,1}
    counter_block::Int64
    counter_layer::Int64
end

"""
Constructor
"""
StateInvertibleOperations() = StateInvertibleOperations([], [], 0, 0)

# Initialize global state
const GLOBAL_STATE_INVOPS = StateInvertibleOperations()
export GLOBAL_STATE_INVOPS

"""
Get current state
"""
current(state::StateInvertibleOperations) = state.Y[state.counter_block]

"""
Reset state
"""
function reset!(state::StateInvertibleOperations)
    state.Y = []
    state.layer_blocks = []
    state.counter_block = 0
    state.counter_layer = 0
end

"""
Determine if the input is related to a new block of invertible operations
"""
isa_newblock(state::StateInvertibleOperations, X) = (state.counter_block == 0) || !(state.Y[end] == X)

"""
Error if mismatch between state and network
"""
function check_coherence(state::StateInvertibleOperations, net::Union{NeuralNetLayer,InvertibleNetwork})
    if state.counter_block != 0 && state.counter_layer != 0 && state.layer_blocks[state.counter_block][state.counter_layer] != net
        reset!(state)
        throw(ArgumentError("Current state does not correspond to current layer, resetting state..."))
    end
end

"""
Update state in the forward pass
"""
function forward_update!(state::StateInvertibleOperations, X::Array{Float32,N}, Y::Array{Float32,N}, net::Union{NeuralNetLayer,InvertibleNetwork}) where N

    if isa_newblock(state, X)
        push!(state.Y, Y)
        push!(state.layer_blocks, [net])
        state.counter_block += 1
        state.counter_layer = 1
    else
        state.Y[state.counter_block] = Y
        push!(state.layer_blocks[state.counter_block], net)
        state.counter_layer += 1
    end

end

"""
Update state in the backward pass
"""
function backward_update!(state::StateInvertibleOperations, X::Array{Float32,N}) where N

    if state.counter_layer == 1 # Check if first layer of current block
        state.Y[state.counter_block] = nothing
        state.counter_block -= 1
        (state.counter_block != 0) && (state.counter_layer = length(state.layer_blocks[state.counter_block]))
    else
        state.Y[state.counter_block] = X
        state.counter_layer -= 1
    end

    state.counter_block == 0 && reset!(state) # reset state when first block/first layer is reached

end


## Chain rules for invertible networks

# General pullback function
function pullback(net::Union{NeuralNetLayer,InvertibleNetwork}, ΔY::Array{Float32,N}; state::StateInvertibleOperations=GLOBAL_STATE_INVOPS) where N

    # Check state coherency
    check_coherence(state, net)

    # Backward pass
    # ΔY isa Tuple && (ΔY = ΔY[1])
    ΔX, X_ = net.backward(ΔY, current(state))

    # Update state
    backward_update!(state, X_)

    return (nothing, ΔX)

end

# Forward-mode AD
# ...

# Reverse-mode AD
function rrule(net::Union{NeuralNetLayer,InvertibleNetwork}, X; state::StateInvertibleOperations=GLOBAL_STATE_INVOPS)

    # Forward pass
    Y = net.forward(X)

    # Update state
    forward_update!(state, X, Y, net)

    # Pullback
    ∂Y_T(ΔY) = pullback(net, ΔY; state=state)

    return Y, ∂Y_T

end