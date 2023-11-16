using ChainRulesCore
export logdetjac, getrrule
import ChainRulesCore: frule, rrule, @non_differentiable

@non_differentiable get_params(::InvertibleNetwork)
@non_differentiable get_params(::ReversedNetwork)
## Tape types and utilities

"""
    I = InvertibleOperationsTape(Y, layer_blocks, counter_block, counter_layer, logdet)

Invertible global state type, it keeps track of invertible blocks of operations (each block being a sequence of contiguous invertible layers)
"""
mutable struct InvertibleOperationsTape
    Y::AbstractVector{<:Any}
    layer_blocks::AbstractVector{<:Any}
    counter_block::Integer
    counter_layer::Integer
    logdet::Union{Nothing,<:Real}
end

"""
 I = InvertibleOperationsTape() = InvertibleOperationsTape([], [], 0, 0, nothing)

Default  Constructor
"""
InvertibleOperationsTape() = InvertibleOperationsTape([], [], 0, 0, nothing)

# Initialize global state
const GLOBAL_STATE_INVOPS = InvertibleOperationsTape()
export GLOBAL_STATE_INVOPS

"""
Get current state of the tape
"""
function current(state::InvertibleOperationsTape)
    state.counter_block == 0 && throw(ArgumentError("Please, run forward pass first, to rebuild global state."))
    return state.Y[state.counter_block]
end

"""
Reset the state of the tape
"""
function reset!(state::InvertibleOperationsTape)
    state.Y = []
    state.layer_blocks = []
    state.counter_block = 0
    state.counter_layer = 0
    state.logdet = nothing
end

"""
Determine if the input is related to a new block of invertible operations
"""
isa_newblock(state::InvertibleOperationsTape, X) = (state.counter_block == 0) || !(state.Y[end] == X)

"""
Error if mismatch between state and network
"""
function check_coherence(state::InvertibleOperationsTape, net::InvertibleNetwork)
    if state.counter_block != 0 && state.counter_layer != 0 && state.layer_blocks[state.counter_block][state.counter_layer] != net
        reset!(state)
        throw(ArgumentError("Current state does not correspond to current layer, resetting state..."))
    end
end

"""
Update state in the forward pass.
"""
function forward_update!(state::InvertibleOperationsTape, X::AbstractArray{T,N}, Y::AbstractArray{T,N}, logdet::Union{Nothing,T}, net::InvertibleNetwork) where {T, N}

    if isa_newblock(state, X)
        push!(state.Y, Y)
        push!(state.layer_blocks, Vector{Any}([net]))
        state.counter_block += 1
        state.counter_layer = 1
    else
        state.Y[state.counter_block] = Y
        push!(state.layer_blocks[state.counter_block], net)
        state.counter_layer += 1
    end
    if logdet isa Float32
        state.logdet === nothing ? (state.logdet = logdet) : (state.logdet += logdet)
    end
end

"""
Update state in the backward pass
"""
function backward_update!(state::InvertibleOperationsTape, X::AbstractArray{T,N}) where {T, N}

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
function pullback(net::InvertibleNetwork, ΔY::AbstractArray{T,N};
                  state::InvertibleOperationsTape=GLOBAL_STATE_INVOPS) where {T, N}

    # Check state coherency
    check_coherence(state, net)

    # Zygote feeds back wrong type ΔY in some cases so convert back if needed
    T2 = typeof(current(state))
    ΔY = convert(T2, ΔY)
    # Backward pass
    ΔX, X_ = net.backward(ΔY, current(state); set_grad=true)
    Δθ = getfield.(get_params(net), :grad)
    # Update state
    backward_update!(state, X_)

    return NoTangent(), NoTangent(), ΔX, Δθ
end


# Reverse-mode AD rule
function ChainRulesCore.rrule(::typeof(forward_net), net::InvertibleNetwork, X::AbstractArray{T, N}, θ...;
                              state::InvertibleOperationsTape=GLOBAL_STATE_INVOPS) where {T, N}

    # Forward pass
    net.logdet ? ((Y, logdet) = net.forward(X)) : (Y = net.forward(X); logdet = nothing)

    # Update state
    forward_update!(state, X, Y, logdet, net)

    # Pullback
    ∂Y_T(ΔY) = pullback(net, ΔY; state=state)

    return Y, ∂Y_T
end


## Logdet utilities for Zygote pullback

logdetjac(; state::InvertibleOperationsTape=GLOBAL_STATE_INVOPS) = state.logdet

## Utility to get the pullback directly for testing

getrrule(net::InvertibleNetwork, X::AbstractArray) = rrule(forward_net, net, X, getfield.(get_params(net), :data))
