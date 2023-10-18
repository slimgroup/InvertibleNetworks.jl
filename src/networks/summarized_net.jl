export SummarizedNet

"""
    G = SummarizedNet(cond_net, sum_net)

 Create a summarized neural conditional approximator from conditional approximator cond_net and summary network sum_net.

 *Input*: 

 - 'cond_net': invertible conditional distribution approximator

 - 'sum_net': Should be flux layer. summary network. Should be invariant to a dimension of interest. 

 *Output*:
 
 - `G`: summarized network.

 *Usage:*

 - Forward mode: `ZX, ZY,  logdet = G.forward(X, Y)`

 - Backward mode: `ΔX, X, ΔY = G.backward(ΔZX, ZX, ZY; Y_save=Y)`

 - inverse mode: `ZX, ZY  logdet = G.inverse(ZX, ZY)`

 *Trainable parameters:*

 - None in `G` itself

 - Trainable parameters in conditional approximator `G.cond_net` and smmary network `G.sum_net`,

 See also: [`ActNorm`](@ref), [`CouplingLayerGlow!`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct SummarizedNet <: InvertibleNetwork
	cond_net::InvertibleNetwork
	sum_net
end

@Flux.functor SummarizedNet

# Forward pass 
function forward(X::AbstractArray{T, N}, Y::AbstractArray{T, N}, S::SummarizedNet) where {T, N}
    S.cond_net(X,  S.sum_net(Y))
end

# Inverse pass 
function inverse(X::AbstractArray{T, N}, Y::AbstractArray{T, N}, S::SummarizedNet) where {T, N}
    S.cond_net.inverse(X,  Y)
end

# Backward pass and compute gradients
function backward(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, Y::AbstractArray{T, N}, S::SummarizedNet; Y_save=nothing) where {T, N}
	ΔX, X, ΔY = S.cond_net.backward(ΔX,X,Y)
    ΔY = S.sum_net.backward(ΔY, Y_save)
    return ΔX, X, ΔY
end

function backward_inv(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, Y::AbstractArray{T, N}, S::SummarizedNet; Y_save=nothing) where {T, N}
    ΔX, X, ΔY = S.cond_net.backward_inv(ΔX,X,Y)
    #ΔY = S.sum_net.backward(ΔY, Y_save)
    return ΔX, X, ΔY
end
