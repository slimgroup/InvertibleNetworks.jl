export SummarizedNet

struct SummarizedNet <: InvertibleNetwork
	cond_net
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
