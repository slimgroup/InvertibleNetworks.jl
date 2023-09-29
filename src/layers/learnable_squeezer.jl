# Learnable up-/down-sampling from Etmann et al., 2020, https://arxiv.org/abs/2005.05220
# The Frechet derivative of the matrix exponential is from Al-Mohy and Higham, 2009, https://epubs.siam.org/doi/10.1137/080716426

export LearnableSqueezer

mutable struct LearnableSqueezer <: InvertibleNetwork

    # Stencil-related fields
    stencil_pars::Parameter
    pars2mat_idx
    stencil_size
    stencil::Union{AbstractArray,Nothing}
    cdims::Union{DenseConvDims,Nothing}

    # Internal flags
    trigger_recompute::Bool
    logdet::Bool
    reversed::Bool

    # Internal parameters related to the stencil exponential or derivative thereof
    log_mat::Union{AbstractArray,Nothing}
    niter_exp_derivative::Union{Nothing,Real}
    tol_exp_derivative::Union{Nothing,Real}

end

@Flux.functor LearnableSqueezer


# Constructor

function LearnableSqueezer(stencil_size::Integer...; logdet::Bool=false, zero_init::Bool=false, niter_exp_derivative::Union{Nothing,Integer}=nothing, tol_exp_derivative::Union{Nothing,Real}=nothing, reversed::Bool=false)

    σ = prod(stencil_size)
    zero_init ? (stencil_pars = vec2par(zeros(Float32, div(σ*(σ-1), 2)), (div(σ*(σ-1), 2), ))) :
                (stencil_pars = vec2par(glorot_uniform(div(σ*(σ-1), 2)), (div(σ*(σ-1), 2), )))
    pars2mat_idx = _skew_symmetric_indices(σ)
    return LearnableSqueezer(stencil_pars, pars2mat_idx, stencil_size, nothing, nothing,
                             true, logdet, reversed,
                             nothing, niter_exp_derivative, tol_exp_derivative)

end


# Forward/inverse/backward

function forward(X::AbstractArray{T,N}, C::LearnableSqueezer; logdet::Union{Nothing,Bool}=nothing) where {T,N}
    C.trigger_recompute && C.reversed && throw(ArgumentError("Must first compute forward to initialize reverse(LearnableSqueezer)"))
    isnothing(logdet) && (logdet = C.logdet)

    # Compute exponential stencil
    if C.trigger_recompute
        _compute_exponential_stencil!(C, size(X, N-1); set_log=true)
        C.cdims = DenseConvDims(size(X), size(C.stencil); stride=C.stencil_size)
        C.trigger_recompute = false
    end

    # Convolution
    X = conv(X, C.stencil, C.cdims)

    return logdet ? (X, convert(T, 0)) : X

end

function inverse(Y::AbstractArray{T,N}, C::LearnableSqueezer; logdet::Union{Nothing,Bool}=nothing) where {T,N}
    C.trigger_recompute && ~C.reversed && throw(ArgumentError("Must first compute forward to initialize LearnableSqueezer"))
    isnothing(logdet) && (logdet = C.logdet)

    # Compute exponential stencil
    if C.trigger_recompute
        size_X = Int.(size(Y).*(C.stencil_size..., 1/prod(C.stencil_size), 1))
        _compute_exponential_stencil!(C, size_X[N-1]; set_log=true)
        C.cdims = DenseConvDims(size_X, size(C.stencil); stride=C.stencil_size)
        C.trigger_recompute = false
    end

    # Convolution (adjoint)
    Y = ∇conv_data(Y, C.stencil, C.cdims)

    return logdet ? (Y, convert(T, 0)) : Y

end

function backward(ΔY::AbstractArray{T,N}, Y::AbstractArray{T,N}, C::LearnableSqueezer; set_grad::Bool=true, trigger_recompute::Bool=true) where {T,N}
    C.trigger_recompute && ~C.reversed && throw(ArgumentError("Must first compute forward to initialize LearnableSqueezer"))

    # Convolution (adjoint)
    X  = ∇conv_data(Y,  C.stencil, C.cdims)
    ΔX = ∇conv_data(ΔY, C.stencil, C.cdims)

    # Parameter gradient
    Δstencil = _mat2stencil_adjoint(∇conv_filter(X, ΔY, C.cdims), C.stencil_size, size(X, N-1))
    ΔA = _Frechet_derivative_exponential(C.log_mat', Δstencil; niter=C.niter_exp_derivative, tol=tol=isnothing(C.tol_exp_derivative) ? nothing : T(C.tol_exp_derivative))
    Δstencil_pars = ΔA[C.pars2mat_idx[1]]-ΔA[C.pars2mat_idx[2]]
    set_grad && (C.stencil_pars.grad = Δstencil_pars)

    # Trigger recomputation
    trigger_recompute && trigger_recompute!(C)

    return set_grad ? (ΔX, X) : (ΔX, Δstencil_pars, X)

end

function backward_inv(ΔX::AbstractArray{T,N}, X::AbstractArray{T,N}, C::LearnableSqueezer; set_grad::Bool=true, trigger_recompute::Bool=true) where {T,N}
    C.trigger_recompute && C.reversed && throw(ArgumentError("Must first compute forward to initialize reverse(LearnableSqueezer)"))

    # Convolution (adjoint)
    Y  = conv(X,  C.stencil, C.cdims)
    ΔY = conv(ΔX, C.stencil, C.cdims)

    # Parameter gradient
    Δstencil = _mat2stencil_adjoint(∇conv_filter(X, ΔY, C.cdims), C.stencil_size, size(X, N-1))
    ΔA = _Frechet_derivative_exponential(C.log_mat', Δstencil; niter=C.niter_exp_derivative, tol=isnothing(C.tol_exp_derivative) ? nothing : T(C.tol_exp_derivative))
    Δstencil_pars = ΔA[C.pars2mat_idx[1]]-ΔA[C.pars2mat_idx[2]]
    set_grad && (C.stencil_pars.grad = -Δstencil_pars)

    # Trigger recomputation
    trigger_recompute && trigger_recompute!(C)

    return set_grad ? (ΔY, Y) : (ΔY, -Δstencil_pars, Y)

end

tag_as_reversed!(C::LearnableSqueezer, tag::Bool) = (C.reversed = tag; return C)

set_params!(C::LearnableSqueezer, θ::AbstractVector{<:Parameter}) = (C.stencil_pars = θ[1]; trigger_recompute!(C))

trigger_recompute!(C::LearnableSqueezer) = (C.trigger_recompute = true)


# Internal utilities for LearnableSqueezer

function _compute_exponential_stencil!(C::LearnableSqueezer, nc::Integer; set_log::Bool=false)
    n = prod(C.stencil_size)
    log_mat = _pars2skewsymm(C.stencil_pars.data, C.pars2mat_idx, n)
    C.stencil = _mat2stencil(_exponential(log_mat), C.stencil_size, nc)
    set_log && (C.log_mat = log_mat)
end

function _mat2stencil(A::AbstractMatrix{T}, k::NTuple{N,Integer}, nc::Integer) where {T,N}
    stencil = similar(A, k..., nc, k..., nc); fill!(stencil, 0)
    @inbounds for i = 1:nc
        selectdim(selectdim(stencil, N+1, i), 2*N+1, i) .= reshape(A, k..., k...)
    end
    return reshape(stencil, k..., nc, :)
end

function _mat2stencil_adjoint(stencil::AbstractArray{T}, k::NTuple{N,Integer}, nc::Integer) where {T,N}
    stencil = reshape(stencil, k..., nc, k..., nc)
    A = similar(stencil, prod(k), prod(k)); fill!(A, 0)
    @inbounds for i = 1:nc
        A .+= reshape(selectdim(selectdim(stencil, N+1, i), 2*N+1, i), prod(k), prod(k))
    end
    return A
end

function _pars2skewsymm(Apars::AbstractVector{T}, pars2mat_idx::NTuple{2,AbstractVector{<:Integer}}, n::Integer) where T
    A = similar(Apars, n, n)
    A[pars2mat_idx[1]] .=  Apars
    A[pars2mat_idx[2]] .= -Apars
    A[diagind(A)] .= 0
    return A
end

function _exponential(A::AbstractMatrix{T}) where T
    expA = copy(A)
    exponential!(expA)
    return expA
end

function _skew_symmetric_indices(σ::Integer)
    CIs = reshape(1:σ^2, σ, σ)
    idx_u = Vector{Int}(undef, 0)
    idx_l = Vector{Int}(undef, 0)
    for i=1:σ, j=i+1:σ # Indices related to (strictly) upper triangular part
        push!(idx_u, CIs[i,j])
    end
    for j=1:σ, i=j+1:σ # Indices related to (strictly) lower triangular part
        push!(idx_l, CIs[i,j])
    end
    return idx_u, idx_l
end

function _Frechet_derivative_exponential(A::AbstractMatrix{T}, ΔA::AbstractMatrix{T}; niter::Union{Nothing,Integer}=nothing, tol::Union{Nothing,T}=nothing) where T

    # Set default options
    isnothing(niter) && (niter = 100)
    isnothing(tol) && (tol = eps(T))

    # Allocating arrays
    dA = copy(ΔA)
    Mk = copy(ΔA)
    Apowk = copy(A)

    @inbounds for k = 2:niter

        # Truncated series    
        Mk .= (Mk*A+Apowk*ΔA)/k
        Apowk .= (Apowk*A)/k
        dA .+= Mk

        # Convergence check
        ~isnothing(tol) && (norm(Mk)/norm(dA) < tol) && break

    end

    return dA

end