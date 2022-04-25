# Invertible conditional HINT layer from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export ConditionalLayerHINT, ConditionalLayerHINT3D

"""
    CH = ConditionalLayerHINT(n_in, n_hidden; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, permute=true, ndims=2) (2D)

    CH = ConditionalLayerHINT3D(n_in, n_hidden; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, permute=true) (3D)

 Create a conditional HINT layer based on coupling blocks and 1 level recursion.

 *Input*:

 - `n_in`, `n_hidden`: number of input and hidden channels of both `X` and `Y`

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 - `permute`: bool to indicate whether to permute `X` and `Y`. Default is `true`

 - `ndims` : number of dimensions

 *Output*:

 - `CH`: Conditional HINT coupling layer.

 *Usage:*

 - Forward mode: `Zx, Zy, logdet = CH.forward_X(X, Y)`

 - Inverse mode: `X, Y = CH.inverse(Zx, Zy)`

 - Backward mode: `ΔX, ΔY, X, Y = CH.backward(ΔZx, ΔZy, Zx, Zy)`

 - Forward mode Y: `Zy = CH.forward_Y(Y)`

 - Inverse mode Y: `Y = CH.inverse(Zy)`

 *Trainable parameters:*

 - None in `CH` itself

 - Trainable parameters in coupling layers `CH.CL_X`, `CH.CL_Y`, `CH.CL_YX` and in
   permutation layers `CH.C_X` and `CH.C_Y`.

 See also: [`CouplingLayerBasic`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
mutable struct ConditionalLayerHINT <: NeuralNetLayer
    CL_X::CouplingLayerHINT
    CL_Y::CouplingLayerHINT
    CL_YX::CouplingLayerBasic
    C_X::Union{Conv1x1, Nothing}
    C_Y::Union{Conv1x1, Nothing}
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor ConditionalLayerHINT

# 2D Constructor from input dimensions
function ConditionalLayerHINT(n_in::Int64, n_hidden::Int64; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1,
                              logdet=true, permute=true, ndims=2)

    # Create basic coupling layers
    CL_X = CouplingLayerHINT(n_in, n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=logdet, permute="none", ndims=ndims)
    CL_Y = CouplingLayerHINT(n_in, n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=logdet, permute="none", ndims=ndims)
    CL_YX = CouplingLayerBasic(n_in, n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=logdet, ndims=ndims)

    # Permutation using 1x1 convolution
    permute == true ? (C_X = Conv1x1(n_in)) : (C_X = nothing)
    permute == true ? (C_Y = Conv1x1(n_in)) : (C_Y = nothing)

    return ConditionalLayerHINT(CL_X, CL_Y, CL_YX, C_X, C_Y, logdet, false)
end

# 3D Constructor from input dimensions
ConditionalLayerHINT3D(args...; kw...) = ConditionalLayerHINT(args...; kw..., ndims=3)

function forward(X::AbstractArray{T, N}, Y::AbstractArray{T, N}, CH::ConditionalLayerHINT; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (CH.logdet && ~CH.is_reversed) : logdet = logdet

    # Y-lane
    ~isnothing(CH.C_Y) ? (Yp = CH.C_Y.forward(Y)) : (Yp = copy(Y))
    logdet ? (Zy, logdet2) = CH.CL_Y.forward(Yp) : Zy = CH.CL_Y.forward(Yp)

    # X-lane: coupling layer
    ~isnothing(CH.C_X) ? (Xp = CH.C_X.forward(X)) : (Xp = copy(X))
    logdet ? (X, logdet1) = CH.CL_X.forward(Xp) : X = CH.CL_X.forward(Xp)

    # X-lane: conditional layer
    logdet ? (Zx, logdet3) = CH.CL_YX.forward(Yp, X)[2:3] : Zx = CH.CL_YX.forward(Yp, X)[2]

    logdet ? (return Zx, Zy, logdet1 + logdet2 + logdet3) : (return Zx, Zy)
end

function inverse(Zx::AbstractArray{T, N}, Zy::AbstractArray{T, N}, CH::ConditionalLayerHINT; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (CH.logdet && CH.is_reversed) : logdet = logdet

    # Y-lane
    logdet ? (Yp, logdet1) = CH.CL_Y.inverse(Zy; logdet=true) : Yp = CH.CL_Y.inverse(Zy; logdet=false)
    ~isnothing(CH.C_Y) ? (Y = CH.C_Y.inverse(Yp)) : (Y = copy(Yp))

    # X-lane: conditional layer
    YZ = tensor_cat(Yp, Zx)
    logdet ? (X, logdet2) = CH.CL_YX.inverse(Yp, Zx)[2:3] : X = CH.CL_YX.inverse(Yp, Zx)[2]

    # X-lane: coupling layer
    logdet ? (Xp, logdet3) = CH.CL_X.inverse(X; logdet=true) : Xp = CH.CL_X.inverse(X; logdet=false)
    ~isnothing(CH.C_X) ? (X = CH.C_X.inverse(Xp)) : (X = copy(Xp))

    logdet ? (return X, Y, logdet1 + logdet2 + logdet3) : (return X, Y)
end

function backward(ΔZx::AbstractArray{T, N}, ΔZy::AbstractArray{T, N}, Zx::AbstractArray{T, N}, Zy::AbstractArray{T, N}, CH::ConditionalLayerHINT; logdet=nothing, set_grad::Bool=true) where {T, N}
    isnothing(logdet) ? logdet = (CH.logdet && ~CH.is_reversed) : logdet = logdet

    # Y-lane
    if set_grad
        ΔYp, Yp = CH.CL_Y.backward(ΔZy, Zy)
    else
        if logdet
            ΔYp, Δθ_CLY, Yp, ∇logdet_CLY = CH.CL_Y.backward(ΔZy, Zy; set_grad=set_grad)
        else
            ΔYp, Δθ_CLY, Yp = CH.CL_Y.backward(ΔZy, Zy; set_grad=set_grad)
        end
    end

    # X-lane: conditional layer
    if set_grad
        ΔYp_, ΔX, X = CH.CL_YX.backward(ΔYp.*0, ΔZx, Yp, Zx)[[1,2,4]]
    else
        if logdet
            ΔYp_, ΔX, Δθ_CLYX, _, X, ∇logdet_CLYX = CH.CL_YX.backward(ΔYp.*0, ΔZx, Yp, Zx; set_grad=set_grad)
        else
            ΔYp_, ΔX, Δθ_CLYX, _, X = CH.CL_YX.backward(ΔYp.*0, ΔZx, Yp, Zx; set_grad=set_grad)
        end
    end
    ΔYp += ΔYp_

    # X-lane: coupling layer
    if set_grad
        ΔXp, Xp = CH.CL_X.backward(ΔX, X)
    else
        if logdet
            ΔXp, Δθ_CLX, Xp, ∇logdet_CLX = CH.CL_X.backward(ΔX, X; set_grad=set_grad)
        else
            ΔXp, Δθ_CLX, Xp = CH.CL_X.backward(ΔX, X; set_grad=set_grad)
        end
    end

    # 1x1 Convolutions
    if isnothing(CH.C_X) || isnothing(CH.C_Y)
        ΔX = copy(ΔXp); X = copy(Xp)
        ΔY = copy(ΔYp); Y = copy(Yp)
    else
        if set_grad
            ΔX, X = CH.C_X.inverse((ΔXp, Xp))
            ΔY, Y = CH.C_Y.inverse((ΔYp, Yp))
        else
            ΔX, Δθ_CX, X = CH.C_X.inverse((ΔXp, Xp); set_grad=set_grad)
            ΔY, Δθ_CY, Y = CH.C_Y.inverse((ΔYp, Yp); set_grad=set_grad)
        end
    end

    if set_grad
        return ΔX, ΔY, X, Y
    else
        Δθ = cat(Δθ_CLX, Δθ_CLY, Δθ_CLYX; dims=1)
        ~isnothing(CH.C_X) && (Δθ = cat(Δθ, Δθ_CX; dims=1))
        ~isnothing(CH.C_Y) && (Δθ = cat(Δθ, Δθ_CY; dims=1))
        if ~logdet
            return ΔX, ΔY, Δθ, X, Y
        else
            ∇logdet = cat(∇logdet_CLX, ∇logdet_CLY, ∇logdet_CLYX; dims=1)
            ~isnothing(CH.C_X) && (∇logdet = cat(∇logdet, 0*Δθ_CX; dims=1))
            ~isnothing(CH.C_Y) && (∇logdet = cat(∇logdet, 0*Δθ_CY; dims=1))
            return ΔX, ΔY, Δθ, X, Y, ∇logdet
        end
    end
end

function backward_inv(ΔX::AbstractArray{T, N}, ΔY::AbstractArray{T, N}, X::AbstractArray{T, N}, Y::AbstractArray{T, N}, CH::ConditionalLayerHINT) where {T, N}

    # 1x1 Convolutions
    if isnothing(CH.C_X) || isnothing(CH.C_Y)
        ΔXp = copy(ΔX); Xp = copy(X)
        ΔYp = copy(ΔY); Yp = copy(Y)
    else
        ΔXp, Xp = CH.C_X.forward((ΔX, X))
        ΔYp, Yp = CH.C_Y.forward((ΔY, Y))
    end

    # X-lane: coupling layer
    ΔX, X = backward_inv(ΔXp, Xp, CH.CL_X)

    # X-lane: conditional layer
    ΔYp_, ΔZx, Zx = backward_inv(ΔYp.*0, ΔX, Yp, X, CH.CL_YX)[[1,2,4]]
    ΔYp += ΔYp_

    # Y-lane
    ΔZy, Zy = backward_inv(ΔYp, Yp, CH.CL_Y)

    return ΔZx, ΔZy, Zx, Zy
end

function forward_Y(Y::AbstractArray{T, N}, CH::ConditionalLayerHINT) where {T, N}
    ~isnothing(CH.C_Y) ? (Yp = CH.C_Y.forward(Y)) : (Yp = copy(Y))
    Zy = CH.CL_Y.forward(Yp; logdet=false)
    return Zy

end

function inverse_Y(Zy::AbstractArray{T, N}, CH::ConditionalLayerHINT) where {T, N}
    Yp = CH.CL_Y.inverse(Zy; logdet=false)
    ~isnothing(CH.C_Y) ? (Y = CH.C_Y.inverse(Yp)) : (Y = copy(Yp))
    return Y
end


## Jacobian-related utils

function jacobian(ΔX::AbstractArray{T, N}, ΔY::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, X::AbstractArray{T, N}, Y::AbstractArray{T, N}, CH::ConditionalLayerHINT; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (CH.logdet && ~CH.is_reversed) : logdet = logdet

    # Selecting parameters
    npars_cx = ~isnothing(CH.C_X)*3
    npars_cy = ~isnothing(CH.C_Y)*3
    npars_clyx = 5
    npars_cl = Int64((length(Δθ)-npars_cx-npars_cy-npars_clyx)/2)
    Δθ_CLX = Δθ[1:npars_cl]
    Δθ_CLY = Δθ[1+npars_cl:2*npars_cl]
    Δθ_CLYX = Δθ[1+2*npars_cl:2*npars_cl+npars_clyx]
    ~isnothing(CH.C_X) && (Δθ_CX = Δθ[2*npars_cl+npars_clyx+1:2*npars_cl+npars_clyx+npars_cx])
    ~isnothing(CH.C_Y) && (Δθ_CY = Δθ[2*npars_cl+npars_clyx+npars_cx+1:end])

    # Y-lane
    if ~isnothing(CH.C_Y)
        ΔYp, Yp = CH.C_Y.jacobian(ΔY, Δθ_CY, Y)
    else
        Yp = copy(Y)
        ΔYp = copy(ΔY)
    end
    if logdet
        ΔZy, Zy, logdet2, GNΔθ_Y = CH.CL_Y.jacobian(ΔYp, Δθ_CLY, Yp)
    else
        ΔZy, Zy = CH.CL_Y.jacobian(ΔYp, Δθ_CLY, Yp)
    end

    # X-lane: coupling layer
    if ~isnothing(CH.C_X)
        ΔXp, Xp = CH.C_X.jacobian(ΔX, Δθ_CX, X)
    else
        Xp = copy(X)
        ΔXp = copy(ΔX)
    end
    if logdet
        ΔX, X, logdet1, GNΔθ_X = CH.CL_X.jacobian(ΔXp, Δθ_CLX, Xp)
    else
        ΔX, X = CH.CL_X.jacobian(ΔXp, Δθ_CLX, Xp)
    end

    # X-lane: conditional layer
    if logdet
        _, ΔZx, _, Zx, logdet3, GNΔθ_YX = CH.CL_YX.jacobian(ΔYp, ΔX, Δθ_CLYX, Yp, X)
    else
        _, ΔZx, _, Zx = CH.CL_YX.jacobian(ΔYp, ΔX, Δθ_CLYX, Yp, X)
    end

    if logdet
        GNΔθ = cat(GNΔθ_X, GNΔθ_Y, GNΔθ_YX; dims=1)
        ~isnothing(CH.C_X) && (GNΔθ = cat(GNΔθ, 0 .*Δθ_CX; dims=1))
        ~isnothing(CH.C_Y) && (GNΔθ = cat(GNΔθ, 0 .*Δθ_CY; dims=1))
        return ΔZx, ΔZy, Zx, Zy, logdet1 + logdet2 + logdet3, GNΔθ
    else
        return ΔZx, ΔZy, Zx, Zy
    end

end

adjointJacobian(ΔZx::AbstractArray{T, N}, ΔZy::AbstractArray{T, N}, Zx::AbstractArray{T, N}, Zy::AbstractArray{T, N}, CH::ConditionalLayerHINT; logdet=nothing) where {T, N} = backward(ΔZx, ΔZy, Zx, Zy, CH; set_grad=false, logdet=logdet)

# Set is_reversed flag in full network tree
function tag_as_reversed!(H::ConditionalLayerHINT, tag::Bool)
    H.is_reversed = tag
    tag_as_reversed!(H.CL_X, tag)
    tag_as_reversed!(H.CL_Y, tag)
    tag_as_reversed!(H.CL_YX, tag)
    return H
end
