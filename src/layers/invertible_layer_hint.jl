# Invertible HINT coupling layer from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerHINT, CouplingLayerHINT3D

"""
    H = CouplingLayerHINT(n_in, n_hidden; logdet=false, permute="none", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, ndims=2) (2D)

    H = CouplingLayerHINT(n_in, n_hidden; logdet=false, permute="none", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, ndims=3) (3D)

    H = CouplingLayerHINT3D(n_in, n_hidden; logdet=false, permute="none", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1) (3D)

 Create a recursive HINT-style invertible layer based on coupling blocks.

 *Input*:

 - `n_in`, `n_hidden`: number of input and hidden channels

 - `logdet`: bool to indicate whether to return the log determinant. Default is `false`.

 - `permute`: string to specify permutation. Options are `"none"`, `"lower"`, `"both"` or `"full"`.

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 - `ndims` : number of dimensions

 *Output*:

 - `H`: Recursive invertible HINT coupling layer.

 *Usage:*

 - Forward mode: `Y = H.forward(X)`

 - Inverse mode: `X = H.inverse(Y)`

 - Backward mode: `ΔX, X = H.backward(ΔY, Y)`

 *Trainable parameters:*

 - None in `H` itself

 - Trainable parameters in coupling layers `H.CL`

 See also: [`CouplingLayerBasic`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
mutable struct CouplingLayerHINT <: NeuralNetLayer
    CL::AbstractArray{CouplingLayerBasic, 1}
    C::Union{Conv1x1, Nothing}
    logdet::Bool
    permute::String
    is_reversed::Bool
end

@Flux.functor CouplingLayerHINT

# Get layer depth for recursion
function get_depth(n_in)
    count = 0
    nc = n_in
    while nc > 4
        nc /= 2
        count += 1
    end
    return count +1
end

# Constructor for given coupling layer and 1 x 1 convolution
CouplingLayerHINT(CL::AbstractArray{CouplingLayerBasic, 1}, C::Union{Conv1x1, Nothing};
    logdet=false, permute="none") = CouplingLayerHINT(CL, C, logdet, permute, false)

# 2D Constructor from input dimensions
function CouplingLayerHINT(n_in::Int64, n_hidden::Int64; logdet=false, permute="none",
                           k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, ndims=2)

    # Create basic coupling layers
    n = get_depth(n_in)
    CL = Array{CouplingLayerBasic}(undef, n)
    for j=1:n
        CL[j] = CouplingLayerBasic(Int(n_in/2^j), n_hidden; k1=k1, k2=k2, p1=p1, p2=p2,
                                   s1=s1, s2=s2, logdet=logdet, ndims=ndims)
    end

    # Permutation using 1x1 convolution
    if permute == "full" || permute == "both"
        C = Conv1x1(n_in)
    elseif permute == "lower"
        C = Conv1x1(Int(n_in/2))
    else
        C = nothing
    end

    return CouplingLayerHINT(CL, C, logdet, permute, false)
end

CouplingLayerHINT3D(args...;kw...) = CouplingLayerHINT(args...; kw..., ndims=3)

# Input is tensor X
function forward(X::AbstractArray{T, N}, H::CouplingLayerHINT; scale=1, permute=nothing, logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (H.logdet && ~H.is_reversed) : logdet = logdet
    isnothing(permute) ? permute = H.permute : permute = permute

    # Permutation
    if permute == "full" || permute == "both"
        X = H.C.forward(X)
    end
    Xa, Xb = tensor_split(X)
    permute == "lower" && (Xb = H.C.forward(Xb))

    # Determine whether to continue recursion
    recursive = false
    if N == 4 && size(X, 3) > 4
        recursive = true
    elseif N == 5 && size(X, 4) > 4
        recursive = true
    end

    # HINT coupling
    if recursive
        # Call function recursively
        Ya, logdet1 = forward(Xa, H; scale=scale+1, permute="none")
        Y_temp, logdet2 = forward(Xb, H; scale=scale+1, permute="none")
        if logdet
            Yb, logdet3 = H.CL[scale].forward(Xa, Y_temp)[[2,3]]
        else
            Yb = H.CL[scale].forward(Xa, Y_temp)[2]
            logdet3 = 0
        end
        logdet_full = logdet1 + logdet2 + logdet3
    else
        # Finest layer
        Ya = copy(Xa)
        if logdet
            Yb, logdet_full = H.CL[scale].forward(Xa, Xb)[[2,3]]
        else
            Yb = H.CL[scale].forward(Xa, Xb)[2]
            logdet_full = 0
        end
    end

    Y = tensor_cat(Ya, Yb)
    permute == "both" && (Y = H.C.inverse(Y))
    if scale == 1
        logdet ? (return Y, logdet_full) : (return Y)
    else
        return Y, logdet_full
    end
end

# Input is tensor Y
function inverse(Y::AbstractArray{T, N} , H::CouplingLayerHINT; scale=1, permute=nothing, logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (H.logdet && H.is_reversed) : logdet = logdet
    isnothing(permute) ? permute = H.permute : permute = permute

    # Permutation
    permute == "both" && (Y = H.C.forward(Y))
    Ya, Yb = tensor_split(Y)

    # Check for recursion
    recursive = (size(Y, N-1) > 4)

    # Coupling layer
    if recursive
        Xa, logdet1 = inverse(Ya, H; scale=scale+1, permute="none")
        if logdet
            Y_temp, logdet2 = H.CL[scale].inverse(Xa, Yb; logdet=true)[[2,3]]
        else
            Y_temp = H.CL[scale].inverse(Xa, Yb)[2]
            logdet2 = 0
        end
        Xb, logdet3 = inverse(Y_temp, H; scale=scale+1, permute="none")
        logdet_full = logdet1 + logdet2 + logdet3
    else
        Xa = copy(Ya)
        if logdet
            Xb, logdet_full = H.CL[scale].inverse(Ya, Yb)[[2,3]]
        else
            Xb = H.CL[scale].inverse(Ya, Yb)[2]
            logdet_full = 0
        end
    end

    # Initial permutation
    permute == "lower" && (Xb = H.C.inverse(Xb))
    X = tensor_cat(Xa, Xb)
    if permute == "full" || permute == "both"
        X = H.C.inverse(X)
    end

    if scale == 1 
        logdet ? (return X, logdet_full) : (return X)
    else
        return X, logdet_full
    end
end

# Input are two tensors ΔY, Y
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, H::CouplingLayerHINT; scale=1, permute=nothing, set_grad::Bool=true) where {T, N}
    isnothing(permute) ? permute = H.permute : permute = permute

    # Initializing output parameter array
    if !set_grad
        nscales_tot = length(H.CL)
        nparams = 5*(nscales_tot-scale+1)
        (permute != "none") && (nparams += 3)
        Δθ = Array{Parameter, 1}(undef, nparams)
        H.logdet && (∇logdet = Array{Parameter, 1}(undef, nparams))
    end

    if permute == "both"
        if set_grad
            ΔY, Y = H.C.forward((ΔY, Y))
        else
            ΔY, Δθ_C, Y = H.C.forward((ΔY, Y); set_grad=set_grad)
            Δθ[end-2:end] .= Δθ_C
            H.logdet && (∇logdet[end-2:end] .= [Parameter(cuzeros(Y, size(H.C.v1))), Parameter(cuzeros(Y, size(H.C.v2))), Parameter(cuzeros(Y, size(H.C.v3)))])
        end
    end
    Ya, Yb = tensor_split(Y)
    ΔYa, ΔYb = tensor_split(ΔY)

    # Determine whether to continue recursion
    recursive = (size(Y, N-1) > 4)

    # HINT coupling
    if recursive
        if set_grad
            ΔXa, Xa = backward(ΔYa, Ya, H; scale=scale+1, permute="none")
            ΔXa_temp, ΔXb_temp, X_temp = H.CL[scale].backward(ΔXa.*0, ΔYb, Xa, Yb)[[1,2,4]]
            ΔXb, Xb = backward(ΔXb_temp, X_temp, H; scale=scale+1, permute="none")
        else
            if H.logdet
                ΔXa, Δθa, Xa, ∇logdet_a = backward(ΔYa, Ya, H; scale=scale+1, permute="none", set_grad=set_grad)
                ΔXa_temp, ΔXb_temp, Δθ_scale, _, X_temp, ∇logdet_scale = H.CL[scale].backward(ΔXa.*0, ΔYb, Xa, Yb; set_grad=set_grad)
                ΔXb, Δθb, Xb, ∇logdet_b = backward(ΔXb_temp, X_temp, H; scale=scale+1, permute="none", set_grad=set_grad)
                ∇logdet[1:5] .= ∇logdet_scale
                ∇logdet[6:5+length(∇logdet_a)] .= ∇logdet_a+∇logdet_b
            else
                ΔXa, Δθa, Xa = backward(ΔYa, Ya, H; scale=scale+1, permute="none", set_grad=set_grad)
                ΔXa_temp, ΔXb_temp, Δθ_scale, _, X_temp = H.CL[scale].backward(ΔXa.*0, ΔYb, Xa, Yb; set_grad=set_grad)
                ΔXb, Δθb, Xb = backward(ΔXb_temp, X_temp, H; scale=scale+1, permute="none", set_grad=set_grad)
            end
            Δθ[1:5] .= Δθ_scale
            Δθ[6:5+length(Δθa)] .= Δθa+Δθb
        end
        ΔXa += ΔXa_temp
    else
        Xa = copy(Ya)
        ΔXa = copy(ΔYa)
        if set_grad
            ΔXa_, ΔXb, Xb = H.CL[scale].backward(ΔYa.*0, ΔYb, Ya, Yb)[[1,2,4]]
        else
            if H.logdet
                ΔXa_, ΔXb, Δθ_scale, _, Xb, ∇logdet_scale = H.CL[scale].backward(ΔYa.*0, ΔYb, Ya, Yb; set_grad=set_grad)
                ∇logdet[1:5] .= ∇logdet_scale
            else
                ΔXa_, ΔXb, Δθ_scale, _, Xb = H.CL[scale].backward(ΔYa.*0, ΔYb, Ya, Yb; set_grad=set_grad)
            end
            Δθ[1:5] .= Δθ_scale
        end
        ΔXa += ΔXa_
    end
    if permute == "lower"
        if set_grad
            ΔXb, Xb = H.C.inverse((ΔXb, Xb))
        else
            ΔXb, Δθ_C, Xb = H.C.inverse((ΔXb, Xb); set_grad=set_grad)
            H.logdet && (∇logdet[end-2:end] .= [Parameter(cuzeros(Y, size(H.C.v1))), Parameter(cuzeros(Y, size(H.C.v2))), Parameter(cuzeros(Y, size(H.C.v3)))])
        end
    end
    ΔX = tensor_cat(ΔXa, ΔXb)
    X = tensor_cat(Xa, Xb)
    if permute == "full" || permute == "both"
        if set_grad
            ΔX, X = H.C.inverse((ΔX, X))
        else
            ΔX, Δθ_C, X = H.C.inverse((ΔX, X); set_grad=set_grad)
            if permute == "full"
                Δθ[end-2:end] .= Δθ_C
                H.logdet && (∇logdet[end-2:end] .= [Parameter(cuzeros(Y, size(H.C.v1))), Parameter(cuzeros(Y, size(H.C.v2))), Parameter(cuzeros(Y, size(H.C.v3)))])
            else
                Δθ[end-2:end] += Δθ_C
                H.logdet && (∇logdet[end-2:end] += [Parameter(cuzeros(Y, size(H.C.v1))), Parameter(cuzeros(Y, size(H.C.v2))), Parameter(cuzeros(Y, size(H.C.v3)))])
            end
        end
    end

    if set_grad
        return ΔX, X
    else
        H.logdet ? (return ΔX, Δθ, X, ∇logdet) : (return ΔX, Δθ, X)
    end
end

# Input are two tensors ΔX, X
function backward_inv(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, H::CouplingLayerHINT; scale=1, permute=nothing) where {T, N}
    isnothing(permute) ? permute = H.permute : permute = permute

    # Permutation
    if permute == "full" || permute == "both"
        (ΔX, X) = H.C.forward((ΔX, X))
    end
    ΔXa, ΔXb = tensor_split(ΔX)
    Xa, Xb = tensor_split(X)
    permute == "lower" && ((ΔXb, Xb) = H.C.forward((ΔXb, Xb)))

    # Check whether to continue recursion
    recursive = (size(X, N-1) > 4)

    # Coupling layer backprop
    if recursive
        ΔY_temp, Y_temp = backward_inv(ΔXb, Xb, H; scale=scale+1, permute="none")
        ΔYa_temp, ΔYb, Yb = backward_inv(0 .*ΔXa, ΔY_temp, Xa, Y_temp, H.CL[scale])[[1,2,4]]
        ΔYa, Ya = backward_inv(ΔXa+ΔYa_temp, Xa, H; scale=scale+1, permute="none")
    else
        ΔYa = copy(ΔXa)
        Ya = copy(Xa)
        ΔYa_temp, ΔYb, Yb = backward_inv(0 .*ΔYa, ΔXb, Xa, Xb, H.CL[scale])[[1,2,4]]
        ΔYa += ΔYa_temp
    end
    ΔY = tensor_cat(ΔYa, ΔYb)
    Y = tensor_cat(Ya, Yb)
    permute == "both" && ((ΔY, Y) = H.C.inverse((ΔY, Y)))
    return ΔY, Y
end


## Jacobian-related functions

function jacobian(ΔX::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, X::AbstractArray{T, N}, H::CouplingLayerHINT;
                  scale=1, permute=nothing, logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (H.logdet && ~H.is_reversed) : logdet = logdet
    isnothing(permute) ? permute = H.permute : permute = permute

    # Selecting parameters
    if permute == "none"
        Δθscale = Δθ[1:5]
        Δθ = Δθ[6:end]
    else
        Δθscale = Δθ[1:5]
        Δθ_C = Δθ[end-2:end]
        Δθ = Δθ[6:end-3]
    end

    # Permutation
    if permute == "full" || permute == "both"
        ΔX, X = H.C.jacobian(ΔX, Δθ_C, X)
    end
    Xa, Xb = tensor_split(X)
    ΔXa, ΔXb = tensor_split(ΔX)
    if permute == "lower"
        ΔXb, Xb = H.C.jacobian(ΔXb, Δθ_C, Xb)
    end

    # Initialize Gauss-Newton approx of logdet term
    if logdet
        nscales_tot = length(H.CL)
        nparams = 5*(nscales_tot-scale+1)
        (permute != "none") && (nparams += 3)
        GNΔθ_full = Array{Parameter, 1}(undef, nparams)
        (permute != "none") && (GNΔθ_full[end-2:end] = [Parameter(cuzeros(X, size(H.C.v1))), Parameter(cuzeros(X, size(H.C.v2))), Parameter(cuzeros(X, size(H.C.v3)))])
    end

    # Determine whether to continue recursion
    recursive = (size(X, N-1) > 4)

    # HINT coupling
    # idx_Δθ_scale = (scale-1)*5+1:scale*5
    if recursive
        # Call function recursively
        if logdet
            ΔYa, Ya, logdet1, GNΔθ1 = jacobian(ΔXa, Δθ, Xa, H; scale=scale+1, permute="none")
            ΔY_temp, Y_temp, logdet2, GNΔθ2 = jacobian(ΔXb, Δθ, Xb, H; scale=scale+1, permute="none")
            _, ΔYb, _, Yb, logdet3, GNΔθ3 = H.CL[scale].jacobian(ΔXa, ΔY_temp, Δθscale, Xa, Y_temp)
            logdet_full = logdet1 + logdet2 + logdet3
            GNΔθ_full[1:5] .= GNΔθ3
            (permute != "none") ? (GNΔθ_full[6:end-3] .= GNΔθ1 + GNΔθ2) : (GNΔθ_full[6:end] .= GNΔθ1 + GNΔθ2)
        else
            ΔYa, Ya = jacobian(ΔXa, Δθ, Xa, H; scale=scale+1, permute="none")
            ΔY_temp, Y_temp = jacobian(ΔXb, Δθ, Xb, H; scale=scale+1, permute="none")
            _, ΔYb, _, Yb = H.CL[scale].jacobian(ΔXa, ΔY_temp, Δθscale, Xa, Y_temp)
        end
    else
        # Coarsest scale
        Ya = copy(Xa)
        ΔYa = copy(ΔXa)
        if logdet
            _, ΔYb, _, Yb, logdet_full, GNΔθ3 = H.CL[scale].jacobian(ΔXa, ΔXb, Δθscale, Xa, Xb)
            GNΔθ_full[1:5] .= GNΔθ3
        else
            _, ΔYb, _, Yb = H.CL[scale].jacobian(ΔXa, ΔXb, Δθscale, Xa, Xb)
        end
    end

    Y = tensor_cat(Ya, Yb)
    ΔY = tensor_cat(ΔYa, ΔYb)
    if permute == "both"
        ΔY, Y = H.C.jacobianInverse(ΔY, Δθ_C, Y)
    end
    logdet ? (return ΔY, Y, logdet_full, GNΔθ_full) : (return ΔY, Y)

end

adjointJacobian(ΔY, Y, H::CouplingLayerHINT; scale=1, permute=nothing) = backward(ΔY, Y, H; scale=scale, permute=permute, set_grad=false)

# Set is_reversed flag in full network tree
function tag_as_reversed!(H::CouplingLayerHINT, tag::Bool)
    H.is_reversed = tag
    nlayers = length(H.CL)
    for j=1:nlayers
        H.CL[j].is_reversed = tag
    end
    return H
end
