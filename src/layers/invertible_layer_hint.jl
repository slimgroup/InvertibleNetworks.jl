# Invertible HINT coupling layer from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerHINT

"""
    H = CouplingLayerHINT(nx, ny, n_in, n_hidden, batchsize;
        logdet=false, permute="none", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1) (2D)

    H = CouplingLayerHINT(nx, ny, nz, n_in, n_hidden, batchsize;
        logdet=false, permute="none", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1) (3D)

 Create a recursive HINT-style invertible layer based on coupling blocks.

 *Input*:

 - `nx`, `ny`, `nz`: spatial dimensions of input

 - `n_in`, `n_hidden`: number of input and hidden channels

 - `logdet`: bool to indicate whether to return the log determinant. Default is `false`.

 - `permute`: string to specify permutation. Options are `"none"`, `"lower"`, `"both"` or `"full"`.

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

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
function CouplingLayerHINT(nx::Int64, ny::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64;
    logdet=false, permute="none", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

    # Create basic coupling layers
    n = get_depth(n_in)
    CL = Array{CouplingLayerBasic}(undef, n)
    for j=1:n
        CL[j] = CouplingLayerBasic(nx, ny, Int(n_in/2^j), n_hidden, batchsize;
            k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=logdet)
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

# 3D Constructor from input dimensions
function CouplingLayerHINT(nx::Int64, ny::Int64, nz::Int64, n_in::Int64, n_hidden::Int64, batchsize::Int64;
    logdet=false, permute="none", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

    # Create basic coupling layers
    n = get_depth(n_in)
    CL = Array{CouplingLayerBasic}(undef, n)
    for j=1:n
        CL[j] = CouplingLayerBasic(nx, ny, nz, Int(n_in/2^j), n_hidden, batchsize;
            k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=logdet)
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

# Input is tensor X
function forward(X, H::CouplingLayerHINT; scale=1, permute=nothing, logdet=nothing)
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
    if typeof(X) <: AbstractArray{Float32, 4} && size(X, 3) > 4
        recursive = true
    elseif typeof(X) <: AbstractArray{Float32, 5} && size(X, 4) > 4
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
            logdet3 = 0f0
        end
        logdet_full = logdet1 + logdet2 + logdet3
    else
        # Finest layer
        Ya = copy(Xa)
        if logdet
            Yb, logdet_full = H.CL[scale].forward(Xa, Xb)[[2,3]]
        else
            Yb = H.CL[scale].forward(Xa, Xb)[2]
            logdet_full = 0f0
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
function inverse(Y, H::CouplingLayerHINT; scale=1, permute=nothing, logdet=nothing)
    isnothing(logdet) ? logdet = (H.logdet && H.is_reversed) : logdet = logdet
    isnothing(permute) ? permute = H.permute : permute = permute

    # Permutation
    permute == "both" && (Y = H.C.forward(Y))
    Ya, Yb = tensor_split(Y)

    # Check for recursion
    recursive = false
    if typeof(Y) <: AbstractArray{Float32, 4} && size(Y, 3) > 4
        recursive = true
    elseif typeof(Y) <: AbstractArray{Float32, 5} && size(Y, 4) > 4
        recursive = true
    end

    # Coupling layer
    if recursive
        Xa, logdet1 = inverse(Ya, H; scale=scale+1, permute="none")
        if logdet
            Y_temp, logdet2 = H.CL[scale].inverse(Xa, Yb; logdet=true)[[2,3]]
        else
            Y_temp = H.CL[scale].inverse(Xa, Yb)[2]
            logdet2 = 0f0
        end
        Xb, logdet3 = inverse(Y_temp, H; scale=scale+1, permute="none")
        logdet_full = logdet1 + logdet2 + logdet3
    else
        Xa = copy(Ya)
        if logdet
            Xb, logdet_full = H.CL[scale].inverse(Ya, Yb)[[2,3]]
        else
            Xb = H.CL[scale].inverse(Ya, Yb)[2]
            logdet_full = 0f0
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
function backward(ΔY, Y, H::CouplingLayerHINT; scale=1, permute=nothing)
    isnothing(permute) ? permute = H.permute : permute = permute
    permute == "both" && ((ΔY, Y) = H.C.forward((ΔY, Y)))
    Ya, Yb = tensor_split(Y)
    ΔYa, ΔYb = tensor_split(ΔY)
    recursive = false
    if typeof(Y) <: AbstractArray{Float32, 4} && size(Y, 3) > 4
        recursive = true
    elseif typeof(Y) <: AbstractArray{Float32, 5} && size(Y, 4) > 4
        recursive = true
    end
    if recursive
        ΔXa, Xa = backward(ΔYa, Ya, H; scale=scale+1, permute="none")
        ΔXa_temp, ΔXb_temp, X_temp = H.CL[scale].backward(ΔXa.*0f0, ΔYb, Xa, Yb)[[1,2,4]]
        ΔXb, Xb = backward(ΔXb_temp, X_temp, H; scale=scale+1, permute="none")
        ΔXa += ΔXa_temp
    else
        Xa = copy(Ya)
        ΔXa = copy(ΔYa)
        ΔXa_, ΔXb, Xb = H.CL[scale].backward(ΔYa.*0f0, ΔYb, Ya, Yb)[[1,2,4]]
        ΔXa += ΔXa_
    end
    permute == "lower" && ((ΔXb, Xb) = H.C.inverse((ΔXb, Xb)))
    ΔX = tensor_cat(ΔXa, ΔXb)
    X = tensor_cat(Xa, Xb)
    if permute == "full" || permute == "both"
        (ΔX, X) = H.C.inverse((ΔX, X))
    end
    return ΔX, X
end

# Input are two tensors ΔX, X
function backward_inv(ΔX, X, H::CouplingLayerHINT; scale=1, permute=nothing)
    isnothing(permute) ? permute = H.permute : permute = permute

    # Permutation
    if permute == "full" || permute == "both"
        (ΔX, X) = H.C.forward((ΔX, X))
    end
    ΔXa, ΔXb = tensor_split(ΔX)
    Xa, Xb = tensor_split(X)
    permute == "lower" && ((ΔXb, Xb) = H.C.forward((ΔXb, Xb)))

    # Check whether to continue recursion
    recursive = false
    if typeof(X) <: AbstractArray{Float32, 4} && size(X, 3) > 4
        recursive = true
    elseif typeof(X)<: AbstractArray{Float32, 5} && size(X, 4) > 4
        recursive = true
    end

    # Coupling layer backprop
    if recursive
        ΔY_temp, Y_temp = backward_inv(ΔXb, Xb, H; scale=scale+1, permute="none")
        ΔYa_temp, ΔYb, Yb = backward_inv(0f0.*ΔXa, ΔY_temp, Xa, Y_temp, H.CL[scale])[[1,2,4]]
        ΔYa, Ya = backward_inv(ΔXa+ΔYa_temp, Xa, H; scale=scale+1, permute="none")
    else
        ΔYa = copy(ΔXa)
        Ya = copy(Xa)
        ΔYa_temp, ΔYb, Yb = backward_inv(0f0.*ΔYa, ΔXb, Xa, Xb, H.CL[scale])[[1,2,4]]
        ΔYa += ΔYa_temp
    end
    ΔY = tensor_cat(ΔYa, ΔYb)
    Y = tensor_cat(Ya, Yb)
    permute == "both" && ((ΔY, Y) = H.C.inverse((ΔY, Y)))
    return ΔY, Y
end


## Jacobian-related functions

function jacobian(ΔX, Δθ::Array{Parameter, 1}, X, H::CouplingLayerHINT; scale=1, permute=nothing, logdet=nothing)
    isnothing(logdet) ? logdet = (H.logdet && ~H.is_reversed) : logdet = logdet
    isnothing(permute) ? permute = H.permute : permute = permute

    # Permutation
    if permute == "full" || permute == "both"
        ΔX, X = H.C.jacobian(ΔX, Δθ[end-2:end], X)
    end
    Xa, Xb = tensor_split(X)
    ΔXa, ΔXb = tensor_split(ΔX)
    if permute == "lower"
        ΔXb, Xb = H.C.jacobian(ΔXb, Δθ[end-2:end], Xb)
    end

    # Initialize Gauss-Newton approx of logdet term
    if logdet
        nscales_tot = length(H.CL)
        nparams = 5*(nscales_tot-scale+1)
        (permute != "none") && (nparams += 3)
        GNΔθ_full = Array{Parameter, 1}(undef, nparams)
        (permute != "none") && (GNΔθ_full[end-2:end] = [Parameter(zeros(Float32, size(H.C.v1))), Parameter(zeros(Float32, size(H.C.v2))), Parameter(zeros(Float32, size(H.C.v3)))])
    end

    # Determine whether to continue recursion
    recursive = false
    if typeof(X) <: AbstractArray{Float32, 4} && size(X, 3) > 4
        recursive = true
    elseif typeof(X) <: AbstractArray{Float32, 5} && size(X, 4) > 4
        recursive = true
    end

    # HINT coupling
    idx_Δθ_scale = (scale-1)*5+1:scale*5
    if recursive
        # Call function recursively
        if logdet
            ΔYa, Ya, logdet1, GNΔθ1 = jacobian(ΔXa, Δθ, Xa, H; scale=scale+1, permute="none")
            ΔY_temp, Y_temp, logdet2, GNΔθ2 = jacobian(ΔXb, Δθ, Xb, H; scale=scale+1, permute="none")
            _, ΔYb, _, Yb, logdet3, GNΔθ3 = H.CL[scale].jacobian(ΔXa, ΔY_temp, Δθ[idx_Δθ_scale], Xa, Y_temp)
            logdet_full = logdet1 + logdet2 + logdet3
            GNΔθ_full[1:5] .= GNΔθ3
            (permute != "none") ? (GNΔθ_full[6:end-3] .= GNΔθ1 + GNΔθ2) : (GNΔθ_full[6:end] .= GNΔθ1 + GNΔθ2)
        else
            ΔYa, Ya = jacobian(ΔXa, Δθ, Xa, H; scale=scale+1, permute="none")
            ΔY_temp, Y_temp = jacobian(ΔXb, Δθ, Xb, H; scale=scale+1, permute="none")
            _, ΔYb, _, Yb = H.CL[scale].jacobian(ΔXa, ΔY_temp, Δθ[idx_Δθ_scale], Xa, Y_temp)
        end
    else
        # Coarsest scale
        Ya = copy(Xa)
        ΔYa = copy(ΔXa)
        if logdet
            _, ΔYb, _, Yb, logdet_full, GNΔθ3 = H.CL[scale].jacobian(ΔXa, ΔXb, Δθ[idx_Δθ_scale], Xa, Xb)
            GNΔθ_full[1:5] .= GNΔθ3
        else
            _, ΔYb, _, Yb = H.CL[scale].jacobian(ΔXa, ΔXb, Δθ[idx_Δθ_scale], Xa, Xb)
        end
    end

    Y = tensor_cat(Ya, Yb)
    ΔY = tensor_cat(ΔYa, ΔYb)
    if permute == "both"
        ΔY, Y = H.C.jacobianInverse(ΔY, Δθ[end-2:end], Y)
    end
    logdet ? (return ΔY, Y, logdet_full, GNΔθ_full) : (return ΔY, Y)

end

function adjointJacobian(ΔY, Y, H::CouplingLayerHINT; scale=1, logdet=nothing, permute=nothing)
    isnothing(logdet) ? logdet = (H.logdet && ~H.is_reversed) : logdet = logdet
    isnothing(permute) ? permute = H.permute : permute = permute

    # Initializing output parameter array
    nscales_tot = length(H.CL)
    nparams = 5*(nscales_tot-scale+1)
    (permute != "none") && (nparams += 3)
    Δθ = Array{Parameter, 1}(undef, nparams)
    logdet && (∇logdet = Array{Parameter, 1}(undef, nparams))

    if permute == "both"
        ΔY_ = deepcopy(ΔY)
        Y_ = deepcopy(Y)
        ΔY, Δθ_C, Y = H.C.adjointJacobianInverse(ΔY, Y)
        ΔY_, Y_ = H.C.forward((ΔY_, Y_))
        Δθ[end-2:end] .= Δθ_C
        logdet && (∇logdet[end-2:end] .= [Parameter(zeros(Float32, size(H.C.v1))), Parameter(zeros(Float32, size(H.C.v2))), Parameter(zeros(Float32, size(H.C.v3)))])
    end
    Ya, Yb = tensor_split(Y)
    ΔYa, ΔYb = tensor_split(ΔY)

    # Determine whether to continue recursion
    recursive = false
    if typeof(Y) <: AbstractArray{Float32, 4} && size(Y, 3) > 4
        recursive = true
    elseif typeof(Y) <: AbstractArray{Float32, 5} && size(Y, 4) > 4
        recursive = true
    end

    # HINT coupling
    if recursive
        if logdet
            ΔXa, Δθa, Xa, ∇logdet_a = H.adjointJacobian(ΔYa, Ya; scale=scale+1, permute="none")
            ΔXa_temp, ΔXb_temp, Δθ_scale, _, X_temp, ∇logdet_scale = H.CL[scale].adjointJacobian(ΔXa.*0f0, ΔYb, Xa, Yb)
            ΔXb, Δθb, Xb, ∇logdet_b = H.adjointJacobian(ΔXb_temp, X_temp; scale=scale+1, permute="none")
            ∇logdet[1:5] .= ∇logdet_scale
        else
            ΔXa, Δθa, Xa = H.adjointJacobian(ΔYa, Ya; scale=scale+1, permute="none")
            ΔXa_temp, ΔXb_temp, Δθ_scale, _, X_temp = H.CL[scale].adjointJacobian(ΔXa.*0f0, ΔYb, Xa, Yb)
            ΔXb, Δθb, Xb = H.adjointJacobian(ΔXb_temp, X_temp; scale=scale+1, permute="none")
        end
        ΔXa += ΔXa_temp
        Δθ[1:5] .= Δθ_scale
        if permute != "none"
            Δθ[6:5+length(Δθa)] .= Δθa+Δθb
            logdet && (∇logdet[6:5+length(∇logdet_a)] .= ∇logdet_a+∇logdet_b)
        else
            Δθ[6:5+length(Δθa)] .= Δθa+Δθb
            logdet && (∇logdet[6:5+length(∇logdet_a)] .= ∇logdet_a+∇logdet_b)
        end
    else
        Xa = copy(Ya)
        ΔXa = copy(ΔYa)
        if logdet
            ΔXa_, ΔXb, Δθ_scale, _, Xb, ∇logdet_scale = H.CL[scale].adjointJacobian(ΔYa.*0f0, ΔYb, Ya, Yb)
            ∇logdet[1:5] .= ∇logdet_scale
        else
            ΔXa_, ΔXb, Δθ_scale, _, Xb = H.CL[scale].adjointJacobian(ΔYa.*0f0, ΔYb, Ya, Yb)
        end
        ΔXa += ΔXa_
        Δθ[1:5] .= Δθ_scale
    end
    if permute == "lower"
        ΔXb, Δθ_C, Xb = H.C.adjointJacobian(ΔXb, Xb)
        Δθ[end-2:end] .= Δθ_C
        logdet && (∇logdet[end-2:end] .= [Parameter(zeros(Float32, size(H.C.v1))), Parameter(zeros(Float32, size(H.C.v2))), Parameter(zeros(Float32, size(H.C.v3)))])
    end
    ΔX = tensor_cat(ΔXa, ΔXb)
    X = tensor_cat(Xa, Xb)
    if permute == "full" || permute == "both"
        ΔX, Δθ_C, X = H.C.adjointJacobian(ΔX, X)
        if permute == "full"
            Δθ[end-2:end] .= Δθ_C
            logdet && (∇logdet[end-2:end] .= [Parameter(zeros(Float32, size(H.C.v1))), Parameter(zeros(Float32, size(H.C.v2))), Parameter(zeros(Float32, size(H.C.v3)))])
        else
            Δθ[end-2:end] += Δθ_C
            logdet && (∇logdet[end-2:end] += [Parameter(zeros(Float32, size(H.C.v1))), Parameter(zeros(Float32, size(H.C.v2))), Parameter(zeros(Float32, size(H.C.v3)))])
        end
    end

    H.logdet ? (return ΔX, Δθ, X, ∇logdet) : (return ΔX, Δθ, X)
end


## Other utils

# Clear gradients
function clear_grad!(H::CouplingLayerHINT)
    for j=1:length(H.CL)
        clear_grad!(H.CL[j])
    end
    ~isnothing(H.C) && clear_grad!(H.C)
end

# Get parameters
function get_params(H::CouplingLayerHINT)
    nlayers = length(H.CL)
    p = get_params(H.CL[1])
    if nlayers > 1
        for j=2:nlayers
            p = cat(p, get_params(H.CL[j]); dims=1)
        end
    end
    ~isnothing(H.C) && (p = cat(p, get_params(H.C); dims=1))
    return p
end

# Set is_reversed flag in full network tree
function tag_as_reversed!(H::CouplingLayerHINT, tag::Bool)
    H.is_reversed = tag
    nlayers = length(H.CL)
    for j=1:nlayers
        H.CL[j].is_reversed = tag
    end
    return H
end
