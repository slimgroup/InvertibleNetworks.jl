# Invertible conditional HINT network from Kruse et. al (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020

export NetworkMultiScaleConditionalHINT, NetworkMultiScaleConditionalHINT3D

"""
    CH = NetworkMultiScaleConditionalHINT(n_in, n_hidden, L, K; split_scales=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

    CH = NetworkMultiScaleConditionalHINT3D(n_in, n_hidden, L, K; split_scales=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

 Create a conditional HINT network for data-driven generative modeling based
 on the change of variables formula.

 *Input*: 

 - 'n_in': number of input channels
 
 - `n_hidden`: number of hidden units in residual blocks

 - `L`: number of scales (outer loop)

 - `K`: number of flow steps per scale (inner loop)

 - `split_scales`: if true, split output in half along channel dimension after each scale. Feed one half through the next layers,
    while saving the remaining channels for the output.

 - `k1`, `k2`: kernel size for first and third residual layer (`k1`) and second layer (`k2`)

 - `p1`, `p2`: respective padding sizes for residual block layers
 
 - `s1`, `s2`: respective strides for residual block layers

 - `ndims` : number of dimensions

 *Output*:
 
 - `CH`: conditional HINT network

 *Usage:*

 - Forward mode: `Zx, Zy, logdet = CH.forward(X, Y)`

 - Inverse mode: `X, Y = CH.inverse(Zx, Zy)`

 - Backward mode: `ΔX, X = CH.backward(ΔZx, ΔZy, Zx, Zy)`

 *Trainable parameters:*

 - None in `CH` itself

 - Trainable parameters in activation normalizations `CH.AN_X[i]` and `CH.AN_Y[i]`, 
 and in coupling layers `CH.CL[i]`, where `i` ranges from `1` to `depth`.

 See also: [`ActNorm`](@ref), [`ConditionalLayerHINT!`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
mutable struct NetworkMultiScaleConditionalHINT <: InvertibleNetwork
    AN_X::AbstractArray{ActNorm, 2}
    AN_Y::AbstractArray{ActNorm, 2}
    CL::AbstractArray{ConditionalLayerHINT, 2}
    XY_dims::Union{Array{Tuple, 1}, Nothing}
    L::Int64
    K::Int64
    split_scales::Bool
    logdet::Bool
    is_reversed::Bool
end

@Flux.functor NetworkMultiScaleConditionalHINT

# Constructor
function NetworkMultiScaleConditionalHINT(n_in::Int64, n_hidden::Int64, L::Int64, K::Int64;
                                          split_scales=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=true, ndims=2)

    AN_X = Array{ActNorm}(undef, L, K)
    AN_Y = Array{ActNorm}(undef, L, K)
    CL = Array{ConditionalLayerHINT}(undef, L, K)
    if split_scales
        XY_dims = Array{Tuple}(undef, L-1)
        channel_factor = 2
    else
        XY_dims = nothing
        channel_factor = 4
    end

    # Create layers
    for i=1:L
        for j=1:K
            AN_X[i, j] = ActNorm(n_in*4; logdet=logdet)
            AN_Y[i, j] = ActNorm(n_in*4; logdet=logdet)
            CL[i, j] = ConditionalLayerHINT(n_in*4, n_hidden; permute=true, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=logdet, ndims=ndims)
        end
        n_in *= channel_factor
    end

    return NetworkMultiScaleConditionalHINT(AN_X, AN_Y, CL, XY_dims, L, K, split_scales, logdet, false)
end

NetworkMultiScaleConditionalHINT3D(args...;kw...) = NetworkMultiScaleConditionalHINT(args...; kw..., ndims=3)

# Forward pass and compute logdet
<<<<<<< HEAD
function forward(X::AbstractArray{T, N}, Y::AbstractArray{T, N}, CH::NetworkMultiScaleConditionalHINT) where {T, N}
    CH.split_scales && (XY_save = array_of_array(X, CH.L-1, 2))
    logdet = 0
=======
function forward(X, Y, CH::NetworkMultiScaleConditionalHINT; logdet=nothing)
    isnothing(logdet) ? logdet = (CH.logdet && ~CH.is_reversed) : logdet = logdet

    CH.split_scales && (XY_save = Array{Array}(undef, CH.L-1, 2))
    logdet_ = 0f0
>>>>>>> make CHmulti reversible and proper logdet
    for i=1:CH.L
        X = wavelet_squeeze(X)
        Y = wavelet_squeeze(Y)
        for j=1:CH.K
	    logdet ? (X_, logdet1) = CH.AN_X[i, j].forward(X) : X_ = CH.AN_X[i, j].forward(X)
            logdet ? (Y_, logdet2) = CH.AN_Y[i, j].forward(Y) : Y_ = CH.AN_Y[i, j].forward(Y)
	    logdet ? (X, Y, logdet3) = CH.CL[i, j].forward(X_, Y_) : (X, Y) = CH.CL[i, j].forward(X_, Y_)
            logdet && (logdet_ += (logdet1 + logdet2 + logdet3)) 
        end
        if CH.split_scales && i < CH.L    # don't split after last iteration
            X, Zx = tensor_split(X)
            Y, Zy = tensor_split(Y)
            XY_save[i, :] = [Zx, Zy]
            CH.XY_dims[i] = size(Zx)
        end
    end
    CH.split_scales && ((X, Y) = cat_states(XY_save, X, Y))
    
    logdet ? (return X, Y, logdet_) : (return X, Y)
end

# Inverse pass and compute gradients
function inverse(Zx::AbstractArray{T, N}, Zy::AbstractArray{T, N}, CH::NetworkMultiScaleConditionalHINT; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (CH.logdet && CH.is_reversed) : logdet = logdet

    CH.split_scales && ((XY_save, Zx, Zy) = split_states(CH.XY_dims, Zx, Zy))
    logdet_ = 0f0
    for i=CH.L:-1:1
        if CH.split_scales && i < CH.L
            Zx = tensor_cat(Zx, XY_save[i, 1])
            Zy = tensor_cat(Zy, XY_save[i, 2])
        end
        for j=CH.K:-1:1
            logdet ? (Zx_, Zy_, logdet1) = CH.CL[i, j].inverse(Zx, Zy; logdet=true) : (Zx_, Zy_) = CH.CL[i, j].inverse(Zx, Zy; logdet=false)
            logdet ? (Zy, logdet2) = CH.AN_Y[i, j].inverse(Zy_; logdet=true) : Zy = CH.AN_Y[i, j].inverse(Zy_; logdet=false)
            logdet ? (Zx, logdet3) = CH.AN_X[i, j].inverse(Zx_; logdet=true) : Zx = CH.AN_X[i, j].inverse(Zx_; logdet=false)
            logdet && (logdet_ += (logdet1 + logdet2 + logdet3))
        end
        Zx = wavelet_unsqueeze(Zx)
        Zy = wavelet_unsqueeze(Zy)
    end
    logdet ? (return Zx, Zy, logdet_) : (return Zx, Zy)
end

# Backward pass and compute gradients
function backward(ΔZx::AbstractArray{T, N}, ΔZy::AbstractArray{T, N}, Zx::AbstractArray{T, N}, Zy::AbstractArray{T, N}, CH::NetworkMultiScaleConditionalHINT; set_grad::Bool=true) where {T, N}

    # Split data and gradients
    if CH.split_scales
        ΔXY_save, ΔZx, ΔZy = split_states(CH.XY_dims, ΔZx, ΔZy)
        XY_save, Zx, Zy = split_states(CH.XY_dims, Zx, Zy)
    end

    if ~set_grad
        Δθ = Array{Parameter, 1}(undef, 0)
        ∇logdet = Array{Parameter, 1}(undef, 0)
    end

    for i=CH.L:-1:1
        if CH.split_scales && i < CH.L
            ΔZx = tensor_cat(ΔZx, ΔXY_save[i, 1])
            ΔZy = tensor_cat(ΔZy, ΔXY_save[i, 2])
            Zx = tensor_cat(Zx, XY_save[i, 1])
            Zy = tensor_cat(Zy, XY_save[i, 2])
        end
        for j=CH.K:-1:1
            if set_grad
                ΔZx_, ΔZy_, Zx_, Zy_ = CH.CL[i, j].backward(ΔZx, ΔZy, Zx, Zy)
                ΔZx, Zx = CH.AN_X[i, j].backward(ΔZx_, Zx_)
                ΔZy, Zy = CH.AN_Y[i, j].backward(ΔZy_, Zy_)
            else
                if CH.logdet
                    ΔZx_, ΔZy_, Δθcl, Zx_, Zy_, ∇logdetcl = CH.CL[i, j].backward(ΔZx, ΔZy, Zx, Zy; set_grad=set_grad)
                    ΔZx, Δθx, Zx, ∇logdetx = CH.AN_X[i, j].backward(ΔZx_, Zx_; set_grad=set_grad)
                    ΔZy, Δθy, Zy, ∇logdety = CH.AN_Y[i, j].backward(ΔZy_, Zy_; set_grad=set_grad)
                    ∇logdet = cat(∇logdetx, ∇logdety, ∇logdetcl, ∇logdet; dims=1)
                else
                    ΔZx_, ΔZy_, Δθcl, Zx_, Zy_ = CH.CL[i, j].backward(ΔZx, ΔZy, Zx, Zy; set_grad=set_grad)
                    ΔZx, Δθx, Zx = CH.AN_X[i, j].backward(ΔZx_, Zx_; set_grad=set_grad)
                    ΔZy, Δθy, Zy = CH.AN_Y[i, j].backward(ΔZy_, Zy_; set_grad=set_grad)
                end
                Δθ = cat(Δθx, Δθy, Δθcl, Δθ; dims=1)
            end
        end
        ΔZx = wavelet_unsqueeze(ΔZx)
        ΔZy = wavelet_unsqueeze(ΔZy)
        Zx = wavelet_unsqueeze(Zx)
        Zy = wavelet_unsqueeze(Zy)
    end
    if set_grad
        print("returned 4")
        return ΔZx, ΔZy, Zx, Zy
    else
        CH.logdet ? (return ΔZx, ΔZy, Δθ, Zx, Zy, ∇logdet) : (return ΔZx, ΔZy, Δθ, Zx, Zy)
    end
end

# Backward reverse pass and compute gradients
function backward_inv(ΔX, ΔY, X, Y, CH::NetworkMultiScaleConditionalHINT)
    for i=1:CH.L
 	    ΔX = wavelet_squeeze(ΔX)
        ΔY = wavelet_squeeze(ΔY)    
	    X  = wavelet_squeeze(X)
        Y  = wavelet_squeeze(Y)    
    	for j=1:CH.K
            ΔX_, X_ = backward_inv(ΔX, X, CH.AN_X[i, j])
            ΔY_, Y_ = backward_inv(ΔY, Y, CH.AN_Y[i, j])
            ΔX, ΔY, X, Y = backward_inv(ΔX_, ΔY_, X_, Y_, CH.CL[i, j])
       end
    end
    return ΔX, ΔY, X, Y
end

# Forward pass and compute logdet
function forward_Y(Y::AbstractArray{T, N}, CH::NetworkMultiScaleConditionalHINT) where {T, N}
    CH.split_scales && (Y_save = array_of_array(Y, CH.L-1))
    for i=1:CH.L
        Y = wavelet_squeeze(Y)
        for j=1:CH.K
            Y_ = CH.AN_Y[i, j].forward(Y; logdet=false)
            Y  = CH.CL[i, j].forward_Y(Y_)
        end
        if CH.split_scales && i < CH.L    # don't split after last iteration
            Y, Zy = tensor_split(Y)
            Y_save[i] = Zy
            CH.XY_dims[i] = size(Zy)
        end
    end
    CH.split_scales && (Y = cat_states(Y_save, Y))
    return Y

end

# Inverse pass and compute gradients
function inverse_Y(Zy::AbstractArray{T, N}, CH::NetworkMultiScaleConditionalHINT) where {T, N}
    CH.split_scales && ((Y_save, Zy) = split_states(CH.XY_dims, Zy))
    for i=CH.L:-1:1
        if CH.split_scales && i < CH.L
            Zy = tensor_cat(Zy, Y_save[i])
        end
        for j=CH.K:-1:1
            Zy_ = CH.CL[i, j].inverse_Y(Zy)
            Zy = CH.AN_Y[i, j].inverse(Zy_; logdet=false)
        end
        Zy = wavelet_unsqueeze(Zy)
    end
    return Zy
end


## Jacobian-related utils


function jacobian(ΔX::AbstractArray{T, N}, ΔY::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, X, Y, CH::NetworkMultiScaleConditionalHINT; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (CH.logdet && ~CH.is_reversed) : logdet = logdet

    if CH.split_scales
        XY_save = array_of_array(ΔX,  CH.L-1, 2)
        ΔXY_save = array_of_array(ΔX,  CH.L-1, 2)
    end

    logdet_ = 0f0
    logdet && (GNΔθ = Array{Parameter, 1}(undef, 0))
    
    idxblk = 0
    for i=1:CH.L
        X = wavelet_squeeze(X)
        ΔX = wavelet_squeeze(ΔX)
        Y = wavelet_squeeze(Y)
        ΔY = wavelet_squeeze(ΔY)
        for j=1:CH.K
	    if logdet
                ΔX_, X_ = CH.AN_X[i, j].jacobian(ΔX, Δθj[1:2], X)
                ΔY_, Y_ = CH.AN_Y[i, j].jacobian(ΔY, Δθj[3:4], Y)
                ΔX, ΔY, X, Y = CH.CL[i, j].jacobian(ΔX_, ΔY_, Δθj[5:end], X_, Y_)
	    else 
                npars_ij = 4+length(get_params(CH.CL[i, j]))
                Δθij = Δθ[idxblk+1:idxblk+npars_ij]
                ΔX_, X_, logdet1, GNΔθ1 = CH.AN_X[i, j].jacobian(ΔX, Δθij[1:2], X)
                ΔY_, Y_, logdet2, GNΔθ2 = CH.AN_Y[i, j].jacobian(ΔY, Δθij[3:4], Y)
                ΔX, ΔY, X, Y, logdet3, GNΔθ3 = CH.CL[i, j].jacobian(ΔX_, ΔY_, Δθij[5:end], X_, Y_)
                logdet_ += (logdet1 + logdet2 + logdet3)
                GNΔθ = cat(GNΔθ, GNΔθ1, GNΔθ2, GNΔθ3; dims=1)
                idxblk += npars_ij
       	    end
        end
        if CH.split_scales && i < CH.L    # don't split after last iteration
            X, Zx = tensor_split(X)
            ΔX, ΔZx = tensor_split(ΔX)
            Y, Zy = tensor_split(Y)
            ΔY, ΔZy = tensor_split(ΔY)
            XY_save[i, :] = [Zx, Zy]
            ΔXY_save[i, :] = [ΔZx, ΔZy]
            CH.XY_dims[i] = size(Zx)
        end
    end
    if CH.split_scales
        X, Y = cat_states(XY_save, X, Y)
        ΔX, ΔY = cat_states(ΔXY_save, ΔX, ΔY)
    end
   
    logdet ? (return ΔX, ΔY, X, Y, logdet_, GNΔθ) : (return ΔX, ΔY, X, Y)
end

adjointJacobian(ΔZx::AbstractArray{T, N}, ΔZy::AbstractArray{T, N}, Zx::AbstractArray{T, N}, Zy::AbstractArray{T, N}, CH::NetworkMultiScaleConditionalHINT) where {T, N} = backward(ΔZx, ΔZy, Zx, Zy, CH; set_grad=false)


## Other utils

# Clear gradients
function clear_grad!(CH::NetworkMultiScaleConditionalHINT)
    L, K = size(CH.CL)
    for i = 1:L
        for j = 1:K
            clear_grad!(CH.AN_X[i, j])
            clear_grad!(CH.AN_Y[i, j])
            clear_grad!(CH.CL[i, j])
        end
    end
end

# Get parameters
function get_params(CH::NetworkMultiScaleConditionalHINT)
    L, K = size(CH.CL)
    p = Array{Parameter, 1}(undef, 0)
    for i = 1:L
        for j = 1:K
            p = cat(p, get_params(CH.AN_X[i, j]); dims=1)
            p = cat(p, get_params(CH.AN_Y[i, j]); dims=1)
            p = cat(p, get_params(CH.CL[i, j]); dims=1)
        end
    end
    return p
end


# Set is_reversed flag in full network tree
function tag_as_reversed!(CH::NetworkMultiScaleConditionalHINT, tag::Bool)
    L, K = size(CH.CL)
    CH.is_reversed = tag
    for i = 1:L
        for j = 1:K
            tag_as_reversed!(CH.AN_X[i, j], tag)
            tag_as_reversed!(CH.AN_Y[i, j], tag)
            tag_as_reversed!(CH.CL[i, j], tag)
        end
    end

    return CH
end
