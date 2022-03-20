# Invertible conditional HINT multiscale network from Kruse et. al (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020

export NetworkMultiScaleConditionalHINT_ciin

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
mutable struct NetworkMultiScaleConditionalHINT_ciin <: InvertibleNetwork
    AN_X::AbstractArray{ActNorm, 2}
    AN_Y::AbstractArray{ActNorm, 2}
    CL::AbstractArray{CondCouplingLayerSpade_additive, 2}
    XY_dims::Union{Array{Array, 1}, Nothing}
    L::Int64
    K::Int64
    split_scales::Bool
    logdet::Bool
    is_reversed::Bool
    squeezer::Squeezer
end

@Flux.functor NetworkMultiScaleConditionalHINT_ciin

# Constructor
function NetworkMultiScaleConditionalHINT_ciin(n_in::Int64, n_hidden::Int64, L::Int64, K::Int64; max_recursion=nothing,
                                          split_scales=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=true, ndims=2, squeezer::Squeezer=ShuffleLayer(), activation::ActivationFunction=SigmoidLayer())

    AN_X = Array{ActNorm}(undef, L, K)
    AN_Y = Array{ActNorm}(undef, L, K)
    CL = Array{CondCouplingLayerSpade_additive}(undef, L, K)
    if split_scales
        XY_dims = fill!(Array{Array}(undef, L-1), [1,1]) #fill in with dummy values so that |> gpu accepts it
        channel_factor = 2
    else
        XY_dims = nothing
        channel_factor = 4
    end

    n_cond = deepcopy(n_in) 

    # Create layers
    for i=1:L
        for j=1:K
            AN_X[i, j] = ActNorm(n_in*4; logdet=logdet)
            AN_Y[i, j] = ActNorm(n_cond*4; logdet=logdet)
            CL[i, j]   = CondCouplingLayerSpade_additive(n_in*4, n_cond*4, n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=logdet,  activation=activation, ndims=ndims)
        end
        n_in *= channel_factor
        n_cond *= channel_factor*2
    end

    return NetworkMultiScaleConditionalHINT_ciin(AN_X, AN_Y, CL, XY_dims, L, K, split_scales, logdet, false, squeezer)
end



# Forward pass and compute logdet
function forward(X::AbstractArray{T, N}, Y::AbstractArray{T, N}, CH::NetworkMultiScaleConditionalHINT_ciin; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (CH.logdet && ~CH.is_reversed) : logdet = logdet


    CH.split_scales && (XY_save = array_of_array(X, CH.L-1))

    logdet_ = 0f0

    for i=1:CH.L
        X = CH.squeezer.forward(X)
        Y = CH.squeezer.forward(Y)

        for j=1:CH.K
            logdet ? (X_, logdet1)   = CH.AN_X[i, j].forward(X) : X_ = CH.AN_X[i, j].forward(X)
            logdet ? (Y_, logdet2)   = CH.AN_Y[i, j].forward(Y) : Y_ = CH.AN_Y[i, j].forward(Y)
            logdet ? (X, logdet3 ) = CH.CL[i, j].forward(X_, Y_) : (X) = CH.CL[i, j].forward(X_, Y_)
            logdet && (logdet_ += (logdet1 + logdet3)) 
        end
        if CH.split_scales && i < CH.L    # don't split after last iteration
            X, Zx = tensor_split(X)
            XY_save[i] = Zx
            CH.XY_dims[i] = collect(size(Zx))
        end
    end

    CH.split_scales && ((X) = cat_states(XY_save, X))


    
    logdet ? (return X, Y, logdet_) : (return X, Y)
end

# Inverse pass and compute gradients
function inverse(Zx::AbstractArray{T, N}, Zy::AbstractArray{T, M}, CH::NetworkMultiScaleConditionalHINT_ciin; logdet=nothing) where {T, N, M}
    isnothing(logdet) ? logdet = (CH.logdet && CH.is_reversed) : logdet = logdet

    CH.split_scales && ((XY_save, Zx) = split_states(Zx,  CH.XY_dims))
    logdet_ = 0f0
    for i=CH.L:-1:1
        if CH.split_scales && i < CH.L
            Zx = tensor_cat(Zx, XY_save[i, 1])
        end
        for j=CH.K:-1:1
            logdet ? (Zx_, logdet1) = CH.CL[i, j].inverse(Zx, Zy;) : (Zx_, Zy_) = CH.CL[i, j].inverse(Zx, Zy;)
            logdet ? (Zy, logdet2) = CH.AN_Y[i, j].inverse(Zy; logdet=true) : Zy = CH.AN_Y[i, j].inverse(Zy_; logdet=false)
            logdet ? (Zx, logdet3) = CH.AN_X[i, j].inverse(Zx_; logdet=true) : Zx = CH.AN_X[i, j].inverse(Zx_; logdet=false)
            logdet && (logdet_ += ( logdet1 + logdet3))
        end
        Zx = CH.squeezer.inverse(Zx) 
        Zy = CH.squeezer.inverse(Zy) 

    end
    logdet ? (return Zx, Zy, logdet_) : (return Zx, Zy)
end

# Backward pass and compute gradients
function backward(ΔZx::AbstractArray{T, N}, ΔZy::AbstractArray{T, M}, Zx::AbstractArray{T, N}, Zy::AbstractArray{T, M}, CH::NetworkMultiScaleConditionalHINT_ciin; set_grad::Bool=true) where {T, N, M}

    # Split data and gradients
    if CH.split_scales
        ΔXY_save, ΔZx = split_states(ΔZx,  CH.XY_dims)
        XY_save, Zx, = split_states(Zx, CH.XY_dims)
    end

    if ~set_grad
        Δθ = Array{Parameter, 1}(undef, 0)
        ∇logdet = Array{Parameter, 1}(undef, 0)
    end

    for i=CH.L:-1:1
        if CH.split_scales && i < CH.L
            ΔZx = tensor_cat(ΔZx, ΔXY_save[i, 1])
            Zx = tensor_cat(Zx, XY_save[i, 1])
        end
        for j=CH.K:-1:1
            if set_grad
                println("here")
                println(norm(ΔZx))
                println(norm(ΔZy))
                ΔZx_, Zx_, ΔZy_  = CH.CL[i, j].backward(ΔZx, Zx, 0f0 .* ΔZy, Zy)
                ΔZx, Zx = CH.AN_X[i, j].backward(ΔZx_, Zx_)
                ΔZy, Zy = CH.AN_Y[i, j].backward(ΔZy_, Zy)
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
        ΔZx = CH.squeezer.inverse(ΔZx)
        ΔZy = CH.squeezer.inverse(ΔZy)
        Zx  = CH.squeezer.inverse(Zx)
        Zy  = CH.squeezer.inverse(Zy)
    end
    if set_grad
        return ΔZx, ΔZy, Zx, Zy
    else
        CH.logdet ? (return ΔZx, ΔZy, Δθ, Zx, Zy, ∇logdet) : (return ΔZx, ΔZy, Δθ, Zx, Zy)
    end
end

# Backward reverse pass and compute gradients
function backward_inv(ΔX, ΔY, X, Y, CH::NetworkMultiScaleConditionalHINT_ciin)

    CH.split_scales && (XY_save = array_of_array(X, CH.L-1, 2))
    CH.split_scales && (ΔXY_save = array_of_array(ΔX, CH.L-1, 2))

    for i=1:CH.L
        ΔX = CH.squeezer.forward(ΔX)
        ΔY = CH.squeezer.forward(ΔY)
        X  = CH.squeezer.forward(X)
        Y  = CH.squeezer.forward(Y)
        for j=1:CH.K
            ΔX_, X_ = backward_inv(ΔX, X, CH.AN_X[i, j])
            ΔY_, Y_ = backward_inv(ΔY, Y, CH.AN_Y[i, j])
            ΔX, ΔY, X, Y = backward_inv(ΔX_, ΔY_, X_, Y_, CH.CL[i, j])
        end
        if CH.split_scales && i < CH.L    # don't split after last iteration
            X, Zx = tensor_split(X)
            Y, Zy = tensor_split(Y)
            XY_save[i, :] = [Zx, Zy]

            ΔX, ΔZx = tensor_split(ΔX)
            ΔY, ΔZy = tensor_split(ΔY)
            ΔXY_save[i, :] = [ΔZx, ΔZy]
            CH.XY_dims[i] = collect(size(ΔZx))
        end
    end

    CH.split_scales && ((X, Y) = cat_states(XY_save, X, Y))
    CH.split_scales && ((ΔX, ΔY) = cat_states(ΔXY_save, ΔX, ΔY))

    return ΔX, ΔY, X, Y
end

# Forward pass and compute logdet
function forward_Y(Y::AbstractArray{T, N}, CH::NetworkMultiScaleConditionalHINT_ciin; save_dims=false) where {T, N}
    CH.split_scales && (Y_save = array_of_array(Y, CH.L-1))

    for i=1:CH.L
        Y = CH.squeezer.forward(Y)
        for j=1:CH.K
            Y_ = CH.AN_Y[i, j].forward(Y; logdet=false)
            Y  = CH.CL[i, j].forward_Y(Y_)
        end
        if CH.split_scales && i < CH.L    # don't split after last iteration
            Y, Zy = tensor_split(Y)
            Y_save[i] = Zy
            save_dims && (CH.XY_dims[i] = collect(size(Zy))) 
        end
    end
    CH.split_scales && (Y = cat_states(Y_save, Y))
    return Y

end

# Inverse pass and compute gradients
function inverse_Y(Zy::AbstractArray{T, N}, CH::NetworkMultiScaleConditionalHINT_ciin) where {T, N}
    CH.split_scales && ((Y_save, Zy) = split_states(Zy, CH.XY_dims))

    for i=CH.L:-1:1
        if CH.split_scales && i < CH.L
            Zy = tensor_cat(Zy, Y_save[i])
        end
        for j=CH.K:-1:1
            Zy_ = CH.CL[i, j].inverse_Y(Zy)
            Zy = CH.AN_Y[i, j].inverse(Zy_; logdet=false)
        end
        Zy = CH.squeezer.inverse(Zy)
    end
    return Zy
end




## Other utils

# Clear gradients
function clear_grad!(CH::NetworkMultiScaleConditionalHINT_ciin)
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
function get_params(CH::NetworkMultiScaleConditionalHINT_ciin)
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
function tag_as_reversed!(CH::NetworkMultiScaleConditionalHINT_ciin, tag::Bool)
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