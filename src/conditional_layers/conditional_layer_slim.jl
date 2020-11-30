# Invertible conditional HINT layer from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export ConditionalLayerSLIM

"""
    CI = ConditionalLayerSLIM(nx1, nx2, nx_in, nx_hidden, ny1, ny2, ny_in, ny_hidden, batchsize, Op;
        type="affine", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

 Create a conditional SLIM layer based on the HINT architecture.

 *Input*: 

 - `nx1`, `nx2`: spatial dimensions of `X`

 - `nx_in`, `nx_hidden`: number of input and hidden channels of `X`

 - `ny1`, `ny2`: spatial dimensions of `Y`
 
 - `ny_in`, `ny_hidden`: number of input and hidden channels of `Y`

 - `Op`: Linear forward modeling operator

 - `type`: string to indicate which type of data coupling layer to use (`"additive"`, `"affine"`, `"learned"`)

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 *Output*:
 
 - `CI`: Conditional SLIM coupling layer.

 *Usage:*

 - Forward mode: `Zx, Zy, logdet = CI.forward_X(X, Y, Op)`

 - Inverse mode: `X, Y = CI.inverse(Zx, Zy, Op)`

 - Backward mode: `ΔX, ΔY, X, Y = CI.backward(ΔZx, ΔZy, Zx, Zy, Op)`

 - Forward mode Y: `Zy = CI.forward_Y(Y)`

 - Inverse mode Y: `Y = CI.inverse(Zy)`

 *Trainable parameters:*

 - None in `CI` itself

 - Trainable parameters in coupling layers `CI.CL_X`, `CI.CL_Y`, `CI.CL_XY` and in
   permutation layers `CI.C_X` and `CI.C_Y`.

 See also: [`CouplingLayerHINT`](@ref), [`AffineCouplingLayerSLIM`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct ConditionalLayerSLIM <: NeuralNetLayer
    CL_X::CouplingLayerHINT
    CL_Y::CouplingLayerHINT
    CL_XY::Union{AdditiveCouplingLayerSLIM, AffineCouplingLayerSLIM, LearnedCouplingLayerSLIM}
    C_X::Conv1x1
    C_Y::Conv1x1
end

@Flux.functor ConditionalLayerSLIM

# Constructor from input dimensions
function ConditionalLayerSLIM(nx1::Int64, nx2::Int64, nx_in::Int64, nx_hidden::Int64, ny1::Int64, ny2::Int64, ny_in::Int64, ny_hidden::Int64,
    batchsize::Int64; type="affine", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)

    # Create basic coupling layers
    CL_X = CouplingLayerHINT(nx1, nx2, nx_in, nx_hidden, batchsize; 
        k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true)
    CL_Y = CouplingLayerHINT(Int(ny1/2), Int(ny2/2), Int(ny_in*4), ny_hidden, batchsize; 
        k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true)

    if type == "affine"
        CL_XY = AffineCouplingLayerSLIM(nx1, nx2, nx_in, nx_hidden, batchsize, identity;
            k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true, permute=false)
    elseif type == "additive"
        CL_XY = AdditiveCouplingLayerSLIM(nx1, nx2, nx_in, nx_hidden, batchsize, identity;
            k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true, permute=false)
    elseif type == "learned"
        CL_XY = LearnedCouplingLayerSLIM( nx1, nx2, nx_in, ny1, ny2, ny_in, nx_hidden, batchsize;
            k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true, permute=false)
    else
        throw("Specified layer type not defined.")
    end

    # Permutation using 1x1 convolution
    C_X = Conv1x1(nx_in)
    C_Y = Conv1x1(Int(ny_in*4))

    return ConditionalLayerSLIM(CL_X, CL_Y, CL_XY, C_X, C_Y)
end

function forward(X, Y, Op, CI::ConditionalLayerSLIM)

    # Y-lane: coupling
    Ys = wavelet_squeeze(Y)
    Yp = CI.C_Y.forward(Ys)
    Zy, logdet2 = CI.CL_Y.forward(Yp)
    Zy = wavelet_unsqueeze(Zy)

    # X-lane
    Xp = CI.C_X.forward(X)
    X, logdet1 = CI.CL_X.forward(Xp)
    if typeof(CI.CL_XY) == LearnedCouplingLayerSLIM
        Zx, logdet3 = CI.CL_XY.forward(X, reshape(Y, :, size(Y, 4)))   # Learn operator
    else
        Zx, logdet3 = CI.CL_XY.forward(X, reshape(Y, :, size(Y, 4)), Op)   # Use provided operator
    end
    logdet = logdet1 + logdet2 + logdet3
    return Zx, Zy, logdet
end

function inverse(Zx, Zy, Op, CI::ConditionalLayerSLIM)

    # Y-lane
    Zy = wavelet_squeeze(Zy)
    Yp = CI.CL_Y.inverse(Zy)
    Ys = CI.C_Y.inverse(Yp)
    Y = wavelet_unsqueeze(Ys)

    # X-lane
    if typeof(CI.CL_XY) == LearnedCouplingLayerSLIM
        X = CI.CL_XY.inverse(Zx, reshape(Y, :, size(Y, 4)))
    else
        X = CI.CL_XY.inverse(Zx, reshape(Y, :, size(Y, 4)), Op)
    end
    Xp = CI.CL_X.inverse(X)
    X = CI.C_X.inverse(Xp)

    return X, Y
end

function backward(ΔZx, ΔZy, Zx, Zy, Op, CI::ConditionalLayerSLIM; set_grad::Bool=true)

    # Y-lane
    ΔZy = wavelet_squeeze(ΔZy)
    Zy = wavelet_squeeze(Zy)
    if set_grad
        ΔYp, Yp = CI.CL_Y.backward(ΔZy, Zy)
        ΔYs, Ys = CI.C_Y.inverse((ΔYp, Yp))
    else
        ΔYp, Δθ_CLY, Yp, ∇logdet_CLY = CI.CL_Y.backward(ΔZy, Zy; set_grad=set_grad)
        ΔYs, Δθ_CY, Ys = CI.C_Y.inverse((ΔYp, Yp); set_grad=set_grad)
    end
    Y = wavelet_unsqueeze(Ys)
    ΔY = wavelet_unsqueeze(ΔYs)

    # X-lane
    if typeof(CI.CL_XY) == LearnedCouplingLayerSLIM
        if set_grad
            ΔX, ΔY_, X = CI.CL_XY.backward(ΔZx, Zx, reshape(Y, :, size(Y, 4)))
        else
            ΔX, ΔY_, Δθ_CLXY, X = CI.CL_XY.backward(ΔZx, Zx, reshape(Y, :, size(Y, 4)); set_grad=set_grad)
        end
    else
        if set_grad
            ΔX, ΔY_, X = CI.CL_XY.backward(ΔZx, Zx, reshape(Y, :, size(Y, 4)), Op)
        else
            ΔX, ΔY_, Δθ_CLXY, X = CI.CL_XY.backward(ΔZx, Zx, reshape(Y, :, size(Y, 4)), Op; set_grad=set_grad)
        end
    end
    ΔY += reshape(ΔY_, size(ΔY))
    if set_grad
        ΔXp, Xp = CI.CL_X.backward(ΔX, X)
        ΔX, X = CI.C_X.inverse((ΔXp, Xp))
    else
        ΔXp, Δθ_CLX, Xp, ∇logdet_CLX = CI.CL_X.backward(ΔX, X; set_grad=set_grad)
        ΔX, Δθ_CX, X = CI.C_X.inverse((ΔXp, Xp); set_grad=set_grad)
    end

    set_grad ? (return ΔX, ΔY, X, Y) : (return ΔX, ΔY, cat(Δθ_CLX, Δθ_CLY, Δθ_CLXY, Δθ_CX, Δθ_CY; dims=1), X, Y, cat(∇logdet_CLX, ∇logdet_CLY, 0f0.*Δθ_CLXY, 0f0.*Δθ_CX, 0f0.*Δθ_CY; dims=1))
end


## Jacobian-related utils

function jacobian(ΔX, ΔY, Δθ::Array{Parameter, 1}, X, Y, Op, CI::ConditionalLayerSLIM)

    # Selecting parameters
    npars_cl = Int64((length(Δθ)-6)/3)
    Δθ_CLX = Δθ[1:npars_cl]
    Δθ_CLY = Δθ[1+npars_cl:2*npars_cl]
    Δθ_CLXY = Δθ[1+2*npars_cl:3*npars_cl]
    Δθ_CX = Δθ[1+3*npars_cl:3*npars_cl+3]
    Δθ_CY = Δθ[3*npars_cl+4:end]

    # Y-lane: coupling
    Ys = wavelet_squeeze(Y)
    ΔYs = wavelet_squeeze(ΔY)
    ΔYp, Yp = CI.C_Y.jacobian(ΔYs, Δθ_CY, Ys)
    ΔZy, Zy, logdet2, GNΔθ_CLY = CI.CL_Y.jacobian(ΔYp, Δθ_CLY, Yp)
    Zy = wavelet_unsqueeze(Zy)
    ΔZy = wavelet_unsqueeze(ΔZy)

    # X-lane
    ΔXp, Xp = CI.C_X.jacobian(ΔX, Δθ_CX, X)
    ΔX, X, logdet1, GNΔθ_CLX = CI.CL_X.jacobian(ΔXp, Δθ_CLX, Xp)
    if typeof(CI.CL_XY) == LearnedCouplingLayerSLIM
        ΔZx, Zx, logdet3 = CI.CL_XY.jacobian(ΔX, reshape(ΔY, :, size(ΔY, 4)), Δθ_CLXY, X, reshape(Y, :, size(Y, 4)))   # Learn operator
    else
        ΔZx, Zx, logdet3 = CI.CL_XY.jacobian(ΔX, reshape(ΔY, :, size(Y, 4)), Δθ_CLXY, X, reshape(Y, :, size(Y, 4)), Op)   # Use provided operator
    end

    logdet = logdet1 + logdet2 + logdet3
    return ΔZx, ΔZy, Zx, Zy, logdet, cat(GNΔθ_CLX, GNΔθ_CLY, 0f0.*Δθ_CLXY, 0f0.*Δθ_CX, 0f0.*Δθ_CY; dims=1)

end

adjointJacobian(ΔZx, ΔZy, Zx, Zy, Op, CI::ConditionalLayerSLIM) = backward(ΔZx, ΔZy, Zx, Zy, Op, CI; set_grad=false)


## Other utils

function forward_Y(Y, CI::ConditionalLayerSLIM)
    Ys = wavelet_squeeze(Y)
    Yp = CI.C_Y.forward(Ys)
    Zy, logdet2 = CI.CL_Y.forward(Yp)
    Zy = wavelet_unsqueeze(Zy)
    return Zy
end

function inverse_Y(Zy, CI::ConditionalLayerSLIM)
    Zy = wavelet_squeeze(Zy)
    Yp = CI.CL_Y.inverse(Zy)
    Ys = CI.C_Y.inverse(Yp)
    Y = wavelet_unsqueeze(Ys)
    return Y
end

# Clear gradients
function clear_grad!(CI::ConditionalLayerSLIM)
    clear_grad!(CI.CL_X)
    clear_grad!(CI.CL_Y)
    clear_grad!(CI.CL_XY)
    clear_grad!(CI.C_X)
    clear_grad!(CI.C_Y)
end

# Get parameters
function get_params(CI::ConditionalLayerSLIM)
    p = get_params(CI.CL_X)
    p = cat(p, get_params(CI.CL_Y); dims=1)
    p = cat(p, get_params(CI.CL_XY); dims=1)
    p = cat(p, get_params(CI.C_X); dims=1)
    p = cat(p, get_params(CI.C_Y); dims=1)
    return p
end
