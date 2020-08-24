# 1x1 convolution operator using Householder matrices.
# Adapted from Putzky and Welling (2019): https://arxiv.org/abs/1911.10914
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020
#

export Conv1x1

"""
    C = Conv1x1(k; logdet=false)

 or

    C = Conv1x1(v1, v2, v3; logdet=false)

 Create network layer for 1x1 convolutions using Householder reflections.

 *Input*:

 - `k`: number of channels

 - `v1`, `v2`, `v3`: Vectors from which to construct matrix.

 - `logdet`: if true, returns logdet in forward pass (which is always zero)

 *Output*:

 - `C`: Network layer for 1x1 convolutions with Householder reflections.

 *Usage:*

 - Forward mode: `Y, logdet = C.forward(X)`

 - Backward mode: `ΔX, X = C.backward((ΔY, Y))`

 *Trainable parameters:*

 - Householder vectors `C.v1`, `C.v2`, `C.v3`

 See also: [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct Conv1x1 <: NeuralNetLayer
    k::Integer
    v1::Parameter
    v2::Parameter
    v3::Parameter
    logdet::Bool
end

@Flux.functor Conv1x1

# Constructor with random initializations
function Conv1x1(k; logdet=false)
    v1 = Parameter(glorot_uniform(k))
    v2 = Parameter(glorot_uniform(k))
    v3 = Parameter(glorot_uniform(k))
    return Conv1x1(k, v1, v2, v3, logdet)
end

function Conv1x1(v1, v2, v3; logdet=false)
    k = length(v1)
    v1 = Parameter(v1)
    v2 = Parameter(v2)
    v3 = Parameter(v3)
    return Conv1x1(k, v1, v2, v3, logdet)
end

function partial_derivative_outer(v::AbstractArray{Float32, 1})
    k = length(v)
    out1 = v * v'
    n = v' * v
    outer = cuzeros(v, k, k, k)
    for i=1:k
        copyto!(view(outer, i, :, :), out1)
    end
    broadcast!(*, outer, v, outer)
    broadcast!(*, outer, -2f0/n, outer)
    for j=1:k
        v1 = view(outer,j, :, j)
        broadcast!(+, v1, v1, v)
        v1 = view(outer,j, j, :)
        broadcast!(+, v1, v1, v)
    end
    broadcast!(*, outer, 1/n, outer)
    return outer
end

function conv1x1_grad_v(X::AbstractArray{Float32, 4}, ΔY::AbstractArray{Float32, 4}, C::Conv1x1; adjoint=false)

    # Reshape input
    nx, ny, n_in, batchsize = size(X)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data
    k = length(v1)

    dv1 = cuzeros(X, k)
    dv2 = cuzeros(X, k)
    dv3 = cuzeros(X, k)

    V1 = v1*v1'/(v1'*v1)
    V2 = v2*v2'/(v2'*v2)
    V3 = v3*v3'/(v3'*v3)

    for i=1:batchsize

        Xi = reshape(X[:,:,:,i], :, n_in)
        ΔYi = reshape(ΔY[:,:,:,i], :, n_in)

        dV1 =partial_derivative_outer(v1)
        dV2 =partial_derivative_outer(v2)
        dV3 =partial_derivative_outer(v3)

        for j=1:k

            ∂V1 = dV1[j, :, :] - 2f0*dV1[j, :, :]*V2 - 2f0*dV1[j, :, :]*V3 + 4f0*dV1[j, :, :]*V2*V3
            ∂V2 = dV2[j, :, :] - 2f0*V1*dV2[j, :, :] - 2f0*dV2[j, :, :]*V3 + 4f0*V1*dV2[j, :, :]*V3
            ∂V3 = dV3[j, :, :] - 2f0*V1*dV3[j, :, :] - 2f0*V2*dV3[j, :, :] + 4f0*V1*V2*dV3[j, :, :]

            if ~adjoint
                dv1[j] += sum(vec((-2f0*Xi*∂V1).*ΔYi)')
                dv2[j] += sum(vec((-2f0*Xi*∂V2).*ΔYi)')
                dv3[j] += sum(vec((-2f0*Xi*∂V3).*ΔYi)')
            else
                dv1[j] += sum(vec((-2f0*Xi*∂V1').*ΔYi)')
                dv2[j] += sum(vec((-2f0*Xi*∂V2').*ΔYi)')
                dv3[j] += sum(vec((-2f0*Xi*∂V3').*ΔYi)')
            end

        end
    end
    return dv1, dv2, dv3
end

function conv1x1_grad_v(X::AbstractArray{Float32, 5}, ΔY::AbstractArray{Float32, 5}, C::Conv1x1; adjoint=false)

    # Reshape input
    nx, ny, nz, n_in, batchsize = size(X)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data
    k = length(v1)

    dv1 = cuzeros(X, k)
    dv2 = cuzeros(X, k)
    dv3 = cuzeros(X, k)

    V1 = v1*v1'/(v1'*v1)
    V2 = v2*v2'/(v2'*v2)
    V3 = v3*v3'/(v3'*v3)

    for i=1:batchsize

        Xi = reshape(X[:,:,:,:,i], :, n_in)
        ΔYi = reshape(ΔY[:,:,:,:,i], :, n_in)

        dV1 =partial_derivative_outer(v1)
        dV2 =partial_derivative_outer(v2)
        dV3 =partial_derivative_outer(v3)

        for j=1:k

            ∂V1 = dV1[j, :, :] - 2f0*dV1[j, :, :]*V2 - 2f0*dV1[j, :, :]*V3 + 4f0*dV1[j, :, :]*V2*V3
            ∂V2 = dV2[j, :, :] - 2f0*V1*dV2[j, :, :] - 2f0*dV2[j, :, :]*V3 + 4f0*V1*dV2[j, :, :]*V3
            ∂V3 = dV3[j, :, :] - 2f0*V1*dV3[j, :, :] - 2f0*V2*dV3[j, :, :] + 4f0*V1*V2*dV3[j, :, :]


            if ~adjoint
                dv1[j] += sum(vec((-2f0*Xi*∂V1).*ΔYi)')
                dv2[j] += sum(vec((-2f0*Xi*∂V2).*ΔYi)')
                dv3[j] += sum(vec((-2f0*Xi*∂V3).*ΔYi)')
            else
                dv1[j] += sum(vec((-2f0*Xi*∂V1').*ΔYi)')
                dv2[j] += sum(vec((-2f0*Xi*∂V2').*ΔYi)')
                dv3[j] += sum(vec((-2f0*Xi*∂V3').*ΔYi)')
            end

        end
    end
    return dv1, dv2, dv3
end

# Forward pass
function forward(X::AbstractArray{Float32, 4}, C::Conv1x1; logdet=nothing)
    isnothing(logdet) ? logdet = C.logdet : logdet = logdet
    nx, ny, n_in, batchsize = size(X)
    Y = cuzeros(X, nx, ny, n_in, batchsize)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data
    k = length(v1)
    for i=1:batchsize
        Xi = reshape(X[:,:,:,i], :, n_in)
        Yi = Xi*(I - 2f0*v1*v1'/(v1'*v1))*(I - 2f0*v2*v2'/(v2'*v2))*(I - 2f0*v3*v3'/(v3'*v3))
        Y[:,:,:,i] = reshape(Yi, nx, ny, n_in, 1)
    end

    logdet == true ? (return Y, 0f0) : (return Y)   # logdet always 0
end

# Forward pass
function forward(X::AbstractArray{Float32, 5}, C::Conv1x1; logdet=nothing)
    isnothing(logdet) ? logdet = C.logdet : logdet = logdet
    nx, ny, nz, n_in, batchsize = size(X)
    Y = cuzeros(X, nx, ny, nz, n_in, batchsize)
    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data
    k = length(v1)
    for i=1:batchsize
        Xi = reshape(X[:,:,:,:,i], :, n_in)
        Yi = Xi*(I - 2f0*v1*v1'/(v1'*v1))*(I - 2f0*v2*v2'/(v2'*v2))*(I - 2f0*v3*v3'/(v3'*v3))
        Y[:,:,:,:,i] = reshape(Yi, nx, ny, nz, n_in, 1)
    end
    logdet == true ? (return Y, 0f0) : (return Y)   # logdet always 0
end

# Forward pass and update weights
function forward(X_tuple::Tuple, C::Conv1x1)
    ΔX = X_tuple[1]
    X = X_tuple[2]
    ΔY = forward(ΔX, C; logdet=false)    # forward propagate residual
    Y = forward(X, C; logdet=false)  # recompute forward state
    Δv1, Δv2, Δv3 = conv1x1_grad_v(Y, ΔX, C; adjoint=true)  # gradient w.r.t. weights
    isnothing(C.v1.grad) ? (C.v1.grad = Δv1) : (C.v1.grad += Δv1)
    isnothing(C.v2.grad) ? (C.v2.grad = Δv2) : (C.v2.grad += Δv2)
    isnothing(C.v3.grad) ? (C.v3.grad = Δv3) : (C.v3.grad += Δv3)
    return ΔY, Y
end

# Inverse pass
function inverse(Y::AbstractArray{Float32, 4}, C::Conv1x1; logdet=nothing)
    isnothing(logdet) ? logdet = C.logdet : logdet = logdet
    nx, ny, n_in, batchsize = size(Y)
    X = cuzeros(Y, nx, ny, n_in, batchsize)
    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data
    k = length(v1)
    for i=1:batchsize
        Yi = reshape(Y[:,:,:,i], :, n_in)
        Xi = Yi*(I - 2f0*v3*v3'/(v3'*v3))'*(I - 2f0*v2*v2'/(v2'*v2))'*(I - 2f0*v1*v1'/(v1'*v1))'
        X[:,:,:,i] = reshape(Xi, nx, ny, n_in, 1)
    end
   logdet == true ? (return X, 0f0) : (return X)   # logdet always 0
end

# Inverse pass
function inverse(Y::AbstractArray{Float32, 5}, C::Conv1x1; logdet=nothing)
    isnothing(logdet) ? logdet = C.logdet : logdet = logdet
    nx, ny, nz, n_in, batchsize = size(Y)
    X = cuzeros(Y, nx, ny, nz, n_in, batchsize)
    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data
    k = length(v1)
    for i=1:batchsize
        Yi = reshape(Y[:,:,:,:,i], :, n_in)
        Xi = Yi*(I - 2f0*v3*v3'/(v3'*v3))'*(I - 2f0*v2*v2'/(v2'*v2))'*(I - 2f0*v1*v1'/(v1'*v1))'
        X[:,:,:,:,i] = reshape(Xi, nx, ny, nz, n_in, 1)
    end
    logdet == true ? (return X, 0f0) : (return X)   # logdet always 0
end

# Inverse pass and update weights
function inverse(Y_tuple::Tuple, C::Conv1x1)
    ΔY = Y_tuple[1]
    Y = Y_tuple[2]
    ΔX = inverse(ΔY, C; logdet=false)    # derivative w.r.t. input
    X = inverse(Y, C; logdet=false)  # recompute forward state
    Δv1, Δv2, Δv3 =  conv1x1_grad_v(X, ΔY, C)  # gradient w.r.t. weights
    isnothing(C.v1.grad) ? (C.v1.grad = Δv1) : (C.v1.grad += Δv1)
    isnothing(C.v2.grad) ? (C.v2.grad = Δv2) : (C.v2.grad += Δv2)
    isnothing(C.v3.grad) ? (C.v3.grad = Δv3) : (C.v3.grad += Δv3)
    return ΔX, X
end

# Clear gradients
function clear_grad!(C::Conv1x1)
    C.v1.grad = nothing
    C.v2.grad = nothing
    C.v3.grad = nothing
end

# Get parameters
get_params(C::Conv1x1) = [C.v1, C.v2, C.v3]
