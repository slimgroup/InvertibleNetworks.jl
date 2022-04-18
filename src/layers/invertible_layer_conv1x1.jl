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

function partial_derivative_outer(v::AbstractArray{T, 1}) where T
    k = length(v)
    out1 = v * v'
    n = v' * v
    outer = cuzeros(v, k, k, k)
    for i=1:k
        copyto!(view(outer, i, :, :), out1)
    end
    broadcast!(*, outer, v, outer)
    broadcast!(*, outer, -2/n, outer)
    for j=1:k
        v1 = view(outer,j, :, j)
        broadcast!(+, v1, v1, v)
        v1 = view(outer,j, j, :)
        broadcast!(+, v1, v1, v)
    end
    broadcast!(*, outer, 1/n, outer)
    return outer
end

function partial_derivative_outer(v::CuArray{T, 1}) where T
    k = length(v)
    out1 = v * v'
    n = v' * v
    outer = cuzeros(v, k, k, k)
    for i=1:k
        copyto!(view(outer, i, :, :), out1)
    end
    broadcast!(*, outer, v, outer)
    broadcast!(*, outer, -2/n, outer)
    for j=1:k
        v1 = view(outer,j, :, j)
        broadcast!(+, v1, v1, v)
        v1 = view(outer,j, j, :)
        broadcast!(+, v1, v1, v)
    end
    broadcast!(*, outer, 1/n, outer)
    return outer
end


function mat_tens_i(out::AbstractVector{T}, Mat::AbstractArray{T, 2},
                    Tens::AbstractArray{T, 3}, Mat2::AbstractArray{T, 2}) where T
    # Computes sum( (Mat * tens) .* Mat2) for each element in the batch
    copyto!(out, map(i -> dot(Mat * Tens[i, :, :], Mat2) , 1:size(Tens, 1)))
    return out
end

function conv1x1_grad_v(X::AbstractArray{T, N}, ΔY::AbstractArray{T, N},
                        C::Conv1x1; adjoint=false) where {T, N}

    # Reshape input
    n_in, batchsize = size(X)[N-1:N]
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

    dV1 = partial_derivative_outer(v1)
    dV2 = partial_derivative_outer(v2)
    dV3 = partial_derivative_outer(v3)

    M1 = (I - 2 * (V2 + V3) + 4*V2*V3)
    M3 = (I - 2 * (V1 + V2) + 4*V1*V2)
    tmp = cuzeros(X, k, k)
    for i=1:k
        # dV1
        mul!(tmp, dV1[i, :, :], M1)
        @views adjoint ? copyto!(dV1[i, :, :], tmp') : copyto!(dV1[i, :, :], tmp)
        # dV2
        v2 = dV2[i, :, :]
        broadcast!(+, tmp, v2, 4 * V1 * v2 * V3 - 2 * (V1 * v2 + v2 * V3))
        @views adjoint ? copyto!(dV2[i, :, :], tmp') : copyto!(dV2[i, :, :], tmp)
        # dV3
        mul!(tmp, M3, dV3[i, :, :])
        @views adjoint ? copyto!(dV3[i, :, :], tmp') : copyto!(dV3[i, :, :], tmp)
    end

    prod_res = cuzeros(X, size(dV1, 1))
    for i=1:batchsize
        Xi = -2f0*reshape(selectdim(X, N, i), :, n_in)
        ΔYi = reshape(selectdim(ΔY, N, i), :, n_in)
        broadcast!(+, dv1, dv1, mat_tens_i(prod_res, Xi, dV1, ΔYi))
        broadcast!(+, dv2, dv2, mat_tens_i(prod_res, Xi, dV2, ΔYi))
        broadcast!(+, dv3, dv3, mat_tens_i(prod_res, Xi, dV3, ΔYi))
    end
    return dv1, dv2, dv3
end


# Forward pass
function forward(X::AbstractArray{T, N}, C::Conv1x1; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = C.logdet : logdet = logdet
    Y = cuzeros(X, size(X)...)
    n_in = size(X, N-1)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data

    for i=1:size(X, N)
        Xi = reshape(selectdim(X, N, i), :, n_in)
        Yi = chain_lr(Xi, v1, v2, v3)
        selectdim(Y, N, i) .= reshape(Yi, size(selectdim(Y, N, i))...)
    end
    logdet == true ? (return Y, 0) : (return Y)   # logdet always 0
end

# Forward pass and update weights
function forward(X_tuple::Tuple, C::Conv1x1; set_grad::Bool=true)
    ΔX = X_tuple[1]
    X = X_tuple[2]
    ΔY = forward(ΔX, C; logdet=false)    # forward propagate residual
    Y = forward(X, C; logdet=false)  # recompute forward state
    Δv1, Δv2, Δv3 = conv1x1_grad_v(Y, ΔX, C; adjoint=true)  # gradient w.r.t. weights
    if set_grad
        isnothing(C.v1.grad) ? (C.v1.grad = Δv1) : (C.v1.grad += Δv1)
        isnothing(C.v2.grad) ? (C.v2.grad = Δv2) : (C.v2.grad += Δv2)
        isnothing(C.v3.grad) ? (C.v3.grad = Δv3) : (C.v3.grad += Δv3)
    else
        Δθ = [Parameter(Δv1), Parameter(Δv2), Parameter(Δv3)]
    end
    set_grad ? (return ΔY, Y) : (return ΔY, Δθ, Y)
end

# Inverse pass
function inverse(Y::AbstractArray{T, N}, C::Conv1x1; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = C.logdet : logdet = logdet
    X = cuzeros(Y, size(Y)...)
    n_in = size(X, N-1)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data

    for i=1:size(Y, N)
        Yi = reshape(selectdim(Y, N, i), :, n_in)
        Xi = chain_lr(Yi, v3, v2, v1)
        selectdim(X, N, i) .= reshape(Xi, size(selectdim(X, N, i))...)
    end
    logdet == true ? (return X, 0) : (return X)   # logdet always 0
end

# Inverse pass and update weights
function inverse(Y_tuple::Tuple, C::Conv1x1; set_grad::Bool=true)
    ΔY = Y_tuple[1]
    Y = Y_tuple[2]
    ΔX = inverse(ΔY, C; logdet=false)    # derivative w.r.t. input
    X = inverse(Y, C; logdet=false)  # recompute forward state
    Δv1, Δv2, Δv3 =  conv1x1_grad_v(X, ΔY, C)  # gradient w.r.t. weights
    if set_grad
        isnothing(C.v1.grad) ? (C.v1.grad = Δv1) : (C.v1.grad += Δv1)
        isnothing(C.v2.grad) ? (C.v2.grad = Δv2) : (C.v2.grad += Δv2)
        isnothing(C.v3.grad) ? (C.v3.grad = Δv3) : (C.v3.grad += Δv3)
    else
        Δθ = [Parameter(Δv1), Parameter(Δv2), Parameter(Δv3)]
    end
    set_grad ? (return ΔX, X) : (return ΔX, Δθ, X)
end


## Jacobian-related functions

function jacobian(ΔX::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, X::AbstractArray{T, N}, C::Conv1x1) where {T, N}
    Y = cuzeros(X, size(X)...)
    ΔY = cuzeros(ΔX, size(ΔX)...)
    n_in = size(X, N-1)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data
    dv1 = Δθ[1].data
    dv2 = Δθ[2].data
    dv3 = Δθ[3].data

    for i=1:size(X, N)
        Xi = reshape(selectdim(X, N, i), :, n_in)
        Yi = chain_lr(Xi, v1, v2, v3)
        selectdim(Y, N, i) .= reshape(Yi, size(selectdim(Y, N, i) )...)

        ΔXi = reshape(selectdim(ΔX, N, i), :, n_in)
        ΔYi = chain_lr(Xi, v1, v2, v3)
        # this is a lot of outer products of 1D vecotrs, need to be cleaned up that's overkill computationnaly
        n1 = norm(v1); n2 = norm(v2); n3 = norm(v3);
        c1 = I - 2f0*v1*v1'/n1^2f0; c2 = I - 2f0*v2*v2'/n2^2f0; c3 = I - 2f0*v3*v3'/n3^2f0;
        ΔYi = chain_lr(ΔXi, v1, v2, v3)
        ΔYi += -2f0*Xi*((dv1*v1'+v1*dv1'-2f0*dot(v1,dv1)*v1*v1'/n1^2f0)/n1^2f0*c2*c3+
                       c1*(dv2*v2'+v2*dv2'-2f0*dot(v2,dv2)*v2*v2'/n2^2f0)/n2^2f0*c3+
                       c1*c2*(dv3*v3'+v3*dv3'-2f0*dot(v3,dv3)*v3*v3'/n3^2f0)/n3^2f0)
        selectdim(ΔY, N, i) .= reshape(ΔYi, size(selectdim(ΔY, N, i))...)
    end

    return ΔY, Y
end

function adjointJacobian(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, C::Conv1x1) where {T, N}
    return inverse((ΔY, Y), C; set_grad=false)
end

function jacobianInverse(ΔY::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, Y::AbstractArray{T, N}, C::Conv1x1) where {T, N}
    return inverse(C).jacobian(ΔY, Δθ[end:-1:1], Y)
end

function adjointJacobianInverse(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, C::Conv1x1) where {T, N}
    ΔX, Δθinv, X = inverse(C).adjointJacobian(ΔX, X)
    return ΔX, Δθinv[end:-1:1], X
end

function inverse(C::Conv1x1)
    return Conv1x1(C.k, C.v3, C.v2, C.v1, C.logdet)
end
