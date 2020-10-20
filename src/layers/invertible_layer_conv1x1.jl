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

function partial_derivative_outer(v::Array{Float32, 1})
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

function partial_derivative_outer(v::CuArray{Float32, 1})
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

function mat_tens_i(out::AbstractArray{Float32, 3}, Mat::AbstractArray{Float32, 2},
                    Tens::AbstractArray{Float32, 3}, Mat2::AbstractArray{Float32, 2})
    tmp = cuzeros(out, size(out, 2), size(out, 3))
    for i=1:size(out, 1)
        mul!(tmp, Mat, Tens[i, :, :])
        broadcast!(*, tmp, tmp, Mat2)
        @views copyto!(out[i, :, :], tmp)
    end
    return out
end

function custom_sum(a::AbstractArray{Float32, 3}, dims::Tuple{Integer, Integer})
    summed = sum(a, dims=dims)
    return dropdims(summed, dims = (findall(size(summed) .== 1)...,))
end

function conv1x1_grad_v(X::AbstractArray{Float32, N}, ΔY::AbstractArray{Float32, N},
                        C::Conv1x1; adjoint=false) where {N}

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

    ∂V1 = deepcopy(dV1)
    ∂V2 = deepcopy(dV2)
    ∂V3 = deepcopy(dV3)

    M1 = (I - 2f0 * (V2 + V3) + 4f0*V2*V3)
    M3 = (I - 2f0 * (V1 + V2) + 4f0*V1*V2)
    tmp = cuzeros(X, k, k)
    for i=1:k
        # ∂V1
        mul!(tmp, ∂V1[i, :, :], M1)
        @views adjoint ? adjoint!(∂V1[i, :, :], tmp) : copyto!(∂V1[i, :, :], tmp)
        # ∂V2
        v2 = ∂V2[i, :, :]
        broadcast!(+, tmp, v2, 4f0 * V1 * v2 * V3 - 2f0 * (V1 * v2 + v2 * V3))
        @views adjoint ? adjoint!(∂V2[i, :, :], tmp) : copyto!(∂V2[i, :, :], tmp)
        # ∂V3
        mul!(tmp, M3, ∂V3[i, :, :])
        @views adjoint ? adjoint!(∂V3[i, :, :], tmp) : copyto!(∂V3[i, :, :], tmp)
    end

    prod_res = cuzeros(X, size(∂V1, 1), prod(size(X)[1:N-2]), n_in)
    inds = [i<N ? (:) : 1 for i=1:N]
    for i=1:batchsize
        inds[end] = i
        Xi = -2f0*reshape(view(X, inds...), :, n_in)
        ΔYi = reshape(view(ΔY, inds...), :, n_in)
        broadcast!(+, dv1, dv1, custom_sum(mat_tens_i(prod_res, Xi, ∂V1, ΔYi), (3, 2)))
        broadcast!(+, dv2, dv2, custom_sum(mat_tens_i(prod_res, Xi, ∂V2, ΔYi), (3, 2)))
        broadcast!(+, dv3, dv3, custom_sum(mat_tens_i(prod_res, Xi, ∂V3, ΔYi), (3, 2)))
    end
    return dv1, dv2, dv3
end

# Forward pass
function forward(X::AbstractArray{Float32, N}, C::Conv1x1; logdet=nothing) where {N}
    isnothing(logdet) ? logdet = C.logdet : logdet = logdet
    Y = cuzeros(X, size(X)...)
    n_in = size(X, N-1)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data

    inds = [i<N ? (:) : 1 for i=1:N]
    for i=1:size(X, N)
        inds[end] = i
        Xi = reshape(view(X, inds...), :, n_in)
        Yi = Xi*(I - 2f0*v1*v1'/(v1'*v1))*(I - 2f0*v2*v2'/(v2'*v2))*(I - 2f0*v3*v3'/(v3'*v3))
        view(Y, inds...) .= reshape(Yi, size(view(Y, inds...))...)
    end
    logdet == true ? (return Y, 0f0) : (return Y)   # logdet always 0
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
function inverse(Y::AbstractArray{Float32, N}, C::Conv1x1; logdet=nothing) where {N}
    isnothing(logdet) ? logdet = C.logdet : logdet = logdet
    X = cuzeros(Y, size(Y)...)
    n_in = size(X, N-1)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data

    inds = [i<N ? (:) : 1 for i=1:N]
    for i=1:size(Y, N)
        inds[end] = i
        Yi = reshape(view(Y, inds...), :, n_in)
        Xi = Yi*(I - 2f0*v3*v3'/(v3'*v3))'*(I - 2f0*v2*v2'/(v2'*v2))'*(I - 2f0*v1*v1'/(v1'*v1))'
        view(X, inds...) .= reshape(Xi, size(view(X, inds...))...)
    end
   logdet == true ? (return X, 0f0) : (return X)   # logdet always 0
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

function jacobian(ΔX::AbstractArray{Float32, N}, Δθ::Array{Parameter, 1}, X::AbstractArray{Float32, N}, C::Conv1x1) where N
    Y = cuzeros(X, size(X)...)
    ΔY = cuzeros(ΔX, size(ΔX)...)
    n_in = size(X, N-1)

    v1 = C.v1.data
    v2 = C.v2.data
    v3 = C.v3.data
    dv1 = Δθ[1].data
    dv2 = Δθ[2].data
    dv3 = Δθ[3].data

    inds = [i<N ? (:) : 1 for i=1:N]
    for i=1:size(X, N)
        inds[end] = i
        Xi = reshape(view(X, inds...), :, n_in)
        n1 = norm(v1); n2 = norm(v2); n3 = norm(v3);
        c1 = I - 2f0*v1*v1'/n1^2f0; c2 = I - 2f0*v2*v2'/n2^2f0; c3 = I - 2f0*v3*v3'/n3^2f0;
        Yi = Xi*c1*c2*c3
        view(Y, inds...) .= reshape(Yi, size(view(Y, inds...))...)

        ΔXi = reshape(view(ΔX, inds...), :, n_in)
        ΔYi = ΔXi*c1*c2*c3+
              -2f0*Xi*((dv1*v1'+v1*dv1'-2f0*dot(v1,dv1)*v1*v1'/n1^2f0)/n1^2f0*c2*c3+
                       c1*(dv2*v2'+v2*dv2'-2f0*dot(v2,dv2)*v2*v2'/n2^2f0)/n2^2f0*c3+
                       c1*c2*(dv3*v3'+v3*dv3'-2f0*dot(v3,dv3)*v3*v3'/n3^2f0)/n3^2f0)
        view(ΔY, inds...) .= reshape(ΔYi, size(view(ΔY, inds...))...)
    end

    return ΔY, Y
end

function adjointJacobian(ΔY::AbstractArray{Float32, N}, Y::AbstractArray{Float32, N}, C::Conv1x1) where N
    return inverse((ΔY, Y), C; set_grad=false)
end

function jacobianInverse(ΔY::AbstractArray{Float32, N}, Δθ::Array{Parameter, 1}, Y::AbstractArray{Float32, N}, C::Conv1x1) where N
    return inverse(C).jacobian(ΔY, Δθ[end:-1:1], Y)
end

function adjointJacobianInverse(ΔX::AbstractArray{Float32, N}, X::AbstractArray{Float32, N}, C::Conv1x1) where N
    ΔX, Δθinv, X = inverse(C).adjointJacobian(ΔX, X)
    return ΔX, Δθinv[end:-1:1], X
end

function inverse(C::Conv1x1)
    return Conv1x1(C.k, C.v3, C.v2, C.v1, C.logdet)
end


## Other utils

# Clear gradients
function clear_grad!(C::Conv1x1)
    C.v1.grad = nothing
    C.v2.grad = nothing
    C.v3.grad = nothing
end

# Get parameters
get_params(C::Conv1x1) = [C.v1, C.v2, C.v3]
