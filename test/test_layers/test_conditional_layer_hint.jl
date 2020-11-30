# Tests for conditional HINT layer (Kruse et al, 2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Random

# Random seed
Random.seed!(11)

#######################################################################################################################
# Test invertibility

# Input
nx = 16
ny = 16
n_channel = 8
n_hidden = 8
batchsize = 2

function inv_test(nx, ny, n_channel, batchsize, rev, logdet, permute)
    print("\nConditional HINT invertibility test with permute=$(permute), logdet=$(logdet), reverse=$(rev)\n")
    CH = ConditionalLayerHINT(nx, ny, n_channel, n_hidden, batchsize; logdet=logdet, permute=permute)
    rev && (CH = reverse(CH))

    # Input image and data
    X = randn(Float32, nx, ny, n_channel, batchsize)
    Y = randn(Float32, nx, ny, n_channel, batchsize)

    # Test inverse
    Zx, Zy = CH.forward(X, Y)[1:2]
    X_, Y_ = CH.inverse(Zx, Zy)

    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)
    @test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-5)

    # Test backward
    ΔZx = randn(Float32, size(Zx))  # random derivative
    ΔZy = randn(Float32, size(Zx))
    ΔX_, ΔY_, X_, Y_ = CH.backward(ΔZx, ΔZy, Zx, Zy)

    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)
    @test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-5)

    # Test inverse Y only
    Zy = CH.forward_Y(Y)
    Y_ = CH.inverse_Y(Zy)

    @test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-5)
end

function loss(CH, X, Y)
    Zx, Zy = CH.forward(X, Y)
    Z = tensor_cat(Zx, Zy)
    f = .5*norm(Z)^2
    ΔZx, ΔZy = tensor_split(Z)
    ΔX, ΔY = CH.backward(ΔZx, ΔZy, Zx, Zy)[1:2]
    isnothing(CH.C_X) ? v1grad = nothing : v1grad=CH.C_X.v1.grad
    return f, ΔX, ΔY, CH.CL_X.CL[1].RB.W1.grad, v1grad
end

function loss_logdet(CH, X, Y)
    Zx, Zy, logdet = CH.forward(X, Y)
    Z = tensor_cat(Zx, Zy)
    f = -log_likelihood(Z) - logdet
    ΔZ = -∇log_likelihood(Z)
    ΔZx, ΔZy = tensor_split(ΔZ)
    ΔX, ΔY = CH.backward(ΔZx, ΔZy, Zx, Zy)[1:2]
    isnothing(CH.C_X) ? v1grad = nothing : v1grad=CH.C_X.v1.grad
    return f, ΔX, ΔY, CH.CL_X.CL[1].RB.W1.grad, v1grad
end

function grad_test_X(nx, ny, n_channel, batchsize, rev, logdet, permute)
    print("\nConditional HINT gradient test for input with permute=$(permute), logdet=$(logdet), reverse=$(rev)\n")
    logdet ? lossf = loss_logdet : lossf = loss
    CH = ConditionalLayerHINT(nx, ny, n_channel, n_hidden, batchsize; logdet=logdet, permute=permute)
    rev && (CH = reverse(CH))

    # Input image
    X0 = randn(Float32, nx, ny, n_channel, batchsize)
    dX = randn(Float32, nx, ny, n_channel, batchsize)

    # Input data
    Y0 = randn(Float32, nx, ny, n_channel, batchsize)
    dY = randn(Float32, nx, ny, n_channel, batchsize)

    f0, gX, gY = lossf(CH, X0, Y0)[1:3]

    maxiter = 5
    h = 0.1f0
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)

    for j=1:maxiter
        f = lossf(CH, X0 + h*dX, Y0 + h*dY)[1]
        err1[j] = abs(f - f0)
        err2[j] = abs(f - f0 - h*dot(dX, gX) - h*dot(dY, gY))
        print(err1[j], "; ", err2[j], "\n")
        h = h/2f0
    end

    @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
end

function grad_test_par(nx, ny, n_channel, batchsize, rev, logdet, permute)
    print("\nConditional HINT gradient test for weights with permute=$(permute), logdet=$(logdet), reverse=$(rev)\n")
    logdet ? lossf = loss_logdet : lossf = loss
    CH0 = ConditionalLayerHINT(nx, ny, n_channel, n_hidden, batchsize; logdet=logdet, permute=permute)
    rev && (CH0 = reverse(CH0))
    CHini = deepcopy(CH0)

    # Perturbation
    X = randn(Float32, nx, ny, n_channel, batchsize)
    Y = randn(Float32, nx, ny, n_channel, batchsize)
    dW = randn(Float32, size(CH0.CL_X.CL[1].RB.W1))

    f0, gW, gv = lossf(CH0, X, Y)[[1,4,5]]

    maxiter = 5
    h = 0.1f0

    err3 = zeros(Float32, maxiter)
    err4 = zeros(Float32, maxiter)

    for j=1:maxiter
        CH0.CL_X.CL[1].RB.W1.data = CHini.CL_X.CL[1].RB.W1.data + h*dW
        f = lossf(CH0, X, Y)[1]
        err3[j] = abs(f - f0)
        err4[j] = abs(f - f0 - h*dot(gW, dW))
        print(err3[j], "; ", err4[j], "\n")
        h = h/2f0
    end

    @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
end


for logdet in [true, false]
    for permute in [true, false]
        for rev in [true, false]
            inv_test(nx, ny, n_channel, batchsize, rev, logdet, permute)
            grad_test_X(nx, ny, n_channel, batchsize, rev, logdet, permute)
            grad_test_par(nx, ny, n_channel, batchsize, rev, logdet, permute)
        end
    end
end


###################################################################################################
# Jacobian-related tests

function jacobian_test_par(nx, ny, n_channel, batchsize, logdet, permute)
    print("\nConditional HINT jacobian test with permute=$(permute), logdet=$(logdet)\n")

    # Initialization
    CH = ConditionalLayerHINT(nx, ny, n_channel, n_hidden, batchsize; logdet=logdet, permute=permute)
    θ = deepcopy(get_params(CH))
    CH0 = ConditionalLayerHINT(nx, ny, n_channel, n_hidden, batchsize; logdet=logdet, permute=permute)
    θ0 = deepcopy(get_params(CH0))
    X = randn(Float32, nx, ny, n_channel, batchsize)
    Y = randn(Float32, nx, ny, n_channel, batchsize)

    # Perturbation
    dX = randn(Float32, nx, ny, n_channel, batchsize)
    dY = randn(Float32, nx, ny, n_channel, batchsize)
    dθ = θ-θ0

    # Jacobian eval
    dZx, dZy, Zx, Zy = CH.jacobian(dX, dY, dθ, X, Y)[1:4]

    # Jacobian Test
    h = 0.1f0
    maxiter = 5
    err5 = zeros(Float32, maxiter)
    err6 = zeros(Float32, maxiter)
    for j=1:maxiter
        set_params!(CH, θ+h*dθ)
        Zx_, Zy_ = CH.forward(X+h*dX, Y+h*dY)[1:2]
        err5[j] = sqrt(norm(Zx_ - Zx)^2+norm(Zy_ - Zy)^2)
        err6[j] = sqrt(norm(Zx_ - Zx - h*dZx)^2+norm(Zy_ - Zy - h*dZy)^2)
        print(err5[j], "; ", err6[j], "\n")
        h = h/2f0
    end
    @test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f1)

    # Adjoint test
    set_params!(CH, θ)
    dZx, dZy, Zx, Zy = CH.jacobian(dX, dY, dθ, X, Y)[1:4]
    dZx_ = randn(Float32, size(dZx))
    dZy_ = randn(Float32, size(dZy))
    dX_, dY_, dθ_ = CH.adjointJacobian(dZx_, dZy_, Zx, Zy)[1:3]
    a = dot(dZx, dZx_)+dot(dZy, dZy_)
    b = dot(dX, dX_)+dot(dY, dY_)+dot(dθ, dθ_)
    @test isapprox(a, b; rtol=1f-3)
end

for logdet in [true, false]
    for permute in [true, false]
        jacobian_test_par(nx, ny, n_channel, batchsize, logdet, permute)
    end
end