# Tests for HINT layer (Kruse et al, 2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Random

Random.seed!(11)

#######################################################################################################################
# Test invertibility
# Input
nx = 16
ny = 16
n_channel = 16
n_hidden = 64
batchsize = 2

function test_inv(nx, ny, n_channel, n_hidden, batchsize, permute, logdet, rev)
    # Input image
    X = randn(Float32, nx, ny, n_channel, batchsize)

    # HINT layer w/o logdet
    HL = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute=permute, logdet=logdet)
    rev && (HL = reverse(HL))
    # Test 
    Y = logdet ? HL.forward(X)[1] : HL.forward(X)
    X_ = HL.inverse(Y)
    @test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)

    X_ = HL.backward(0f0.*Y, Y)[2]
    @test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)
end

function loss(HL, X)
    Y = HL.forward(X)
    f = .5*norm(Y)^2
    ΔY = copy(Y)
    ΔX, X_ = HL.backward(ΔY, Y)
    return f, ΔX, HL.CL[1].RB.W1.grad, X_
end

function loss_logdet(HL, X)
    Y, logdet = HL.forward(X)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔX, X_ = HL.backward(ΔY, Y)
    return f, ΔX, HL.CL[1].RB.W1.grad, X_
end

function grad_test_X(nx, ny, n_channel, n_hidden, batchsize, permute, logdet, rev)
    logdet ? lossf = loss_logdet : lossf = loss
    # Input image
    X0 = randn(Float32, nx, ny, n_channel, batchsize)
    dX = randn(Float32, nx, ny, n_channel, batchsize)
    # Test for input X
    HL = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute=permute, logdet=logdet)
    rev && (HL = reverse(HL))
        
    f0, gX, X_ = lossf(HL, X0)[[1,2,4]]
    @test isapprox(norm(X_ - X0)/norm(X0), 0f0; atol=1f-5)

    maxiter = 5
    h = 0.1f0
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)

    print("\nGradient test ΔX for permute=$(permute), reverse=$(rev), logdet=$(logdet)\n")
    for j=1:maxiter
        f = lossf(HL, X0 + h*dX)[1]
        err1[j] = abs(f - f0)
        err2[j] = abs(f - f0 - h*dot(dX, gX))
        print(err1[j], "; ", err2[j], "\n")
        h = h/2f0
    end

    @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
    @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)
end

function grad_test_layer(nx, ny, n_channel, n_hidden, batchsize, permute, logdet, rev)
    logdet ? lossf = loss_logdet : lossf = loss
    # Test for weights
    HL = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute=permute, logdet=logdet)
    HL0 = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute=permute, logdet=logdet)
    rev && (HL = reverse(HL))
    rev && (HL0 = reverse(HL0))

    X = glorot_uniform(nx, ny, n_channel, batchsize)
    HL.CL[1].RB.W1.data *= 4f0
    HL0.CL[1].RB.W1.data *= 4f0 # make weights larger
    HLini = deepcopy(HL0)
    dW = HL.CL[1].RB.W1.data - HL0.CL[1].RB.W1.data

    f0, gX, gW, X_ = lossf(HL0, X)
    @test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)

    maxiter = 5
    h = 0.5f0
    err3 = zeros(Float32, maxiter)
    err4 = zeros(Float32, maxiter)

    print("\nGradient test weights for permute=$(permute), reverse=$(rev), logdet=$(logdet)\n")
    for j=1:maxiter
        HL0.CL[1].RB.W1.data = HLini.CL[1].RB.W1.data + h*dW
        f = lossf(HL0, X)[1]
        err3[j] = abs(f - f0)
        err4[j] = abs(f - f0 - h*dot(gW, dW))
        print(err3[j], "; ", err4[j], "\n")
        h = h/2f0
    end

    @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
end

# Loop over all possible permutation options
options = ["none", "lower", "both", "full"]

for permute in options
    for logdet in [true, false]
        for rev in [true, false]
            test_inv(nx, ny, n_channel, n_hidden, batchsize, permute, logdet, rev)
            grad_test_layer(nx, ny, n_channel, n_hidden, batchsize, permute, logdet, rev)
            grad_test_X(nx, ny, n_channel, n_hidden, batchsize, permute, logdet, rev)
        end
    end
end