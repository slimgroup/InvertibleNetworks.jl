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

# Input image
X = randn(Float32, nx, ny, n_channel, batchsize)

# Loop over all possible permutation options
options = ["none", "lower", "both", "full"]
for j=1:length(options)
    
    # HINT layer w/o logdet
    permute = options[j]
    HL = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute=permute)

    # Test 
    Y = HL.forward(X)
    X_ = HL.inverse(Y)
    @test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)

    Y = HL.forward(X)
    X_ = HL.backward(0f0.*Y, Y)[2]
    @test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)

    # Test with logdet
    HL = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute=permute, logdet=true)

    Y, logdet = HL.forward(X)
    X_ = HL.inverse(Y)
    @test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)

    Y, logdet = HL.forward(X)
    X_ = HL.backward(0f0.*Y, Y)[2]
    @test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)

    # Reverse layer w/o logdet
    HL = reverse(CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute=permute))

    # Test 
    Y = HL.forward(X)
    X_ = HL.inverse(Y)
    @test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)

    Y = HL.forward(X)
    X_ = HL.backward(0f0.*Y, Y)[2]
    @test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)

    # Test with logdet
    HL = reverse(CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute=permute, logdet=true))

    Y, logdet = HL.forward(X)
    X_ = HL.inverse(Y)
    @test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)

    Y, logdet = HL.forward(X)
    X_ = HL.backward(0f0.*Y, Y)[2]
    @test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)
end


#######################################################################################################################
# Gradient test

# Input image
X = randn(Float32, nx, ny, n_channel, batchsize)
X0 = randn(Float32, nx, ny, n_channel, batchsize)
dX = X - X0

function loss(HL, X)
    Y = HL.forward(X)
    f = .5*norm(Y)^2
    ΔY = copy(Y)
    ΔX, X_ = HL.backward(ΔY, Y)
    return f, ΔX, HL.CL[1].RB.W1.grad, X_
end

# Test for input X
HINT = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute="lower", logdet=false)
layers = [HINT, reverse(HINT)]
for j=1:length(layers)
    HL = layers[j]
    
    f0, gX, X_ = loss(HL, X0)[[1,2,4]]
    @test isapprox(norm(X_ - X0)/norm(X0), 0f0; atol=1f-6)

    maxiter = 5
    h = 0.1f0
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)

    print("\nGradient test ΔX\n")
    for j=1:maxiter
        f = loss(HL, X0 + h*dX)[1]
        err1[j] = abs(f - f0)
        err2[j] = abs(f - f0 - h*dot(dX, gX))
        print(err1[j], "; ", err2[j], "\n")
        h = h/2f0
    end

    @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
    @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)
end

# Test for weights
HINT = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute="lower")
HINT0 = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute="lower")
layers = [HINT, reverse(HINT)]
layers0 = [HINT0, reverse(HINT0)]

for j=1:length(layers)

    HL = layers[j]
    HL0 = layers0[j]

    X = glorot_uniform(nx, ny, n_channel, batchsize)
    HL.CL[1].RB.W1.data *= 4f0; HL0.CL[1].RB.W1.data *= 4f0 # make weights larger
    HLini = deepcopy(HL0)
    dW = HL.CL[1].RB.W1.data - HL0.CL[1].RB.W1.data

    f0, gX, gW, X_ = loss(HL0, X)
    @test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)

    maxiter = 5
    h = 0.5f0
    err3 = zeros(Float32, maxiter)
    err4 = zeros(Float32, maxiter)

    print("\nGradient test weights\n")
    for j=1:maxiter
        HL0.CL[1].RB.W1.data = HLini.CL[1].RB.W1.data + h*dW
        f = loss(HL0, X)[1]
        err3[j] = abs(f - f0)
        err4[j] = abs(f - f0 - h*dot(gW, dW))
        print(err3[j], "; ", err4[j], "\n")
        h = h/2f0
    end

    @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
end

# Test for weights and logdet
function loss_logdet(HL, X)
    Y, logdet = HL.forward(X)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔX, X_ = HL.backward(ΔY, Y)
    return f, ΔX, HL.CL[1].RB.W1.grad, X_
end

HINT = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute="lower", logdet=true)
HINT0 = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; permute="lower", logdet=true)
layers = [HINT, reverse(HINT)]
layers0 = [HINT0, reverse(HINT0)]

for j=1:length(layers)

    HL = layers[j]
    HL0 = layers0[j]

    X = glorot_uniform(nx, ny, n_channel, batchsize)
    HL.CL[1].RB.W1.data *= 8f0; HL0.CL[1].RB.W1.data *= 8f0 # make weights larger
    HLini = deepcopy(HL0)
    dW = HL.CL[1].RB.W1.data - HL0.CL[1].RB.W1.data

    f0, gX, gW, X_ = loss_logdet(HL0, X)
    @test isapprox(norm(X_ - X)/norm(X), 0f0; atol=1f-5)

    maxiter = 5
    h = 0.5f0
    err3 = zeros(Float32, maxiter)
    err4 = zeros(Float32, maxiter)

    print("\nGradient test weights\n")
    for j=1:maxiter
        HL0.CL[1].RB.W1.data = HLini.CL[1].RB.W1.data + h*dW
        f = loss_logdet(HL0, X)[1]
        err3[j] = abs(f - f0)
        err4[j] = abs(f - f0 - h*dot(gW, dW))
        print(err3[j], "; ", err4[j], "\n")
        h = h/2f0
    end

    @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
end