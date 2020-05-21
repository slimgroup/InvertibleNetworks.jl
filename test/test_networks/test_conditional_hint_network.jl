# Conditional HINT network from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Random
Random.seed!(11)

# Define network
nx = 64
ny = 64
n_in = 2
n_hidden = 4
batchsize = 2
L = 2
K = 2
# multiscale = true
multiscale = false

if multiscale
    CH0 = NetworkMultiScaleConditionalHINT(nx, ny, n_in, batchsize, n_hidden, L, K; split_scales=false, k1=3, k2=1, p1=1, p2=0)
else
    CH0 = NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, L*K; k1=3, k2=1, p1=1, p2=0)
end
nets = [CH0, reverse(CH0)]

# Loop over networks and reversed counterpart
for j = 1:length(nets)

    CH = nets[j]

    ###################################################################################################
    # Invertibility

    # Test layers
    test_size = 10
    X = randn(Float32, nx, ny, n_in, test_size)
    Y = X + .1f0*randn(Float32, nx, ny, n_in, test_size)

    # Forward-backward
    Zx, Zy, logdet = CH.forward(X, Y)
    X_, Y_ = CH.backward(0f0.*Zx, 0f0.*Zy, Zx, Zy)[3:4]
    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-3)
    @test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-3)

    # Forward-inverse
    Zx, Zy, logdet = CH.forward(X, Y)
    X_, Y_ = CH.inverse(Zx, Zy)
    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-3)
    @test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-3)

    # Y-lane only
    if ~CH.is_reversed
        Zyy = CH.forward_Y(Y)
        Yy = CH.inverse_Y(Zyy)
        @test isapprox(norm(Y - Yy)/norm(Y), 0f0; atol=1f-3)
    end

end

###################################################################################################
# Gradient test

# Loss
function loss(CH, X, Y)
    Zx, Zy, logdet = CH.forward(X, Y)
    f = -log_likelihood(tensor_cat(Zx, Zy)) - logdet
    ΔZ = -∇log_likelihood(tensor_cat(Zx, Zy))
    ΔZx, ΔZy = tensor_split(ΔZ)
    ΔX, ΔY = CH.backward(ΔZx, ΔZy, Zx, Zy)[1:2]
    return f, ΔX, ΔY
end

# Gradient test w.r.t. input
if multiscale
    CH0 = NetworkMultiScaleConditionalHINT(nx, ny, n_in, batchsize, n_hidden, L, K; split_scales=false, k1=3, k2=1, p1=1, p2=0)
else
    CH0 = NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, L*K; k1=3, k2=1, p1=1, p2=0)
end
nets = [CH0, reverse(CH0)]

# Loop over networks and reversed counterpart
for j = 1:length(nets)

    CH = nets[j]

    test_size = 10
    X = randn(Float32, nx, ny, n_in, test_size)
    Y = X + .1f0*randn(Float32, nx, ny, n_in, test_size)
    X0 = randn(Float32, nx, ny, n_in, test_size)
    Y0 = X0 + .1f0*randn(Float32, nx, ny, n_in, test_size)
    dX = X - X0
    dY = Y - Y0

    f0, ΔX, ΔY = loss(CH, X0, Y0)
    h = 0.1f0
    maxiter = 6
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)

    print("\nGradient test cond. HINT net: input\n")
    for j=1:maxiter
        f = loss(CH, X0 + h*dX, Y0 + h*dY)[1]
        err1[j] = abs(f - f0)
        err2[j] = abs(f - f0 - h*dot(dX, ΔX) - h*dot(dY, ΔY))
        print(err1[j], "; ", err2[j], "\n")
        h = h/2f0
    end

    @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)

end
