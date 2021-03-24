# Conditional HINT network from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Random
Random.seed!(11)

# Define network
nx = 16
ny = 16
n_in = 2
n_hidden = 4
batchsize = 2
L = 2
K = 2

function inv_test(nx, ny, n_in, batchsize, logdet, squeeze_type, split_scales)
    print("\nMultiscale Conditional HINT invertibility test with squeeze_type=$(squeeze_type), split_scales=$(split_scales), logdet=$(logdet)\n")
    CH = NetworkMultiScaleConditionalHINT(n_in, n_hidden, L, K; squeeze_type = squeeze_type, logdet=logdet, split_scales=split_scales)

    # Input image and data
    X = randn(Float32, nx, ny, n_in, batchsize)
    Y = randn(Float32, nx, ny, n_in, batchsize)


    # Test inverse
    Zx, Zy = CH.forward(X, Y)[1:2]
    X_, Y_ = CH.inverse(Zx, Zy)

    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-3)
    @test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-3)

    # Test backward
    ΔZx = randn(Float32, size(Zx))  # random derivative
    ΔZy = randn(Float32, size(Zx))
    ΔX_, ΔY_, X_, Y_ = CH.backward(ΔZx, ΔZy, Zx, Zy)

    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-3)
    @test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-3)

    # Test inverse Y only
    Zy = CH.forward_Y(Y)
    Y_ = CH.inverse_Y(Zy)

    @test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-3)
end   

# Loss
function loss(CH, X, Y)
    if CH.logdet 
        Zx, Zy, logdet = CH.forward(X, Y)
        f = -log_likelihood(tensor_cat(Zx, Zy)) - logdet
    else
        Zx, Zy = CH.forward(X, Y)
        f = -log_likelihood(tensor_cat(Zx, Zy)) 
    end 
    ΔZ       = -∇log_likelihood(tensor_cat(Zx, Zy))
    ΔZx, ΔZy = tensor_split(ΔZ)
    ΔX, ΔY   = CH.backward(ΔZx, ΔZy, Zx, Zy)[1:2]
    return f, ΔX, ΔY
end



function grad_test_X(nx, ny, n_channel, batchsize, logdet, squeeze_type, split_scales)
    #logdet ? lossf = loss_logdet : lossf = loss
    print("\nMultiscale Conditional HINT invertibility test with squeeze_type=$(squeeze_type), split_scales=$(split_scales), logdet=$(logdet)\n")
    CH = NetworkMultiScaleConditionalHINT(n_in, n_hidden, L, K; squeeze_type = squeeze_type, logdet=logdet, split_scales=split_scales)

    # Input image
    X0 = randn(Float32, nx, ny, n_channel, batchsize)
    dX = randn(Float32, nx, ny, n_channel, batchsize)

    # Input data
    Y0 = randn(Float32, nx, ny, n_channel, batchsize)
    dY = randn(Float32, nx, ny, n_channel, batchsize)

    f0, gX, gY = loss(CH, X0, Y0)[1:3]

    maxiter = 5
    h = 0.1f0
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)

    for j=1:maxiter
        f = loss(CH, X0 + h*dX, Y0 + h*dY)[1]
        err1[j] = abs(f - f0)
        err2[j] = abs(f - f0 - h*dot(dX, gX) - h*dot(dY, gY))
        print(err1[j], "; ", err2[j], "\n")
        h = h/2f0
    end

    @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
end

function grad_test_par(nx, ny, n_channel, batchsize, logdet, squeeze_type, split_scales)
    print("\nMultiscale Conditional HINT invertibility test with squeeze_type=$(squeeze_type), split_scales=$(split_scales), logdet=$(logdet)\n")
    #logdet ? lossf = loss_logdet : lossf = loss
    CH = NetworkMultiScaleConditionalHINT(n_in, n_hidden, L, K; squeeze_type = squeeze_type, logdet=logdet, split_scales=split_scales)


    CHini = deepcopy(CH0)

    # Perturbation
    X = randn(Float32, nx, ny, n_channel, batchsize)
    Y = randn(Float32, nx, ny, n_channel, batchsize)
    dW = randn(Float32, size(CH0.CL_X.CL[1].RB.W1))

    f0, gW, gv = loss(CH0, X, Y)[[1,4,5]]

    maxiter = 5
    h = 0.1f0

    err3 = zeros(Float32, maxiter)
    err4 = zeros(Float32, maxiter)

    for j=1:maxiter
        CH0.CL_X.CL[1].RB.W1.data = CHini.CL_X.CL[1].RB.W1.data + h*dW
        f = loss(CH0, X, Y)[1]
        err3[j] = abs(f - f0)
        err4[j] = abs(f - f0 - h*dot(gW, dW))
        print(err3[j], "; ", err4[j], "\n")
        h = h/2f0
    end

    @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
end

for squeeze_type in ["normal", "wavelet", "Haar"]
    for split_scales in [true, false]
        for logdet in [true, false]
            inv_test(nx, ny, n_in, batchsize, logdet, squeeze_type, split_scales)
            grad_test_X(nx, ny, n_in, batchsize, logdet, squeeze_type, split_scales)
            #grad_test_par(nx, ny, n_in, batchsize, logdet, squeeze_type, split_scales)    
        end
    end
end

###################################################################################################
# Jacobian-related tests: 
### NEED THESE
