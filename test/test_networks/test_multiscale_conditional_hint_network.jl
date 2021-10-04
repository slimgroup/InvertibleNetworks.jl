# Test Multiscale Conditional HINT network from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Random
Random.seed!(14)

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
    CH = NetworkMultiScaleConditionalHINT(n_in, n_hidden, L, K; squeezer = squeeze_type, logdet=logdet, split_scales=split_scales)

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
    print("\nMultiscale Conditional HINT invertibility test with squeeze_type=$(squeeze_type), split_scales=$(split_scales), logdet=$(logdet)\n")
    CH = NetworkMultiScaleConditionalHINT(n_in, n_hidden, L, K; squeezer = squeeze_type, logdet=logdet, split_scales=split_scales)


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

shuffle_sq = ShuffleLayer()
wavelet_sq = WaveletLayer()
Haar_sq    = HaarLayer()

for squeeze_i in [shuffle_sq, wavelet_sq, Haar_sq]
    for split_scales in [true, false]
        for logdet in [true, false]
            inv_test(nx, ny, n_in, batchsize, logdet, squeeze_i, split_scales)
            grad_test_X(nx, ny, n_in, batchsize, logdet, squeeze_i, split_scales)
        end
    end
end

###################################################################################################
# Jacobian-related tests: NetworkMultiScaleConditionalHINT

# Gradient test

# Initialization
CH = NetworkMultiScaleConditionalHINT(n_in, n_hidden, L, K; split_scales=false, k1=3, k2=1, p1=1, p2=0, squeezer = shuffle_sq); 
CH.forward(randn(Float32, nx, ny, n_in, batchsize), randn(Float32, nx, ny, n_in, batchsize))
θ = deepcopy(get_params(CH))

CH0 = NetworkMultiScaleConditionalHINT(n_in, n_hidden, L, K; split_scales=false, k1=3, k2=1, p1=1, p2=0, squeezer = shuffle_sq); 

CH0.forward(randn(Float32, nx, ny, n_in, batchsize), randn(Float32, nx, ny, n_in, batchsize))
θ0 = deepcopy(get_params(CH0))

X = randn(Float32, nx, ny, n_in, batchsize)
Y = randn(Float32, nx, ny, n_in, batchsize)

# Perturbation (normalized)
dθ = θ-θ0; dθ .*= (norm.(θ).+1f-6)./(norm.(dθ).+1f-6)
dX = randn(Float32, nx, ny, n_in, batchsize); dX *= norm(X)/norm(dX)
dY = randn(Float32, nx, ny, n_in, batchsize); dY *= norm(Y)/norm(dY)

# Jacobian eval
dZx, dZy, Zx, Zy, _, _ = CH.jacobian(dX, dY, dθ, X, Y)

# Test
print("\nJacobian test\n")
h = 0.1f0
maxiter = 5
err5 = zeros(Float32, maxiter)
err6 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(CH, θ+h*dθ)
    Zx_, Zy_, _ = CH.forward(X+h*dX, Y+h*dY)
    err5[j] = sqrt(norm(Zx_ - Zx)^2+norm(Zy_ - Zy)^2)
    err6[j] = sqrt(norm(Zx_ - Zx - h*dZx)^2+norm(Zy_ - Zy - h*dZy)^2)
    print(err5[j], "; ", err6[j], "\n")
    global h = h/2f0
end

@test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Adjoint test

set_params!(CH, θ)
dZx, dZy, Zx, Zy, _, _ = CH.jacobian(dX, dY, dθ, X, Y)
dZx_ = randn(Float32, size(dZx)); dZy_ = randn(Float32, size(dZy))
dX_, dY_, dθ_, _, _, _ = CH.adjointJacobian(dZx_, dZy_, Zx, Zy)
a = dot(dZx, dZx_)+dot(dZy, dZy_)
b = dot(dX, dX_)+dot(dY, dY_)+dot(dθ, dθ_)
@test isapprox(a, b; rtol=1f-1)  ####### need to check low accuracy here
