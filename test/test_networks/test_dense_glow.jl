# Generative model w/ Glow architecture from Kingma & Dhariwal (2018)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Random

# Random seed
Random.seed!(5);

# Define network
n_in = 1
n_hidden = 4
batchsize = 2
K = 2

# Test dense separately because RB is very different
dense = true

for (nx,L) in [(32,2),(2,1)] # nx=2 is difficult because can only do one multiscale L
    N = (nx)
    println("Testing Dense Glow with dimensions nx=$(N)")
    # Network and input
    G = NetworkGlow(n_in, n_hidden, L, K; dense=dense, nx=nx, ndims=length(N))
    X = rand(Float32, N..., n_in, batchsize)

    # Invertibility
    Y = G.forward(X)[1]
    X_ = G.inverse(Y)

    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

    ###################################################################################################
    # Test gradients are set and cleared
    G.backward(Y, Y)

    P = get_params(G)
    gsum = 0
    for p in P
        ~isnothing(p.grad) && (  gsum += 1)
    end

    param_factor = 11
    @test isequal(gsum, L*K*param_factor)

    clear_grad!(G)
    gsum = 0
    for p in P
        ~isnothing(p.grad) && (  gsum += 1)
    end
    @test isequal(gsum, 0)

    ###################################################################################################
    # Gradient test
    function loss_dense(L, X)
        Y, logdet = L.forward(X)
        f = -log_likelihood(Y) - logdet
        ΔY = -∇log_likelihood(Y)
        ΔX, X_ = L.backward(ΔY, Y)
        return f, ΔX, L.CL[1,1].RB.params[1].grad
    end

    # Gradient test w.r.t. input
    G = NetworkGlow(n_in, n_hidden, L, K; dense=dense, nx=nx,  ndims=length(N))
    X = rand(Float32, N..., n_in, batchsize)
    X0 = rand(Float32, N..., n_in, batchsize)
    dX = X - X0

    f0, ΔX = loss_dense(G, X0)[1:2]
    h = 0.1f0
    maxiter = 4
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)

    print("\nGradient test glow: input\n")
    for j=1:maxiter
        f = loss_dense(G, X0 + h*dX,)[1]
        err1[j] = abs(f - f0)
        err2[j] = abs(f - f0 - h*dot(dX, ΔX))
        print(err1[j], "; ", err2[j], "\n")
         h = h/2f0
    end

    @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


    # Gradient test w.r.t. parameters
    X = rand(Float32, N..., n_in, batchsize)
    G = NetworkGlow(n_in, n_hidden, L, K; dense=dense, nx=nx,   ndims=length(N))
    G0 = NetworkGlow(n_in, n_hidden, L, K; dense=dense, nx=nx,  ndims=length(N))
    Gini = deepcopy(G0)

    # Test one parameter from residual block and 1x1 conv
    dW = G.CL[1,1].RB.params - G0.CL[1,1].RB.params

    f0, ΔX, ΔW = loss_dense(G0, X)
    h = 0.1f0
    maxiter = 4
    err3 = zeros(Float32, maxiter)
    err4 = zeros(Float32, maxiter)

    print("\nGradient test glow: parameters\n")
    for j=1:maxiter
        set_params!(G0.CL[1,1].RB, Gini.CL[1,1].RB.params + h*dW)
        f = loss_dense(G0, X)[1]
        err3[j] = abs(f - f0)
        err4[j] = abs(f - f0 - h*dot(dW[1].data, ΔW))
        print(err3[j], "; ", err4[j], "\n")
        h = h/2f0
    end

    @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)
end