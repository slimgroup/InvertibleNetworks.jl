# Generative model w/ Glow architecture from Kingma & Dhariwal (2018)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Random
using Statistics 


# Random seed
Random.seed!(36);

function loss(G, X, Cond;summarized=false)
    Y, ZC, logdet = G.forward(X, Cond)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    if summarized
        ΔX = G.backward(ΔY, Y, ZC; Y_save=Cond)[1]
        return f, ΔX, G.cond_net.CL[1,1].RB.W1.grad
    else 
        ΔX =  G.backward(ΔY, Y, ZC)[1]
        return f, ΔX, G.CL[1,1].RB.W1.grad
    end
end

function gradients_set(G, n_in,n_cond,N; summarized=false)
    X = rand(Float32, N..., n_in, batchsize)
    Cond = rand(Float32, N..., n_cond, batchsize)

    XZ, CondZ = G.forward(X,Cond)

    # Set gradients 
    summarized ? G.backward(XZ, XZ, CondZ; Y_save=Cond) : G.backward(XZ, XZ, CondZ)

    P = get_params(G)
    gsum = 0
    for p in P
        ~isnothing(p.grad) && (gsum += 1)
    end
    summarized ?  (@test isequal(gsum, L*K*10+12)) : (@test isequal(gsum, L*K*10))
   
    clear_grad!(G)
    gsum = 0
    for p in P
        ~isnothing(p.grad) && (gsum += 1)
    end
    @test isequal(gsum, 0)
end

# Define network
nx = 16
ny = 16
nz = 16
n_in = 4
n_cond = 2
n_hidden = 4
batchsize = 4
L = 2
K = 2

stol = 1.5f0
for split_scales in [false,true]
    for N in [(16*nx),(nx,ny),(nx,ny,nz)]
        println("Test with split_scales = $(split_scales) N = $(N)")
        
        # Network and inputs
        G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales, ndims=length(N))

        X = randn(Float32, N..., n_in, batchsize)
        Cond = rand(Float32, N..., n_cond, batchsize)

        # Invertibility
        XZ, CondZ = G.forward(X,Cond)
        X_ = G.inverse(XZ, CondZ) # saving the cond output is important in split scales because of reshapes
        @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

        ###################################################################################################
        # Test gradients are set and cleared
        gradients_set(G, n_in, n_cond,N;)

        ###################################################################################################
        # Gradient test w.r.t. input
        X0 = randn(Float32, N..., n_in, batchsize)
        Cond0 = randn(Float32, N..., n_cond, batchsize)

        dX = X - X0

        f0, ΔX = loss(G, X0, Cond0)[1:2]
        h = 0.1f0
        maxiter = 4
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        print("\nGradient test glow: input\n")
        for j=1:maxiter
            f = loss(G, X0 + h*dX, Cond0)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dX, ΔX))
            print(err1[j], "; ", err2[j], "\n")
            h = h/2f0
        end

        rate1 = err1[1:end-1]./err1[2:end]
        rate2 = err2[1:end-1]./err2[2:end]

        @test isapprox(mean(rate1), 2f0; atol=stol)
        @test isapprox(mean(rate2), 4f0; atol=stol)

        # Gradient test w.r.t. parameters
        G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales, ndims=length(N))
        G0 = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales, ndims=length(N))
        Gini = deepcopy(G0)

        # Test one parameter from residual block
        dW = G.CL[1,1].RB.W1.data - G0.CL[1,1].RB.W1.data

        f0, ΔX, ΔW = loss(G0, X, Cond)
        h = 0.1f0
        maxiter = 4
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        print("\nGradient test glow: parameter\n")
        for j=1:maxiter
            G0.CL[1,1].RB.W1.data = Gini.CL[1,1].RB.W1.data + h*dW

            f = loss(G0, X, Cond)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dW, ΔW))
            print(err1[j], "; ", err2[j], "\n")
            h = h/2f0
        end

        rate1 = err1[1:end-1]./err1[2:end]
        rate2 = err2[1:end-1]./err2[2:end]

        @test isapprox(mean(rate1),2f0; atol=stol)
        @test isapprox(mean(rate2), 4f0; atol=stol)
    end
end

# with summary network
for split_scales in [false,true]
    for N in [(16*nx),(nx,ny),(nx,ny,nz)]
        println("Test with split_scales = $(split_scales) N = $(N) and summarized=$(true)")
        
        # Network and inputs
        G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales, ndims=length(N))
        sum_net = ResNet(n_cond, 16, 3; norm=nothing,ndims=length(N)) # make sure it doesnt have any weird normalizations
        G = SummarizedNet(G, sum_net)

        X = randn(Float32, N..., n_in, batchsize)
        Cond = randn(Float32, N..., n_cond, batchsize)

        # Invertibility
        XZ, CondZ = G.forward(X,Cond)
        X_ = G.inverse(XZ, CondZ) # saving the cond output is important in split scales because of reshapes
        @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

        ###################################################################################################
        # Test gradients are set and cleared
        gradients_set(G, n_in, n_cond,N; summarized=true)

        ###################################################################################################
        # Gradient test w.r.t. input
        X0 = randn(Float32, N..., n_in, batchsize)
        Cond0 = randn(Float32, N..., n_cond, batchsize)

        dX = X - X0

        f0, ΔX = loss(G, X0, Cond0; summarized=true)[1:2]
        h = 0.1f0
        maxiter = 4
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        print("\nGradient test glow: input\n")
        for j=1:maxiter
            f = loss(G, X0 + h*dX, Cond0; summarized=true)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dX, ΔX))
            print(err1[j], "; ", err2[j], "\n")
            h = h/2f0
        end

        rate1 = err1[1:end-1]./err1[2:end]
        rate2 = err2[1:end-1]./err2[2:end]

        @test isapprox(mean(rate1),2f0; atol=stol)
        @test isapprox(mean(rate2), 4f0; atol=stol)

        # Gradient test w.r.t. parameters
        G0 = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales, ndims=length(N))
        sum_net = ResNet(n_cond, 16, 3; norm=nothing,ndims=length(N)) # make sure it doesnt have any weird normalizations
        G0 = SummarizedNet(G0, sum_net)
        Gini = deepcopy(G0)

        # Test one parameter from residual block
        dW = G.cond_net.CL[1,1].RB.W1.data - G0.cond_net.CL[1,1].RB.W1.data

        f0, ΔX, ΔW = loss(G0, X, Cond; summarized=true)
        h = 0.1f0
        maxiter = 4
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        print("\nGradient test glow: parameter\n")
        for j=1:maxiter
            G0.cond_net.CL[1,1].RB.W1.data = Gini.cond_net.CL[1,1].RB.W1.data + h*dW

            f = loss(G0, X, Cond; summarized=true)[1]
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 - h*dot(dW, ΔW))
            print(err1[j], "; ", err2[j], "\n")
            h = h/2f0
        end

        rate1 = err1[1:end-1]./err1[2:end]
        rate2 = err2[1:end-1]./err2[2:end]

        @test isapprox(mean(rate1),2f0; atol=stol)
        @test isapprox(mean(rate2),4f0; atol=stol)
    end
end