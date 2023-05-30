# Generative model w/ Glow architecture from Kingma & Dhariwal (2018)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Random

# Random seed
Random.seed!(7);

function loss(G, X, Cond)
    Y, ZC, logdet = G.forward(X, Cond)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔX, X_ = G.backward(ΔY, Y, ZC)
    return f, ΔX, G.CL[1,1].RB.W1.grad, G.CL[1,1].C.v1.grad
end

function gradients_set(G, n_in,n_cond,N)
    X = rand(Float32, N..., n_in, batchsize)
    Cond = rand(Float32, N..., n_cond, batchsize)

    XZ, CondZ = G.forward(X,Cond)
    # Set gradients 
    G.backward(XZ, XZ, CondZ)

    P = get_params(G)
    gsum = 0
    for p in P
        ~isnothing(p.grad) && (gsum += 1)
    end
    @test isequal(gsum, L*K*10+2)

    clear_grad!(G)
    gsum = 0
    for p in P
        ~isnothing(p.grad) && (gsum += 1)
    end
    @test isequal(gsum, 0)
end

# Define network
nx = 32
ny = 32
nz = 32
n_in = 3
n_cond = 2
n_hidden = 4
batchsize = 2
L = 2
K = 2

for split_scales in [false,true]
    for N in [(nx),(nx,ny),(nx,ny,nz)]
        println("Test with split_scales = $(split_scales) N = $(N) ")
        ########################################### Test with split_scales = true N = (nx,ny) #########################

        # Network and inputs
        G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales, ndims=length(N))
        X = rand(Float32, N..., n_in, batchsize)
        Cond = rand(Float32, N..., n_cond, batchsize)

        # Invertibility
        XZ, CondZ = G.forward(X,Cond)
        X_ = G.inverse(XZ, CondZ) # saving the cond output is important in split scales because of reshapes
        @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

        ###################################################################################################
        # Test gradients are set and cleared
        gradients_set(G, n_in, n_cond,N)

        ###################################################################################################
        # Gradient test w.r.t. input
        X0 = rand(Float32, N..., n_in, batchsize)
        Cond0 = rand(Float32, N..., n_cond, batchsize)

        dX = X - X0

        f0, ΔX = loss(G, X0, Cond0)[1:2]
        h = 0.01f0
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

        @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
        @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)


        # Gradient test w.r.t. parameters
        G0 = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N))
        Gini = deepcopy(G0)

        # Test one parameter from residual block and 1x1 conv
        dW = G.CL[1,1].RB.W1.data - G0.CL[1,1].RB.W1.data
        dv = G.CL[1,1].C.v1.data - G0.CL[1,1].C.v1.data

        f0, ΔX, ΔW, Δv = loss(G0, X, Cond)
        h = 0.01f0
        maxiter = 4
        err3 = zeros(Float32, maxiter)
        err4 = zeros(Float32, maxiter)

        print("\nGradient test glow: parameter\n")
        for j=1:maxiter
            G0.CL[1,1].RB.W1.data = Gini.CL[1,1].RB.W1.data + h*dW
            G0.CL[1,1].C.v1.data = Gini.CL[1,1].C.v1.data + h*dv

            f = loss(G0, X, Cond)[1]
            err3[j] = abs(f - f0)
            err4[j] = abs(f - f0 - h*dot(dW, ΔW) - h*dot(dv, Δv))
            print(err3[j], "; ", err4[j], "\n")
            h = h/2f0
        end

        @test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f0)
        @test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f0)
    end
end

