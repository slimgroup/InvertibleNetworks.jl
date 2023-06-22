# Generative model w/ Glow architecture from Kingma & Dhariwal (2018)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Random
using Statistics 

# Random seed
Random.seed!(10);

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
        println("Test with split_scales = $(split_scales) N = $(N) and summarized=$(summary)")
        
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
        X0 = rand(Float32, N..., n_in, batchsize)
        Cond0 = rand(Float32, N..., n_cond, batchsize)

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

        @test isapprox(mean(rate1),2f0; atol=stol)
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

for split_scales in [false,true]
    for N in [(16*nx),(nx,ny),(nx,ny,nz)]
        println("Test with split_scales = $(split_scales) N = $(N) and summarized=$(true)")
        
        # Network and inputs
        G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales, ndims=length(N))
        sum_net = ResNet(n_cond, 16, 3; norm=nothing,ndims=length(N)) # make sure it doesnt have any weird normalizations
        G = SummarizedNet(G, sum_net)

        X = randn(Float32, N..., n_in, batchsize)
        Cond = rand(Float32, N..., n_cond, batchsize)

        # Invertibility
        XZ, CondZ = G.forward(X,Cond)
        X_ = G.inverse(XZ, CondZ) # saving the cond output is important in split scales because of reshapes
        @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

        ###################################################################################################
        # Test gradients are set and cleared
        gradients_set(G, n_in, n_cond,N; summarized=true)

        ###################################################################################################
        # Gradient test w.r.t. input
        X0 = rand(Float32, N..., n_in, batchsize)
        Cond0 = rand(Float32, N..., n_cond, batchsize)

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
        @test isapprox(mean(rate2), 4f0; atol=stol)
    end
end







########################################### Test with split_scales = true N = (nx,ny) #########################
# Invertibility

# Network and input
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N))
X = rand(Float32, N..., n_in, batchsize)
Cond = rand(Float32, N..., n_cond, batchsize)

Y, Cond = G.forward(X,Cond)
X_ = G.inverse(Y,Cond) # saving the cond is important in split scales because of reshapes

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

# Test gradients are set and cleared
G.backward(Y, Y, Cond)

P = get_params(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, L*K*10)

clear_grad!(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, 0)

###################################################################################################
# Gradient test

function loss(G, X, Cond)
    Y, ZC, logdet = G.forward(X, Cond)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔX = G.backward(ΔY, Y, ZC)[1]
    return f, ΔX, G.CL[1,1].RB.W1.grad, G.CL[1,1].C.v1.grad
end


# Gradient test w.r.t. input
#G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N))
X = rand(Float32, N..., n_in, batchsize)
Cond = rand(Float32, N..., n_cond, batchsize)
X0 = rand(Float32, N..., n_in, batchsize)
Cond0 = rand(Float32, N..., n_cond, batchsize)

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
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)


# Gradient test w.r.t. parameters
X = rand(Float32, N..., n_in, batchsize)
#G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N))
G0 = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N))
Gini = deepcopy(G0)

# Test one parameter from residual block and 1x1 conv
dW = G.CL[1,1].RB.W1.data - G0.CL[1,1].RB.W1.data
dv = G.CL[1,1].C.v1.data - G0.CL[1,1].C.v1.data

f0, ΔX, ΔW, Δv = loss(G0, X, Cond)
h = 0.1f0
maxiter = 4
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    G0.CL[1,1].RB.W1.data = Gini.CL[1,1].RB.W1.data + h*dW
    G0.CL[1,1].C.v1.data = Gini.CL[1,1].C.v1.data + h*dv

    f = loss(G0, X, Cond)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW, ΔW) - h*dot(dv, Δv))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f0)



########################################### Test with split_scales = true N = (nx,ny) and summary network #########################
# Invertibility
sum_net = ResNet(n_cond, 16, 3; norm=nothing) # make sure it doesnt have any weird normalizations

# Network and input
flow = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales,ndims=length(N))
G = SummarizedNet(flow, sum_net)

X = rand(Float32, N..., n_in, batchsize);
Cond = rand(Float32, N..., n_cond, batchsize);

Y, ZCond = G.forward(X,Cond)
X_ = G.inverse(Y,ZCond) # saving the cond is important in split scales because of reshapes

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

# Test gradients are set and cleared
G.backward(Y, Y, ZCond; Y_save = Cond)

P = get_params(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, L*K*10+12) # depends on summary net you use

clear_grad!(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, 0)


# Gradient test


# Gradient test w.r.t. input
X = rand(Float32, N..., n_in, batchsize);
Cond = rand(Float32, N..., n_cond, batchsize);
X0 = rand(Float32, N..., n_in, batchsize);
Cond0 = rand(Float32, N..., n_cond, batchsize);

dX = X - X0

f0, ΔX = loss_sum(G, X0, Cond0)[1:2]
h = 0.1f0
maxiter = 4
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    f = loss_sum(G, X0 + h*dX, Cond0)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)


# Gradient test w.r.t. parameters
X = rand(Float32, N..., n_in, batchsize)
flow0 = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales,ndims=length(N))
G0 = SummarizedNet(flow0, sum_net)
Gini = deepcopy(G0)

# Test one parameter from residual block and 1x1 conv
dW = G.cond_net.CL[1,1].RB.W1.data - G0.cond_net.CL[1,1].RB.W1.data
dv = G.cond_net.CL[1,1].C.v1.data - G0.cond_net.CL[1,1].C.v1.data

f0, ΔX, ΔW, Δv = loss_sum(G0, X, Cond)
h = 0.1f0
maxiter = 4
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    G0.cond_net.CL[1,1].RB.W1.data = Gini.cond_net.CL[1,1].RB.W1.data + h*dW
    G0.cond_net.CL[1,1].C.v1.data = Gini.cond_net.CL[1,1].C.v1.data + h*dv

    f = loss_sum(G0, X, Cond)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW, ΔW) - h*dot(dv, Δv))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f0)


N = (nx,ny,nz)
########################################### Test with split_scales = true N = (nx,ny,nz) #########################
# Invertibility

# Network and input
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N))
X = rand(Float32, N..., n_in, batchsize)
Cond = rand(Float32, N..., n_cond, batchsize)

Y, Cond = G.forward(X,Cond)
X_ = G.inverse(Y,Cond) # saving the cond is important in split scales because of reshapes

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

# Test gradients are set and cleared
G.backward(Y, Y, Cond)

P = get_params(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, L*K*10)

clear_grad!(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, 0)


# Gradient test


# Gradient test w.r.t. input
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N))
X = rand(Float32, N..., n_in, batchsize)
Cond = rand(Float32, N..., n_cond, batchsize)
X0 = rand(Float32, N..., n_in, batchsize)
Cond0 = rand(Float32, N..., n_cond, batchsize)

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
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)


# Gradient test w.r.t. parameters
X = rand(Float32, N..., n_in, batchsize)
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N))
G0 = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N))
Gini = deepcopy(G0)

# Test one parameter from residual block and 1x1 conv
dW = G.CL[1,1].RB.W1.data - G0.CL[1,1].RB.W1.data
dv = G.CL[1,1].C.v1.data - G0.CL[1,1].C.v1.data

f0, ΔX, ΔW, Δv = loss(G0, X, Cond)
h = 0.1f0
maxiter = 4
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    G0.CL[1,1].RB.W1.data = Gini.CL[1,1].RB.W1.data + h*dW
    G0.CL[1,1].C.v1.data = Gini.CL[1,1].C.v1.data + h*dv

    f = loss(G0, X, Cond)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW, ΔW) - h*dot(dv, Δv))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f0)


########################################### Test with split_scales = true N = (nx,ny,nz) and Summary network #########################
# Invertibility
sum_net_3d = ResNet(n_cond, 16, 3; ndims=3, norm=nothing) # make sure it doesnt have any weird normalizati8ons

# Network and input
flow = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales,ndims=length(N));
G = SummarizedNet(flow, sum_net_3d)

X = rand(Float32, N..., n_in, batchsize);
Cond = rand(Float32, N..., n_cond, batchsize);

Y, ZCond = G.forward(X,Cond);
X_ = G.inverse(Y,ZCond); # saving the cond is important in split scales because of reshapes

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

# Test gradients are set and cleared
G.backward(Y, Y, ZCond; Y_save=Cond)

P = get_params(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, L*K*10+12)

clear_grad!(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, 0)


# Gradient test


# Gradient test w.r.t. input
X = rand(Float32, N..., n_in, batchsize);
Cond = rand(Float32, N..., n_cond, batchsize);
X0 = rand(Float32, N..., n_in, batchsize);
Cond0 = rand(Float32, N..., n_cond, batchsize);

dX = X - X0;

f0, ΔX = loss_sum(G, X0, Cond0)[1:2];
h = 0.1f0
maxiter = 4
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    f = loss_sum(G, X0 + h*dX, Cond0)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)

# Gradient test w.r.t. parameters
X = rand(Float32, N..., n_in, batchsize)
flow0 = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K; split_scales=split_scales,ndims=length(N))
G0 = SummarizedNet(flow0, sum_net_3d)
Gini = deepcopy(G0)

# Test one parameter from residual block and 1x1 conv
dW = G.cond_net.CL[1,1].RB.W1.data - G0.cond_net.CL[1,1].RB.W1.data
dv = G.cond_net.CL[1,1].C.v1.data - G0.cond_net.CL[1,1].C.v1.data

f0, ΔX, ΔW, Δv = loss_sum(G0, X, Cond);
h = 0.1f0
maxiter = 4
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    G0.cond_net.CL[1,1].RB.W1.data = Gini.cond_net.CL[1,1].RB.W1.data + h*dW
    G0.cond_net.CL[1,1].C.v1.data = Gini.cond_net.CL[1,1].C.v1.data + h*dv

    f = loss_sum(G0, X, Cond)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW, ΔW) - h*dot(dv, Δv))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f0)

