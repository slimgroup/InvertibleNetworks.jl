# Generative model w/ Glow architecture from Kingma & Dhariwal (2018)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using Revise
using InvertibleNetworks, LinearAlgebra, Test, Random
using Flux 

# Random seed
Random.seed!(3);

# Define network
nx = 32
ny = 32
nz = 32
n_in = 2
n_cond = 2
n_hidden = 4
batchsize = 2
L = 2
K = 2
split_scales = true
N = (nx,ny)

########################################### Test with split_scales = true N = (nx,ny) #########################
# Invertibility

# Network and input
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N))
X = rand(Float32, N..., n_in, batchsize);
Cond = rand(Float32, N..., n_cond, batchsize);

Y, Cond = G.forward(X,Cond);
X_ = G.inverse(Y,Cond); # saving the cond is important in split scales because of reshapes

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

# Test gradients are set and cleared
G.backward(Y, Y, Cond, Cond);

P = get_params(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, L*K*10+2)

clear_grad!(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, 0)

###################################################################################################
# Gradient test


# Gradient test
function loss_y(G, X, Cond)
    Y, ZC, logdet = G.forward(X, Cond)
    f = -log_likelihood(Y) -log_likelihood(ZC) - logdet
    ΔY  = -∇log_likelihood(Y)
    ΔZC = -∇log_likelihood(ZC)
    ΔX, X_, ΔY = G.backward(ΔY, Y, ΔZC, ZC)
    return f, ΔX, ΔY, G.CL[1,1].RB.W1.grad, G.CL[1,1].C.v1.grad
end

# Gradient test w.r.t. condition input
G = NetworkConditionalGlow(n_in, n_cond, n_hidden, L, K;split_scales=split_scales,ndims=length(N))
X = rand(Float32, N..., n_in, batchsize);
Cond = rand(Float32, N..., n_cond, batchsize);
X0 = rand(Float32, N..., n_in, batchsize);
Cond0 = rand(Float32, N..., n_cond, batchsize);

dCond = Cond - Cond0

f0, ΔX, ΔY = loss_y(G, X0, Cond0)[1:3]
h = 0.1f0
maxiter = 4
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    f = loss_y(G, X0, Cond0 + h*dCond)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dCond, ΔY))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)



function loss(G, X, Cond)
    Y, ZC, logdet = G.forward(X, Cond)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔX, X_ = G.backward(ΔY, Y, 0f0 .* ZC, ZC)
    return f, ΔX, G.CL[1,1].RB.W1.grad, G.CL[1,1].C.v1.grad
end

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
G.backward(Y, Y, ZCond, ZCond; Y_save = Cond)

P = get_params(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, L*K*10+2+12) # depends on summary net you use

clear_grad!(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, 0)


# Gradient test
function loss_sum(G, X, Cond)
    Y, ZC, logdet = G.forward(X, Cond)
    f = -log_likelihood(Y) -log_likelihood(ZC) - logdet
    ΔY = -∇log_likelihood(Y)
    ΔZC = -∇log_likelihood(ZC)
    ΔX,  X_, ΔC = G.backward(ΔY, Y, ΔZC,  ZC; Y_save=Cond)
    return f, ΔX, ΔC, G.cond_net.CL[1,1].RB.W1.grad, G.cond_net.CL[1,1].C.v1.grad
end

# Gradient test w.r.t. input
X = rand(Float32, N..., n_in, batchsize);
Cond = rand(Float32, N..., n_cond, batchsize);
X0 = rand(Float32, N..., n_in, batchsize);
Cond0 = rand(Float32, N..., n_cond, batchsize);

dCond = Cond - Cond0

f0, ΔX, ΔC  = loss_sum(G, X0, Cond0)[1:3]
h = 0.1f0
maxiter = 4
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test glow: input\n")
for j=1:maxiter
    f = loss_sum(G, X0, Cond0 + h*dCond)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dCond, ΔC))
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

f0, ΔX, ΔC, ΔW, Δv = loss_sum(G0, X, Cond)
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
G.backward(Y, Y, Cond,Cond)

P = get_params(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, L*K*10+2)

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
G.backward(Y, Y,ZCond,ZCond; Y_save=Cond)

P = get_params(G)
gsum = 0
for p in P
    ~isnothing(p.grad) && (global gsum += 1)
end
@test isequal(gsum, L*K*10+2+12)

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

f0, ΔX,ΔC, ΔW, Δv = loss_sum(G0, X, Cond);
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

