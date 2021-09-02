# HINT network from Kruse et al. (2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Random
Random.seed!(11)

# Define network
nx = 32
ny = 32
n_in = 2
n_hidden = 4
batchsize = 2
L = 2
K = 2

# single scale network
CH0  = NetworkHINT(n_in, n_hidden, L*K; k1=3, k2=1, p1=1, p2=0)
CH1  = NetworkHINT(n_in, n_hidden, L*K; logdet=false, k1=3, k2=1, p1=1, p2=0)

nets = [CH0, CH1, reverse(CH0), reverse(CH1)]

function test_inv(CH, nx, ny, n_in)
    print("\nInvertibility test HINT network\n")
    # Test layers
    test_size = 10
    X = randn(Float32, nx, ny, n_in, test_size)

    if CH.logdet 
        Zx = CH.forward(X)[1]
    else
        Zx = CH.forward(X)
    end    

    # Forward-backward
    X_ = CH.backward(0f0.*Zx, Zx)[2]
    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-3)

    # Forward-inverse
    X_ = CH.inverse(Zx)
    @test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-3)
end

# Loss
function loss(CH, X)
    if CH.logdet 
        Zx, logdet = CH.forward(X)
        f = -log_likelihood(Zx) - logdet
    else
        Zx = CH.forward(X)
        f = -log_likelihood(Zx) 
    end    

    ΔZx = -∇log_likelihood(Zx)
    ΔX = CH.backward(ΔZx, Zx)[1]
    return f, ΔX
end

function test_grad(CH, nx, ny, n_in)
    test_size = 10
    X = randn(Float32, nx, ny, n_in, test_size)
    X0 = randn(Float32, nx, ny, n_in, test_size)
    dX = X - X0

    f0, ΔX = loss(CH, X0)
    h = 0.1f0
    maxiter = 6
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)

    print("\nGradient test cond. HINT net: input\n")
    for j=1:maxiter
        f = loss(CH, X0 + h*dX)[1]
        err1[j] = abs(f - f0)
        err2[j] = abs(f - f0 - h*dot(dX, ΔX))
        print(err1[j], "; ", err2[j], "\n")
        h = h/2f0
    end

    @test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
end

# Loop over networks and reversed counterpart
for CH in nets
    test_inv(CH, nx, ny, n_in)
    test_grad(CH, nx, ny, n_in)
end

###################################################################################################
# Jacobian-related tests: NetworkConditionalHINT

# Gradient test

# Initialization
CH = NetworkHINT(n_in, n_hidden, L*K; k1=3, k2=1, p1=1, p2=0, logdet=true); 
CH.forward(randn(Float32, nx, ny, n_in, batchsize))
θ = deepcopy(get_params(CH))

CH0 = NetworkHINT(n_in, n_hidden, L*K; k1=3, k2=1, p1=1, p2=0, logdet=true); 
CH0.forward(randn(Float32, nx, ny, n_in, batchsize))
θ0 = deepcopy(get_params(CH0))

X = randn(Float32, nx, ny, n_in, batchsize)

# Perturbation (normalized)
dθ = θ-θ0; dθ .*= norm.(θ0)./(norm.(dθ).+1f-10)
dX = randn(Float32, nx, ny, n_in, batchsize); dX *= norm(X)/norm(dX)

# Jacobian eval
dZx, Zx, _ = CH.jacobian(dX, dθ, X)

# Test
print("\nJacobian test\n")
h = 0.1f0
maxiter = 5
err5 = zeros(Float32, maxiter)
err6 = zeros(Float32, maxiter)
for j=1:maxiter
    set_params!(CH, θ+h*dθ)
    Zx_, _ = CH.forward(X+h*dX)
    err5[j] = sqrt(norm(Zx_ - Zx)^2
    err6[j] = sqrt(norm(Zx_ - Zx - h*dZx)^2
    print(err5[j], "; ", err6[j], "\n")
    global h = h/2f0
end

@test isapprox(err5[end] / (err5[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err6[end] / (err6[1]/4^(maxiter-1)), 1f0; atol=1f1)

# Adjoint test
set_params!(CH, θ)
dZx, Zx, _= CH.jacobian(dX, dθ, X)
dZx_ = randn(Float32, size(dZx)); 
dX_, dθ_, _= CH.adjointJacobian(dZx_, Zx)
a = dot(dZx, dZx_)
b = dot(dX, dX_)+dot(dθ, dθ_)
@test isapprox(a, b; rtol=1f-3)
