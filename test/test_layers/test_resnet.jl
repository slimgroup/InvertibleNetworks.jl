# Test ResNet

using LinearAlgebra, InvertibleNetworks, Test

# Input
nx = 28
ny = 28
n_in = 4
n_hidden = 8
batchsize = 2
nblocks = 4

# Input
X = glorot_uniform(nx, ny, n_in, batchsize)
X0 = glorot_uniform(nx, ny, n_in, batchsize)
dX = X - X0

# ResNet
RN0 = ResNet(n_in, n_hidden, nblocks; k=3, p=1, s=1, norm=nothing, n_out=nothing)
θ0 = deepcopy(get_params(RN0))
RN = ResNet(n_in, n_hidden, nblocks; k=3, p=1, s=1, norm=nothing, n_out=nothing)
θ = deepcopy(get_params(RN))

# Observed data
Y = RN.forward(X)

RN_dummy = ResNet(n_in, n_hidden, nblocks; k=3, p=1, s=1, norm=nothing, n_out=nothing)
function loss(θ, X, Y; RN=RN_dummy)
    set_params!(RN, θ)
    Y_ = RN.forward(X)
    ΔY = Y_ - Y
    f = .5f0*norm(ΔY)^2
    ΔX = RN.backward(ΔY, X)
    return f, ΔX, deepcopy(get_grads(RN))
end


###################################################################################################
# Gradient tests

# Gradient test w.r.t. input
f0, ΔX = loss(θ, X0, Y)[1:2]
h = 0.1f0
maxiter = 5
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test\n")
for j=1:maxiter
    f = loss(θ, X0 + h*dX, Y)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for weights
dθ = θ-θ0; dθ .*= norm.(dθ)./(norm.(θ0).+1f-10)

f0, ΔX, Δθ = loss(θ, X0, Y)
h = 0.1f0
maxiter = 5
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test convolutions\n")
for j=1:maxiter
    f = loss(θ+h*dθ, X0, Y)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dθ, Δθ))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)