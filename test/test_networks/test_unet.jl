using InvertibleNetworks, LinearAlgebra, Test, Random

Random.seed!(11)

# Input
nx = 16
ny = 16
nz = 16
n_in = 2
n_hiddens = [4,8,4]
ds = [1,4,1]
batchsize = 2

# Unrolled loop
L = NetworkUNET(n_in, n_hiddens, ds; n_grad=n_in, early_squeeze=true, ndims=2)

# Initializations
x = 10*randn(Float32, nx, ny, n_in, batchsize)

###################################################################################################

# Test invertibility of inverse pass
y = L.forward(x)

# UL = L
# UL.early_squeeze && (y  = squeeze(y; pattern="checkerboard"))
# x = UL.L.inverse(y)
# UL.early_squeeze && (x  = unsqueeze(x; pattern="checkerboard"))

# x, _ = tensor_split(x; split_index=UL.n_grad)
# x  = UL.AN.inverse(x)[1]   # normalize

x_ = L.inverse(y)
@test isapprox(norm(x_ - x)/norm(x), 0f0, atol=1e-5)

# Test invertibility of backwards pass
Δx, x_ = L.backward(0f0.*y, y)
@test isapprox(norm(x_ - x)/norm(x), 0f0, atol=1e-5)

###################################################################################################

# Initializations
η = randn(Float32, nx, ny, n_in, batchsize)
η0 = randn(Float32, nx, ny, n_in, batchsize)
Δη = η - η0

# Observed data
η_ = L.forward(η)   # only need η

function loss(L, η0, η)
    η_ = L.forward(η0)  # reshape
    Δη = η_ - η
    f = .5f0*norm(Δη)^2
    Δη_ = L.backward(Δη, η_)[1]
    return f, Δη_, L.L.C[1].v1.grad, L.L.RB[1].W1.grad
end

# Gradient test for input
f0, gη = loss(L, η0, η_)[1:2]
h = 0.1f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test loop unrolling\n")
for j=1:maxiter
    f = loss(L, η0 + h*Δη, η_)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(Δη, gη))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f0)


# Gradient test for weights
L0 = NetworkUNET(n_in, n_hiddens, ds; n_grad=n_in, early_squeeze=true, ndims=2)

L_ini = deepcopy(L0)
dv = L.L.C[1].v1.data - L0.L.C[1].v1.data   # just test for 2 parameters
dW = L.L.RB[1].W1.data - L0.L.RB[1].W1.data
f0, gη, gv, gW = loss(L0, η, η_)
h = 0.05f0
maxiter = 5
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test loop unrolling\n")
for j=1:maxiter
    L0.L.C[1].v1.data = L_ini.L.C[1].v1.data + h*dv
    L0.L.RB[1].W1.data  = L_ini.L.RB[1].W1.data + h*dW
    f = loss(L0, η, η_)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dv, gv) - h*dot(dW, gW))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f0)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f0)
