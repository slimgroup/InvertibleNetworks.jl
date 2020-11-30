using LinearAlgebra, InvertibleNetworks, ForwardDiff, Flux, Test

# Input dimension
nx = 101
ny = 201
n_in = 10
batchsize = 30
X0 = randn(Float32, nx, ny, n_in, batchsize)
dX = randn(Float32, size(X0))

# Convolution dimension
k = 5
p = 2
s = 1
n_out = 20
cdims = DenseConvDims((nx, ny, n_in, batchsize), (k, k, n_in, n_out); stride=(s, s), padding=(p, p))
W0 = randn(Float32, k, k, n_in, n_out)
dW = randn(Float32, k, k, n_in, n_out)

# Function
function fun(X, W)
    return conv(X, W, cdims)
end

function jacobian(ΔX, ΔW, X, W)
    Y = conv(X, W, cdims)
    ΔY = conv(ΔX, W, cdims)+conv(X, ΔW, cdims)
    return ΔY, Y
end

# AD (forward-mode)
jacobian_ = t -> ForwardDiff.derivative(t -> fun(X0+t*dX, W0+t*dW), t)
dfun = jacobian_(0f0)

# "Manual" differentiation
dfun_ = jacobian(dX, dW, X0, W0)[1]

# Discrepancy
@test isapprox(dfun, dfun_; rtol=1f-3)

# Timings
@time for i = 1:10 jacobian_(0f0); end;
@time for i = 1:10 jacobian(dX, dW, X0, W0)[1]; end;