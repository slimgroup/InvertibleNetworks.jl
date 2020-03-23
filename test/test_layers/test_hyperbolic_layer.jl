using InvertibleNetworks, NNlib, LinearAlgebra, Test

# Data
nx = 28
ny = 28
n_in = 3
batchsize = 1
n_hidden = 3
k = 3   # kernel size
s = 1   # stride
p = 1   # padding

###################################################################################################
# Test layer invertibility

# Data
X0 = randn(Float32, nx, ny, n_in, batchsize)
X1 = randn(Float32, nx, ny, n_in, batchsize)

# Layer
H1 = HyperbolicLayer(nx, ny, n_in, batchsize, k, s, p; action="down", α=.2f0, hidden_factor=8)

Y0, Y1 = H1.forward(X0, X1)
X0_, X1_ = H1.inverse(Y0, Y1)

# Test invertibility
@test isapprox(norm(X0 - X0_)/norm(X0_), 0f0; atol=1f-6)
@test isapprox(norm(X1 - X1_)/norm(X1_), 0f0; atol=1f-6)

###################################################################################################
# Gradient tests

# Initial guess and residual
X00 = randn(Float32, nx, ny, n_in, batchsize)
X01 = randn(Float32, nx, ny, n_in, batchsize)
dX0 = X0 - X00
dX1 = X1 - X01

# Loss
function loss(H, X_prev_in, X_curr_in, Y_curr, Y_new)

    X_curr, X_new = H.forward(X_prev_in, X_curr_in)
    f = .5f0*norm(X_new - Y_new)^2 + .5*norm(X_curr - Y_curr)^2
    ΔX_curr = X_curr - Y_curr
    ΔX_new = X_new - Y_new
    ΔX_prev_, ΔX_curr_, X_prev_, X_curr_ = H.backward(ΔX_curr, ΔX_new, X_curr, X_new)

    # Check inverse is correct
    @test isapprox(norm(X_prev_ - X_prev_in)/norm(X_prev_in), 0f0; atol=1f-6)
    @test isapprox(norm(X_curr_ - X_curr_in)/norm(X_curr_in), 0f0; atol=1f-6) 
    
    return f, ΔX_prev_, ΔX_curr_, H.W.grad, H.b.grad
end


# Observed data
Y_curr, Y_new = H1.forward(X0, X1)

# Gradient test for X
maxiter = 10
print("\nGradient test hyperbolic layer input\n")
f0, ΔX_prev, ΔX_curr = loss(H1, X00, X01, Y_curr, Y_new)[1:3]
h = .1f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)
for j=1:maxiter
    f = loss(H1, X00 + h*dX0, X01 + h*dX1, Y_curr, Y_new)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX0, ΔX_prev) - h*dot(dX1, ΔX_curr))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for W and b
H0 = HyperbolicLayer(nx, ny, n_in, batchsize, k, s, p; action="down", α=.2f0, hidden_factor=8)
Hini = deepcopy(H0)
dW = H1.W.data - H0.W.data
db = H1.b.data - H0.b.data
maxiter = 10
print("\nGradient test hyperbolic layer weights\n")
f0, ΔX_prev, ΔX_curr, ΔW, Δb = loss(H0, X0, X1, Y_curr, Y_new)
h = .1f0
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)
for j=1:maxiter
    H0.W.data = Hini.W.data + h*dW
    H0.b.data = Hini.b.data + h*db
    f = loss(H0, X0, X1, Y_curr, Y_new)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dW, ΔW) - h*dot(db, Δb))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)

