using LinearAlgebra, InvertibleNetworks, ForwardDiff, Test, PyPlot

# Input dimension
n = 10:10:300
t_noAD = Array{Float32, 1}(undef, length(n))
t_AD = Array{Float32, 1}(undef, length(n))
n_in = 3
batchsize = 5

# # of evaluations
neval = 10

# Actnorm parameters
s0 = randn(Float32, 1, 1, 1, n_in, 1)
b0 = randn(Float32, 1, 1, 1, n_in, 1)
ds = randn(Float32, size(s0))
db = randn(Float32, size(b0))

# Actnorm jacobian (w/out AD)
function jacobian(ΔX::Array{Float32, 5}, Δs::Array{Float32, 5}, Δb::Array{Float32, 5}, X::Array{Float32, 5}, s::Array{Float32, 5}, b::Array{Float32, 5})
    Y = X .* s .+ b
    ΔY = X .* Δs .+ ΔX .* s .+ Δb
    return ΔY, Y
end

# Loop over input size
for i = 1:length(n)

    # Input
    nx = n[i]
    ny = n[i]
    nz = n[i]
    X0 = randn(Float32, nx, ny, nz, n_in, batchsize)
    dX = randn(Float32, size(X0))

    # Actnorm jacobian (w/ AD)
    function fun(X, s, b)
        return X .* s .+ b
    end
    jacobianAD = t -> ForwardDiff.derivative(t -> fun(X0+t*dX, s0+t*ds, b0+t*db), t)

    # Eval
    dfun = jacobianAD(0f0)
    dfun_ = jacobian(dX, ds, db, X0, s0, b0)[1]

    # Discrepancy
    @test isapprox(dfun, dfun_; rtol=1f-3)

    # Timings
    t_AD[i] = (@elapsed for i = 1:neval jacobianAD(0f0); end;)/neval
    t_noAD[i] = (@elapsed for i = 1:neval jacobian(dX, ds, db, X0, s0, b0)[1]; end;)/neval

    # Print msg
    println("[", i, "/", length(n), "] --- size= ", n[i], ", time= AD:", t_AD[i], ", no AD: ", t_noAD[i])

end

# Plotting
figure()
title(string("Timings for ActNorm jacobian evaluations (AD vs no AD); n_ch=", n_in, ", batchsize=", batchsize))
loglog(n, t_AD)
loglog(n, t_noAD)
legend(["AD", "no AD"])
xlabel("n (size(input)=n^3)")
grid("on")
savefig("AD_vs_noAD.png")