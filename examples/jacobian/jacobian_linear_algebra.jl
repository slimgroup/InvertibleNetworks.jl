# Example of Jacobian usage for HINT layer (Kruse et al, 2020)
# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: October 2020

using InvertibleNetworks, LinearAlgebra, Test

# Initialize HINT layer
nx = 32
ny = 32
n_channel = 32
n_hidden = 64
batchsize = 2
logdet = true
N = CouplingLayerHINT(nx, ny, n_channel, n_hidden, batchsize; logdet=logdet, permute="full")
θ = get_params(N)

## f: X × θ ↦ Y, ∂f: ΔX × Δθ ↦ ΔY

# Compute Jacobian
X = randn(Float32, nx, ny, n_channel, batchsize)
J = Jacobian(N, X; io_mode="X×θ↦Y")

# Evaluate Jacobian
ΔX = randn(Float32, size(X))
Δθ = Array{Parameter, 1}(undef, length(θ)); for i = 1:length(θ) Δθ[i] = Parameter(randn(Float32, size(θ[i]))); end;
ΔY = J*(ΔX, Δθ)

# Evaluate adjoint Jacobian
JT = adjoint(J)
ΔY_ = randn(Float32, size(X))
ΔX_, Δθ_ =  JT*ΔY_

# Adjoint Test
a = dot(ΔX, ΔX_)+dot(Δθ, Δθ_)
b = dot(ΔY, ΔY_)
@test isapprox(a, b; rtol=1f-3)

## f(x; ⋅) : θ ↦ Y, ∂f(x): Δθ ↦ ΔY

# Compute Jacobian
X = randn(Float32, nx, ny, n_channel, batchsize)
J = Jacobian(N, X; io_mode="θ↦Y")

# Evaluate Jacobian
ΔX = randn(Float32, size(X))
Δθ = Array{Parameter, 1}(undef, length(θ)); for i = 1:length(θ) Δθ[i] = Parameter(randn(Float32, size(θ[i]))); end;
ΔY = J*Δθ

# Evaluate adjoint Jacobian
JT = adjoint(J)
ΔY_ = randn(Float32, size(X))
Δθ_ =  JT*ΔY_

# Adjoint Test
a = dot(Δθ, Δθ_)
b = dot(ΔY, ΔY_)
@test isapprox(a, b; rtol=1f-3)