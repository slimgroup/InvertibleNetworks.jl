# Invertible CNN layer from Dinh et al. (2017)/Kingma & Dhariwal (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test

# Input
nx = 32
ny = 32
n_channel = 32
n_hidden = 64
batchsize = 2
k1 = 4
k2 = 3

# Input image
Xa = glorot_uniform(nx, ny, n_channel, batchsize)
Xb = glorot_uniform(nx, ny, n_channel, batchsize)

X0a = glorot_uniform(nx, ny, n_channel, batchsize)
X0b = glorot_uniform(nx, ny, n_channel, batchsize)

# Invertible splitting layer
L = CouplingLayer(nx, ny, n_channel, n_hidden, batchsize; k1=k1, k2=k2, p1=0, p2=1, logdet=true)

# Forward + backward
Ya, Yb = L.forward(Xa, Xb)[1:2]
Y0a, Y0b, logdet = L.forward(X0a, X0b)
ΔYa = Y0a - Ya
ΔYb = Y0b - Yb
ΔXa, ΔXb, X0a_, X0b_ = L.backward(ΔYa, ΔYb, Y0a, Y0b)

@test isapprox(norm(X0a_ - X0a)/norm(X0a), 0f0, atol=1f-2)
@test isapprox(norm(X0b_ - X0b)/norm(X0b), 0f0, atol=1f-2)

function hint(X)
    if size(X, 3) > 4
        Xa, Xb = tensor_split(X)
        Ya = hint(Xa)
        Yb = L.forward(hint(Xa), Xb)
    else

    end
end