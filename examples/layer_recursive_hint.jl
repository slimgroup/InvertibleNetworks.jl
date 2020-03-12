# Invertible CNN layer from Dinh et al. (2017)/Kingma & Dhariwal (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test

# Input
nx = 32
ny = 32
n_channel = 64
n_hidden = 64
batchsize = 2
k1 = 4
k2 = 3

# Input image
X = glorot_uniform(nx, ny, n_channel, batchsize)
X0 = glorot_uniform(nx, ny, n_channel, batchsize)

# Get network depth
function get_depth(n_channel)
    count = 0
    nc = n_channel
    while nc > 4
        nc /= 2
        count += 1
    end
    return count +1
end

# Create network
n = get_depth(n_channel)
L = Array{CouplingLayer}(undef, n) 
for j=1:n
    L[j] = CouplingLayer(nx, ny, Int(n_channel/2^j), n_hidden, batchsize; k1=k1, k2=k2, p1=0, p2=1)
end

function hint(X, scale)
    print(size(X, 3), "\n")
    if size(X, 3) > 4
        Xa, Xb = tensor_split(X)
        Ya = hint(Xa, scale+1)
        Yb = L[scale].forward(hint(Xa, scale+1), Xb)[1]
    else
        global j += 1
        Xa, Xb = tensor_split(X)
        Ya = copy(Xa)
        Yb = L[scale].forward(Xa, Xb)[1]
    end
    Y = cat(Ya, Yb; dims=3)
    return Y
end