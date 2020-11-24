using InvertibleNetworks, LinearAlgebra, Test, Flux, Zygote

# Initialize invertible/non-invertible layers
nx = 32
ny = 32
n_ch = 16
n_hidden = 64
batchsize = 2
logdet = false
N1 = CouplingLayerHINT(nx, ny, n_ch, n_hidden, batchsize; logdet=logdet, permute="full")
N2 = CouplingLayerHINT(nx, ny, n_ch, n_hidden, batchsize; logdet=logdet, permute="full")
N3 = CouplingLayerHINT(nx, ny, n_ch, n_hidden, batchsize; logdet=logdet, permute="full")
N4 = Chain(Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1), x -> relu.(x), Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1))
N5 = Chain(Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1), x -> relu.(x), Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1))
N6 = CouplingLayerHINT(nx, ny, n_ch, n_hidden, batchsize; logdet=logdet, permute="full")
N7 = CouplingLayerHINT(nx, ny, n_ch, n_hidden, batchsize; logdet=logdet, permute="full")
N8 = Chain(Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1), x -> relu.(x), Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1))
N9 = Chain(Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1), x -> relu.(x), Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1))
N10 = CouplingLayerHINT(nx, ny, n_ch, n_hidden, batchsize; logdet=logdet, permute="full")

# Integrated backward pass w/ Zygote
N = Chain(N1, N2, N3, N4, N5, N6, N7, N8, N9, N10)
