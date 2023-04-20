using InvertibleNetworks, LinearAlgebra, Test, Flux

import InvertibleNetworks: reset!
import ChainRulesCore: rrule

# Initialize invertible/non-invertible layers
nx = 32
ny = 32
n_ch = 16
n_hidden = 64
batchsize = 2
logdet = false
N1 = CouplingLayerHINT(n_ch, n_hidden; logdet=logdet, permute="full")
N2 = CouplingLayerHINT(n_ch, n_hidden; logdet=logdet, permute="full")
N3 = CouplingLayerHINT(n_ch, n_hidden; logdet=logdet, permute="full")
N4 = Chain(Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1), x -> relu.(x), Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1))
N5 = Chain(Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1), x -> relu.(x), Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1))
N6 = CouplingLayerHINT(n_ch, n_hidden; logdet=logdet, permute="full")
N7 = CouplingLayerHINT(n_ch, n_hidden; logdet=logdet, permute="full")
N8 = Chain(Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1), x -> relu.(x), Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1))
N9 = Chain(Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1), x -> relu.(x), Conv((3, 3), n_ch => n_ch; stride = 1, pad = 1))
N10 = CouplingLayerHINT(n_ch, n_hidden; logdet=logdet, permute="full")

# Forward pass + gathering pullbacks
function fw(X)
    X1, ∂1 = getrrule(N1, X)
    X2, ∂2 = getrrule(N2, X1)
    X3, ∂3 = getrrule(N3, X2)
    X5, ∂5 = Flux.Zygote.pullback(Chain(N4, N5), X3)
    X6, ∂6 = getrrule(N6, X5)
    X7, ∂7 = getrrule(N7, X6)
    X9, ∂9 = Flux.Zygote.pullback(Chain(N8, N9), X7)
    X10, ∂10 = getrrule(N10, X9)
    d1 = x -> ∂1(x)[3]
    d2 = x -> ∂2(x)[3]
    d3 = x -> ∂3(x)[3]
    d5 = x -> ∂5(x)[1]
    d6 = x -> ∂6(x)[3]
    d7 = x -> ∂7(x)[3]
    d9 = x -> ∂9(x)[1]
    d10 = x -> ∂10(x)[3]
    return X10, ΔY -> d1(d2(d3(d5(d6(d7(d9(d10(ΔY))))))))
end

# Gradient Test
X = randn(Float32, nx, ny, n_ch, batchsize)
Y0 = randn(Float32, nx, ny, n_ch, batchsize)

loss(X) = 0.5f0*norm(N(X) - Y0)^2

Y, ∂Y = fw(X)

g = ∂Y(Y-Y0)

ΔX_ = randn(Float32, nx, ny, n_ch, batchsize)

# # Integrated backward pass w/ Zygote
reset!(GLOBAL_STATE_INVOPS)
N = Chain(N1, N2, N3, N4, N5, N6, N7, N8, N9, N10);

g2 = gradient(X -> loss(X), X)

@test g ≈ g2[1] rtol=1f-6

## test Reverse network AD

Nrev = reverse(N10)
Xrev, ∂rev = getrrule(Nrev, X)
grev = ∂rev(Xrev-Y0)

g2rev = gradient(X -> 0.5f0*norm(Nrev(X) - Y0)^2, X)

@test grev[3] ≈ g2rev[1] rtol=1f-6
