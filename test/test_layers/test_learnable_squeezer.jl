using InvertibleNetworks, LinearAlgebra, Test, Flux, Random
device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
Random.seed!(11)


# Dimensions
n = (2*17, 3*11, 4*7)
nc = 4
batchsize = 3
k = (2, 3, 4)

for N = 1:3

    # Test invertibility
    C = LearnableSqueezer(k[1:N]...) |> device
    X = randn(Float32, n[1:N]..., nc, batchsize) |> device
    Y = randn(Float32, div.(n, k)[1:N]..., prod(k[1:N])*nc, batchsize) |> device
    @test X ≈ C.inverse(C.forward(X)) rtol=1f-6
    @test Y ≈ C.forward(C.inverse(Y)) rtol=1f-6


    # Test backward/inverse coherence
    ΔY = randn(Float32, div.(n, k)[1:N]..., prod(k[1:N])*nc, batchsize) |> device
    Y  = randn(Float32, div.(n, k)[1:N]..., prod(k[1:N])*nc, batchsize) |> device
    X_ = C.inverse(Y)
    _, X = C.backward(ΔY, Y)
    @test X ≈ X_ rtol=1f-6


    # Gradient test (input)
    ΔY = randn(Float32, div.(n, k)[1:N]..., prod(k[1:N])*nc, batchsize) |> device
    ΔX = randn(Float32, n[1:N]..., nc, batchsize) |> device
    X  = randn(Float32, n[1:N]..., nc, batchsize) |> device
    Y = C.forward(X)
    ΔX_, _ = C.backward(ΔY, Y)
    @test dot(ΔX, ΔX_) ≈ dot(C.forward(ΔX), ΔY) rtol=1f-4


    # Gradient test (parameters)
    using CUDA
    T = Float64
    C = LearnableSqueezer(k[1:N]...) |> device; C.stencil_pars.data = cu(C.stencil_pars.data)
    X  = CUDA.randn(T, n[1:N]..., nc, batchsize)
    ΔY_ = CUDA.randn(T, div.(n, k)[1:N]..., prod(k[1:N])*nc, batchsize)
    θ = copy(C.stencil_pars.data)
    Δθ = CUDA.randn(T, size(θ)); Δθ *= norm(θ)/norm(Δθ)

    t = T(1e-5)
    C.stencil_pars.data = θ+t*Δθ/2; C.reset = true
    Yp1 = C.forward(X)
    C.stencil_pars.data = θ-t*Δθ/2; C.reset = true
    Ym1 = C.forward(X)
    ΔY = (Yp1-Ym1)/t
    C.stencil_pars.data = θ; C.reset = true
    Y = C.forward(X)
    C.backward(ΔY_, Y)
    Δθ_ = C.stencil_pars.grad

    @test dot(ΔY, ΔY_) ≈ dot(Δθ, Δθ_) rtol=T(1e-4)

end