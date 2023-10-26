using InvertibleNetworks, LinearAlgebra, Test, Flux, Random
device = InvertibleNetworks.CUDA.functional() ? gpu : cpu
Random.seed!(42)


# Test utils for LearnableSqueezers

function test_inv_LS(n::NTuple{N,Integer}, nc::Integer, batchsize::Integer, stencil_size::NTuple{N,Integer}, logdet::Bool, do_reverse::Bool) where N

    # Init
    C = LearnableSqueezer(stencil_size...; logdet=logdet) |> device
    do_reverse && (C = reverse(C))
    input_size = ~do_reverse ? (n..., nc, batchsize) : (div.(n, stencil_size)..., prod(stencil_size)*nc, batchsize)
    output_size = do_reverse ? (n..., nc, batchsize) : (div.(n, stencil_size)..., prod(stencil_size)*nc, batchsize)
    X = randn(Float32, input_size) |> device
    Y = randn(Float32, output_size) |> device

    # Test
    if logdet
        @test X ≈ C.inverse(C.forward(X)[1]) rtol=1f-6
        @test Y ≈ C.forward(C.inverse(Y))[1] rtol=1f-6
    else
        @test X ≈ C.inverse(C.forward(X)) rtol=1f-6
        @test Y ≈ C.forward(C.inverse(Y)) rtol=1f-6
    end

end

function loss_LS(LS, X, B)
    LS.logdet ? ((Y, logdet) = LS.forward(X)) : (Y = LS.forward(X))
    LS.logdet ? (f = norm(Y-B)^2/2-logdet) : (f = norm(Y-B)^2/2)
    ΔY = Y-B
    ΔX, X_ = LS.backward(ΔY, Y)
    return f, ΔX, LS.stencil_pars.grad, X_
end

function test_grad_input_LS(n::NTuple{N,Integer}, nc::Integer, batchsize::Integer, stencil_size::NTuple{N,Integer}, logdet::Bool, do_reverse::Bool) where N

    # Init
    C = LearnableSqueezer(stencil_size...; logdet=logdet) |> device
    do_reverse && (C = reverse(C))
    input_size = ~do_reverse ? (n..., nc, batchsize) : (div.(n, stencil_size)..., prod(stencil_size)*nc, batchsize)
    output_size = do_reverse ? (n..., nc, batchsize) : (div.(n, stencil_size)..., prod(stencil_size)*nc, batchsize)
    X = randn(Float32, input_size) |> device
    ΔX = randn(Float32, input_size) |> device; ΔX .*= norm(X)/norm(ΔX)
    B = randn(Float32, output_size) |> device
    Y = C.forward(X); logdet && (Y = Y[1]); B .*= norm(Y)/norm(B)
    
    # Test
    maxiter = 5
    h = 0.01f0
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)
    f0, ΔX0 = loss_LS(C, X, B)[1:2]
    print("\nGradient test input (LearnableSqueezer) for reverse=$(do_reverse), logdet=$(logdet)\n")
    for j=1:maxiter
        f = loss_LS(C, X+h*ΔX, B)[1]
        err1[j] = abs(f-f0)
        err2[j] = abs(f-f0-h*dot(ΔX, ΔX0))
        print(err1[j], "; ", err2[j], "\n")
        h = h/2f0
    end
    @test isapprox(err1[end] / (maximum(err1)/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err2[end] / (maximum(err2)/4^(maxiter-1)), 1f0; atol=1f1)

end

function test_grad_params_LS(n::NTuple{N,Integer}, nc::Integer, batchsize::Integer, stencil_size::NTuple{N,Integer}, logdet::Bool, do_reverse::Bool) where N

    # Init
    C0 = LearnableSqueezer(stencil_size...; logdet=logdet) |> device
    do_reverse && (C0 = reverse(C0))
    θ0 = copy(C0.stencil_pars.data)
    C = LearnableSqueezer(stencil_size...; logdet=logdet) |> device
    do_reverse && (C = reverse(C))
    θ = copy(C.stencil_pars.data)
    dθ = θ-θ0
    input_size = ~do_reverse ? (n..., nc, batchsize) : (div.(n, stencil_size)..., prod(stencil_size)*nc, batchsize)
    output_size = do_reverse ? (n..., nc, batchsize) : (div.(n, stencil_size)..., prod(stencil_size)*nc, batchsize)

    # Parameter gradient    
    X = randn(Float32, input_size) |> device
    Y = C0.forward(X); logdet && (Y = Y[1])
    B = randn(Float32, output_size) |> device
    f0 = loss_LS(C0, X, B)[1]
    ∇θ = copy(C0.stencil_pars.grad)

    # Test
    maxiter = 5
    h = 1f0
    err3 = zeros(Float32, maxiter)
    err4 = zeros(Float32, maxiter)
    f0 = loss_LS(C0, X, B)[1]
    print("\nGradient test weights (LearnableSqueezer) for reverse=$(do_reverse), logdet=$(logdet)\n")
    for j=1:maxiter
        set_params!(C, [Parameter(θ0+h*dθ)])
        f = loss_LS(C, X, B)[1]
        err3[j] = abs(f-f0)
        err4[j] = abs(f-f0-h*dot(∇θ, dθ))
        print(err3[j], "; ", err4[j], "\n")
        h = h/2f0
    end
    @test isapprox(err3[end] / (maximum(err3)/2^(maxiter-1)), 1f0; atol=1f1)
    @test isapprox(err4[end] / (maximum(err4)/4^(maxiter-1)), 1f0; atol=1f1)

end


# Dimensions
n = (3*11, 2*17, 4*7)
nc = 4
batchsize = 3
stencil_size = (3, 2, 4)

# Tests
for N = 1:3, logdet = [true, false], do_reverse = [false, true]

    test_inv_LS(n[1:N], nc, batchsize, stencil_size[1:N], logdet, do_reverse)
    test_grad_input_LS(n[1:N], nc, batchsize, stencil_size[1:N], logdet, do_reverse)
    test_grad_params_LS(n[1:N], nc, batchsize, stencil_size[1:N], logdet, do_reverse)

end