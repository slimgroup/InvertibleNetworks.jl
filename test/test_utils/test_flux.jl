using InvertibleNetworks, Flux, Test, LinearAlgebra

# Define network
nx = 1
ny = 1
n_in = 2
n_hidden = 10
batchsize = 32

# net
AN = ActNorm(n_in; logdet = false)
C = CouplingLayerGlow(n_in, n_hidden; logdet = false, k1 = 1, k2 = 1, p1 = 0, p2 = 0)
pan, pc = deepcopy(get_params(AN)), deepcopy(get_params(C))
model = Chain(AN, C)

# dummy input & target
X = randn(Float32, nx, ny, n_in, batchsize)
Y = model(X)
X0 = rand(Float32, nx, ny, n_in, batchsize) .+ 1

# loss fn
loss(model, X, Y) = Flux.mse(Y, model(X))

# old, implicit-style Flux
θ = Flux.params(model)
opt = Descent(0.001)

l, grads = Flux.withgradient(θ) do
    loss(model, X0, Y)
end

for θi in θ
    @test θi ∈ keys(grads.grads)
    @test !isnothing(grads.grads[θi])
    @test size(grads.grads[θi]) == size(θi)
end

Flux.update!(opt, θ, grads)

for i = 1:5
    li, grads = Flux.withgradient(θ) do
        loss(model, X, Y)
    end

    @info "Loss: $li"
    @test li != l
    global l = li

    Flux.update!(opt, θ, grads)
end