# Train networks with flux. Only guaranteed to work with logdet=false for now. 
# So you can train them as invertible networks like this, not as normalizing flows. 
using InvertibleNetworks, Flux

# Glow Network
model = NetworkGlow(2, 32, 2, 5; logdet=false)

# dummy input & target
X = randn(Float32, 16, 16, 2, 2) 
Y = 2 .* X .+ 1

# loss fn
loss(model, X, Y) = Flux.mse(Y, model(X))

θ = Flux.params(model)
opt = ADAM(0.0001f0)

for i = 1:500
    l, grads = Flux.withgradient(θ) do
        loss(model, X, Y)
    end
    @show l
    Flux.update!(opt, θ, grads)
end