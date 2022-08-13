using InvertibleNetworks, Flux, LinearAlgebra, Random

# Architecture parameters
device = cpu
n_in   = 2

# Create network
Random.seed!(123);
G = NetworkGlow(n_in, 1, 1, 1;);

# trainable parameters
ps = Flux.params(G)

function logp(G, X;)
	Z, logdet = G(X) 
    #Z, logdet = G.forward(X) # MATHIAS HELP ME WHY THIS DOESNT WORK???????

    norm(Z)^2 / (prod(size(X)))  - logdet / (prod(size(X)[1:(end-1)])) 
end

opt = ADAM(0.1)
X = 20f0 .* randn(Float32, 16,16,n_in,4) |> device;

for i = 1:10
    # # DOESNT WORK YET
    # loss, grad = Flux.withgradient(ps) do
    #     logp(G, X;)
    # end
    # Flux.Optimise.update!(opt, ps, grad)

    # Get gradient wrt to network
    gr_g = gradient(net -> logp(net,X), G)
    Flux.Optimise.update!(opt, ps, gr_g[1])
    loss = logp(G, X;)

    println("loss at iter=$(i) is $(loss)") 
    clear_grad!(G) 
end





