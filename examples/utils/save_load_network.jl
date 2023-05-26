using InvertibleNetworks, Flux, BSON

device = gpu # Will probably by training on GPU, otherwise please change to CPU

L = 2
K = 2 
n_hidden = 8
n_chan_params = 2
G = NetworkGlow(n_chan_params, n_hidden,  L, K; );
G = G |> device;

# Update network parameters by training
batch_size = 4
X = randn(32,32,n_chan_params,batch_size)|> device;
opt = ADAM()

Zx, lgdet = G.forward(X );
G.backward(Zx / batch_size, Zx);

for p in get_params(G) 
	Flux.update!(opt,p.data,p.grad)
end

# Save network parameters using BSON 
net_params = get_params(G) |> cpu; # Needs to be on CPU when saving. 
BSON.@save "net_params.bson" net_params;

# Load in network params
net_params_loaded = BSON.load("net_params.bson")[:net_params]; 

# Make sure you put parameters into exactly same architecture
G_new = NetworkGlow(n_chan_params, n_hidden,  L, K;);
set_params!(G_new,net_params_loaded)
G_new = G_new |> device;

# Test that network was loaded correctly
println(sum(G_new(X)[1]-G(X)[1]).^2)
