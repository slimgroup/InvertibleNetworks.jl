using InvertibleNetworks
using Random
using PyPlot
using LinearAlgebra
using JLD2
using Flux
using Downloads 

# Training params
device = gpu

noise_lev = 0.005f0 # Additive noise
n_epochs = 60
batch_size = 64
n_slices = 16
lr = 2f-3 # can increase to 2f-3 if you do clipnorm
clip_norm = 5f0

plot_every = 1

data_path = "compass_samples_64.jld2"
if isfile(data_path) == false
    println("Downloading data...");
    Downloads.download("ftp://slim.gatech.edu/data/synth/Compass/compass_samples_64.jld2", data_path)
end

#X_3d  = jldopen(data_path, "r")["X"][:,:,:,:,1:(end-4)]

#take out water to focus on more interesting features
inds_no_water = []
for i in 1:size(X_3d)[end]
	if !(1480f0 in X_3d[:,:,:,:,i])
		append!(inds_no_water,i)
	end
end
n_total = length(inds_no_water)

Xs = zeros(Float32, 64,64,2,n_total*n_slices)
for j in 1:n_total
	for i in 1:n_slices
		slice = X_3d[i,:,:,:,inds_no_water[j]] 
		slice = (slice .- minimum(slice)) ./ (maximum(slice) .- minimum(slice))
		Xs[:,:,1,(j-1)*n_slices+i] = slice

		slice = X_3d[:,i,:,:,inds_no_water[j]] 
		slice = (slice .- minimum(slice)) ./ (maximum(slice) .- minimum(slice))
		Xs[:,:,2,(j-1)*n_slices+i] = slice
	end 
end 
Xs = reshape(Xs, 64,64,1,:)
n_train = size(Xs)[end]

train_split = 20000
X_train = Xs[:,:,:,1:train_split]
X_test  = Xs[:,:,:,(train_split+1):end]

n_train_total  = size(X_train)[end]
n_batches = cld(n_train_total, batch_size)-1
n_train = n_batches*batch_size
X_train = X_train[:,:,:,1:n_train]

n_test_total   = size(X_test)[end]
n_batches = cld(n_test_total, batch_size)-1
n_test = n_batches*batch_size
X_test = X_test[:,:,:,1:n_test]

nx, ny, nz, _, = size(X_train) 
N = nx*ny;

# Testing batches latent 
X_test_latent  = X_test[:,:,:,1:batch_size];
X_test_latent  .+= noise_lev*randn(Float32, size(X_test_latent));

# Architecture parametrs
L = 4 
K = 12 
n_hidden = 64 
split_scales = true
# Create network

# Random seed
Random.seed!(20) 
ZX_noise = randn(Float32, nx, ny, 1, batch_size);

Random.seed!(20)
G = NetworkGlow(1, n_hidden, L, K;  split_scales=split_scales, activation=SigmoidLayer(low=0.5f0,high=1.0f0)) 
G = G |> device

n_batches = cld(n_train, batch_size)

# Optimizer
opt =  ADAM(lr)

# Training logs
loss = [];
logdet = [];
loss_test = [];
logdet_test = [];

for e=1:n_epochs
	idx_e = reshape(randperm(n_train), batch_size, n_batches)
	for b = 1:5#n_batches # batch loop
			X = X_train[:, :, :, idx_e[:,b]]
			X .+= noise_lev*randn(Float32, size(X))

			Zx, lgdet = G.forward(X|> device)

			# Loss function is l2 norm 
			append!(loss, norm(Zx)^2 / (N*batch_size))  # normalize by image size and batch size
			append!(logdet, -lgdet / N) # logdet is internally normalized by batch size

			G.backward((Zx / batch_size)[:], (Zx)[:])	

			for p in get_params(G)
				Flux.update!(opt,p.data,p.grad)
			end
			clear_grad!(G)

			print("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches,  
			"; f l2 = ",  loss[end], 
			"; lgdet = ", logdet[end], "; f = ", loss[end] + logdet[end], "\n")
			Base.flush(Base.stdout)
	end

    # Evaluate network on train and test batch_size for qq plots
    ZX_test, lgdet_test = G.forward(X_test_latent |> device);

    append!(logdet_test, -lgdet_test / N)
    append!(loss_test, norm(ZX_test)^2f0 / (N*batch_size));


    if mod(e,plot_every) == 0 
    	# Training logs
	    sum_train = loss + logdet
		sum_test = loss_test + logdet_test

		fig1 = figure(figsize=(9,6))
		subplot(3,1,1); title("L2 Term: train="*string(loss[end])*" test="*string(loss_test[end]))
		plot(loss, label="train");
		plot(n_batches:n_batches:n_batches*e, loss_test, label="test"); 
		axhline(y=1,color="red",linestyle="--",label="Normal Noise")
		xlabel("Parameter Update"); legend();
		ylim(0,2)

		subplot(3,1,2); title("Logdet Term: train="*string(logdet[end])*" test="*string(logdet_test[end]))
		plot(logdet);
		plot(n_batches:n_batches:n_batches*e, logdet_test);
		xlabel("Parameter Update") ;

		subplot(3,1,3); title("Total Objective: train="*string(sum_train[end])*" test="*string(sum_test[end]))
		plot(sum_train); 
		plot(n_batches:n_batches:n_batches*e, sum_test); 
		xlabel("Parameter Update") ;

		tight_layout()
		savefig("log.png");

		# Make generative samples
		X_gen = G.inverse( ZX_noise[:]|> device) |> cpu;

		fig4 = figure(figsize=(10,5))
		subplot(2,4,1); imshow(X_gen[:,:,1,1]'|> cpu)
		subplot(2,4,2); imshow(X_gen[:,:,1,2]'|> cpu)
		subplot(2,4,3); imshow(X_gen[:,:,1,3]'|> cpu)
		subplot(2,4,4); imshow(X_gen[:,:,1,4]'|> cpu)

		subplot(2,4,5); imshow(X_gen[:,:,1,5]'|> cpu)
		subplot(2,4,6); imshow(X_gen[:,:,1,6]'|> cpu)
		subplot(2,4,7); imshow(X_gen[:,:,1,7]'|> cpu)
		subplot(2,4,8); imshow(X_gen[:,:,1,8]'|> cpu)

		savefig("test_gen.png");
	end
end

