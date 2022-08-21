using InvertibleNetworks
using PyPlot
using Flux
using LinearAlgebra
using MLDatasets
using Statistics
using ProgressMeter: Progress, next!
using Images

function posterior_sampler(G, y, size_x; num_samples=16)
	# make samples from posterior
    Y_repeat = repeat(y |> cpu, 1, 1, 1, num_samples) |> device
    ZX_noise = randn(Float32, size_x[1:(end-1)]..., num_samples) |> device;
    X_post = G.inverse(ZX_noise, Y_repeat);
end

# Training hyperparameters
device      = cpu #GPU doesnt really accelerate at this small size. But it is quick on cpu
plot_every  = 1
x_noise = 0.0f0 # Additive noise
lr      = 1f-3
epochs  = 10
batch_size = 256

# Load in training and val data
n_train = 2048
n_val   = batch_size
train_x, _ = MNIST(split=:train)[1:(n_train+n_val)];
train_x    = Float32.(train_x[:,:,1:(n_train)];);
val_x      = Float32.(train_x[:,:,1:(n_val)];);


# Resize to a power of two to make multiscale splitting easier. 
nx = 16; ny = 16;
N = nx*ny;
X_train = zeros(Float32, nx, ny, 1, n_train)
X_val   = zeros(Float32, nx, ny, 1, n_val)
#X_train[:,:,1,:] = train_x
#X_val[:,:,1,:] = val_x
for i in 1:n_train
	X_train[:,:,:,i] = imresize(train_x[:,:,i]', (nx, ny))
end
for i in 1:n_val
	X_val[:,:,:,i] = imresize(val_x[:,:,i]', (nx, ny))
end


# Make Forward operator A
mask_size = 3
mask_start = div((nx-mask_size),2)
A  = ones(Float32,nx,ny)
A[mask_start:(end-mask_start),mask_start:(end-mask_start)] .= 0f0
sigma = 0f0

# Make observations y 
Y_train = A .* X_train #+ sigma*randn(Float32, size(X_train))
Y_val   = A .* X_val   #+ sigma*randn(Float32, size(X_test))

# Architecture parametrs
chan_x = 1
chan_y = 1
L = 3
K = 10
n_hidden = 64

# Create network
G = NetworkConditionalGlow(chan_x, chan_y, n_hidden,  L, K;);
G = G |> device;

# training batches 
batches = cld(n_train, batch_size)

# Optimizer
opt = ADAM(lr)

# Training logs 
l2_train = [];
logdet_train = [];

l2_val = [];
logdet_val = [];

progress = Progress(epochs*batches)

e = 1 
b = 1
for e=1:epochs # epoch loop
	idx_e = reshape(1:n_train, batch_size, batches)
    for b = 1:batches # batch loop
    	X = X_train[:, :, :, idx_e[:,b]];
        X .+= x_noise*randn(Float32, size(X));

        Y = Y_train[:, :, :, idx_e[:,b]];
    
        X = X |> device;
        Y = Y |> device;

        ZX, logdet_i = G.forward(X, Y);

        G.backward(ZX / batch_size, ZX, Y)

        for p in get_params(G) 
        	Flux.update!(opt, p.data, p.grad)
        end
        clear_grad!(G) # clear gradients unless you need to accumulate

        # progress meter
        append!(l2_train, norm(ZX)^2 / (N*batch_size))  # normalize by image size and batch size
        append!(logdet_train, -logdet_i / N) # logdet is internally normalized by batch size

    	next!(progress; showvalues=[
    		(:l2, l2_train[end]),
			(:logdet, logdet_train[end]),
    		(:objective, l2_train[end] + logdet_train[end])])
    end

    # Evaluate network on train and val batch for visualization
    X = X_val
    X .+= x_noise*randn(Float32, size(X))

    X = X |> device
    Y = Y_val |> device

    ZX_val, lgdet_val_i = G.forward(X, Y); #|> cpu;

    append!(l2_val, norm(ZX_val)^2f0 / (N*batch_size));
    append!(logdet_val, -lgdet_val_i / (N))
end

 # Make generative conditional samples
num_plot = 5
fig = figure(figsize=(15, 17)); suptitle("Conditional Glow: epoch = $(e)")
for i in 1:num_plot
	#i = 1
	y = Y_val[:,:,:,i:i]
	x = X_val[:,:,:,i:i] |> cpu;
	X_post = posterior_sampler(G, y, size(x); num_samples=64)
	y = y[:,:,1,1] |> cpu;

	X_post_mean = mean(X_post; dims=ndims(X_post))
	X_post_var = var(X_post;  dims=ndims(X_post))

	ssim_val = round(assess_ssim(X_post_mean[:,:,1,1], x[:,:,1,1]) ,digits=2)

	subplot(num_plot,7,1+7*(i-1)); imshow(x[:,:,1,1],  cmap="gray")
	axis("off"); title(L"$x_{gt} \sim p(x,y)$");

	subplot(num_plot,7,2+7*(i-1)); imshow(y[:,:,1,1],  cmap="gray")
	axis("off"); title(L"$y \sim p(x,y)$"); 

	subplot(num_plot,7,3+7*(i-1)); imshow(X_post_mean[:,:,1,1] |> cpu,  cmap="gray")
	axis("off"); title(L"$\mathrm{E}_{x} p_{\theta}(x | y)$"*"\n ssim="*string(ssim_val)) ; 

	subplot(num_plot,7,4+7*(i-1)); imshow(X_post_var[:,:,1,1]' |> cpu,  cmap="magma")
	axis("off"); title(L"$\mathrm{Var} p_{\theta}(x | y)$") ; 

	subplot(num_plot,7,5+7*(i-1)); imshow(X_post[:,:,1,1]' |> cpu,  cmap="gray")
	axis("off"); title(L"$x\sim p_{\theta}(x | y)$") ; 

	subplot(num_plot,7,6+7*(i-1)); imshow(X_post[:,:,1,2]' |> cpu, cmap="gray")
	axis("off"); title(L"$x\sim p_{\theta}(x | y)$") ; 

	subplot(num_plot,7,7+7*(i-1)); imshow(X_post[:,:,1,3]' |> cpu, cmap="gray")
	axis("off"); title(L"$x\sim p_{\theta}(x | y)$") ; 
end
tight_layout()


############# Training metric logs
sum_train = l2_train + logdet_train
sum_val   = l2_val + logdet_val

fig = figure("training logs ")
title("Total objective: train="*string(round(sum_train[end];digits=3))*" val="*string(round(sum_val[end];digits=3)))
plot(sum_train); 
plot(batches:batches:batches*(epochs), sum_val); 
xlabel("Parameter Update") ;
tight_layout()


