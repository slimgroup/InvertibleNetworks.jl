# Take around 3 minutes on CPU
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
device = cpu #GPU does not accelerate at this small size. quicker on cpu
lr     = 4f-3
epochs = 15
batch_size = 128

# Load in training and val data
n_train = 2048
n_val   = Int(div(2048,10))
xs, _ = MNIST(split=:train)[1:(n_train+n_val)];
train_x = Float32.(xs[:,:,1:n_train];);
val_x   = Float32.(xs[:,:,n_train+1:end];);


# Resize to a power of two to make multiscale splitting easier. 
nx = 16; ny = 16;
N = nx*ny;
X_train = zeros(Float32, nx, ny, 1, n_train)
X_val   = zeros(Float32, nx, ny, 1, n_val)
for i in 1:n_train
	X_train[:,:,:,i] = imresize(train_x[:,:,i]', (nx, ny))
end
for i in 1:n_val
	X_val[:,:,:,i] = imresize(val_x[:,:,i]', (nx, ny))
end

# Make Forward operator A
mask_size = 3 #number of pixels to zero out
mask_start = div((nx-mask_size),2)
A  = ones(Float32,nx,ny)
A[mask_start:(end-mask_start),mask_start:(end-mask_start)] .= 0f0

# Make observations y 
Y_train = A .* X_train 
Y_val   = A .* X_val   

# Architecture parametrs
chan_x = size(X_train)[end-1]
chan_y = size(Y_train)[end-1]
L = 2
K = 10
n_hidden = 32

# Create network
G = NetworkConditionalGlow(chan_x, chan_y, n_hidden,  L, K;) |> device;

# Number of training batches 
batches = cld(n_train, batch_size)

# Optimizer
opt = ADAM(lr)

# Training logs 
loss_train = [];
loss_val = [];
progress = Progress(epochs*batches);

for e=1:epochs # epoch loop
	idx_e = reshape(1:n_train, batch_size, batches)
    for b = 1:batches # batch loop
    	X = X_train[:, :, :, idx_e[:,b]] |> device;
        Y = Y_train[:, :, :, idx_e[:,b]] |> device;

        ZX, logdet_i = G.forward(X, Y);

        G.backward(ZX / batch_size, ZX, Y)

        for p in get_params(G) 
        	Flux.update!(opt, p.data, p.grad)
        end
        clear_grad!(G) # clear gradients unless you need to accumulate

        # progress meter
        append!(loss_train, norm(ZX)^2 / (N*batch_size) - logdet_i / N)  # normalize by image size and batch size
     
    	next!(progress; showvalues=[
    		(:objective, loss_train[end])])
    end

    # Evaluate network validation batch 
    X = X_val |> device
    Y = Y_val |> device

    ZX, logdet_i = G.forward(X, Y); 

    append!(loss_val, norm(ZX)^2 / (N*batch_size) - logdet_i / N)  # normalize by image size and batch size
end


#Make generative conditional samples
num_plot = 5
fig = figure(figsize=(15, 17)); suptitle("Conditional Glow: epoch = $(epochs)")
for i in 1:num_plot
	y = Y_val[:,:,:,i:i]
	x = X_val[:,:,:,i:i] |> cpu;
	X_post = posterior_sampler(G, y, size(x); num_samples=64) 

	X_post_mean = mean(X_post; dims=ndims(X_post)) |> cpu
	X_post_var  = var(X_post;  dims=ndims(X_post)) |> cpu

	ssim_val = round(assess_ssim(X_post_mean[:,:,1,1], x[:,:,1,1]) ,digits=2)

	subplot(num_plot,7,1+7*(i-1)); imshow(x[:,:,1,1],  cmap="gray")
	axis("off"); title(L"$x_{gt} \sim p(x,y)$");

	subplot(num_plot,7,2+7*(i-1)); imshow(y[:,:,1,1] |> cpu,  cmap="gray")
	axis("off"); title(L"$y \sim p(x,y)$"); 

	subplot(num_plot,7,3+7*(i-1)); imshow(X_post_mean[:,:,1,1] ,  cmap="gray")
	axis("off"); title(L"$\mathrm{E} \, p_{\theta}(x | y)$"*"\n ssim="*string(ssim_val)) ; 

	subplot(num_plot,7,4+7*(i-1)); imshow(X_post_var[:,:,1,1] ,  cmap="magma")
	axis("off"); title(L"$\mathrm{Var} \, p_{\theta}(x | y)$") ; 

	subplot(num_plot,7,5+7*(i-1)); imshow(X_post[:,:,1,1] |> cpu,  cmap="gray")
	axis("off"); title(L"$x\sim p_{\theta}(x | y)$") ; 

	subplot(num_plot,7,6+7*(i-1)); imshow(X_post[:,:,1,2] |> cpu, cmap="gray")
	axis("off"); title(L"$x\sim p_{\theta}(x | y)$") ; 

	subplot(num_plot,7,7+7*(i-1)); imshow(X_post[:,:,1,3] |> cpu, cmap="gray")
	axis("off"); title(L"$x\sim p_{\theta}(x | y)$") ; 
end
tight_layout()

############# Training metric logs
final_obj_train = round(loss_train[end];digits=3)
final_obj_val = round(loss_val[end];digits=3)

fig = figure()
title("Total objective: train=$(final_obj_train) validation=$(final_obj_val)")
plot(loss_train); 
plot(batches:batches:batches*(epochs), loss_val); 
xlabel("Parameter Update") ;


