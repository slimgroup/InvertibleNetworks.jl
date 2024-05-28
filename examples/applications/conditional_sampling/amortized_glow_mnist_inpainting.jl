# Take around 6 minutes on CPU
using Pkg

Pkg.add("InvertibleNetworks")
Pkg.add("Flux")
Pkg.add("MLDatasets")
Pkg.add("ProgressMeter")
Pkg.add("MLUtils")
Pkg.add("ImageTransformations")

using InvertibleNetworks
using Flux
using LinearAlgebra
using MLDatasets
using ImageTransformations
using MLUtils
using ProgressMeter: Progress, next!

function posterior_sampler(G, y, size_x; device=gpu, num_samples=1, batch_size=16)
    # make samples from posterior for train sample 
    X_dummy = randn(Float32, size_x[1:end-1]...,batch_size) |> device
    Y_repeat = repeat(y |>cpu, 1, 1, 1, batch_size) |> device
    _, Zy_fixed, _ = G.forward(X_dummy, Y_repeat); #needs to set the proper sizes here

    X_post = zeros(Float32, size_x[1:end-1]...,num_samples)
    for i in 1:div(num_samples, batch_size)
      Zx_noise_i = randn(Float32, size_x[1:end-1]...,batch_size)|> device
      X_post[:,:,:, (i-1)*batch_size+1 : i*batch_size] = G.inverse(
          Zx_noise_i,
          Zy_fixed
        ) |> cpu;
  end
  X_post
end

# Training hyperparameters
device = cpu #GPU does not accelerate at this small size. quicker on cpu
lr     = 2f-3
epochs = 30
batch_size = 128

# Load in training data
n_total = 2048
validation_perc = 0.9
X, _ = MNIST(split=:train)[1:(n_total)];

# Resize spatial size to a power of two to make Real-NVP multiscale easier. 
nx = 16; ny = 16;
N = nx*ny
Xs = zeros(Float32, nx, ny, 1, n_total);
for i in 1:n_total
	Xs[:,:,:,i] = imresize(X[:,:,i]', (nx, ny));
end

# Make Forward operator A
mask_size = 3 #number of pixels to zero out
mask_start = div((nx-mask_size),2)
A  = ones(Float32,nx,ny)
A[mask_start:(end-mask_start),mask_start:(end-mask_start)] .= 0f0

# Make observations y 
Ys = A .* Xs;

# To modify for your aplpication load in your own Ys and Xs here.
# julia> size(Ys)
# (16, 16, 1, 2048) (nx,ny,n_chan,n_batch)

# julia> size(Xs)
# (16, 16, 1, 2048) (nx,ny,n_chan,n_batch)

# Use MLutils to split into training and validation/test set
XY_train, XY_val = splitobs((Xs, Ys); at=validation_perc, shuffle=true);
train_loader = DataLoader(XY_train, batchsize=batch_size, shuffle=true, partial=false);

# Number of training batches 
n_train  = numobs(XY_train)
n_val    = numobs(XY_val)
batches  = cld(n_train, batch_size)
progress = Progress(epochs*batches);

# Architecture parametrs
chan_x = 1    # not RGB so chan=1
chan_y = 1    # not RGB so chan=1
L = 2         # Number of multiscale levels
K = 10        # Number of Real-NVP layers per multiscale level
n_hidden = 32 # Number of hidden channels in convolutional residual blocks

# Create network
G = NetworkConditionalGlow(chan_x, chan_y, n_hidden,  L, K; split_scales=true ) |> device;

# Optimizer
opt = ADAM(lr)

# Training logs 
loss_train = []; loss_val = [];

for e=1:epochs # epoch loop
    for (X, Y) in train_loader #batch loop
        ZX, ZY, logdet_i = G.forward(X|> device, Y|> device);
        G.backward(ZX / batch_size, ZX, ZY)

        for p in get_params(G) 
        	Flux.update!(opt, p.data, p.grad)
        end; clear_grad!(G) # clear gradients unless you need to accumulate

        #Progress meter
        append!(loss_train, norm(ZX)^2 / (N*batch_size) - logdet_i / N)  # normalize by image size and batch size
    	next!(progress; showvalues=[(:objective, loss_train[end]),(:l2norm, norm(ZX)^2 / (N*batch_size))])
    end
    # Evaluate network on validation set 
    X = getobs(XY_val[1]) |> device;
    Y = getobs(XY_val[2]) |> device;

    ZX, ZY,logdet_i = G.forward(X, Y); 
    append!(loss_val, norm(ZX)^2 / (N*n_val) - logdet_i / N)  # normalize by image size and batch size
end


Pkg.add("Statistics")
Pkg.add("Plots")
using Statistics
using Plots

# Training logs
plot(loss_train;label="Train"); 
plot(batches:batches:batches*(epochs), loss_val;label="Validation"); 
xlabel!("Parameter update"); ylabel!("Negative log likelihood objective") ;
savefig("log.png")

# Make Figure like README
ind = 1
x = XY_val[1][:,:,:,ind:ind]
y = XY_val[2][:,:,:,ind:ind]
X_post = posterior_sampler(G, y, size(x); device=device, num_samples=64) |> cpu

X_post_mean = mean(X_post; dims=ndims(X_post)) 
X_post_var  = var(X_post;  dims=ndims(X_post)) 

p1 = heatmap(x[:,:,1,1];title="Ground truth",cbar=false,clims = (0, 1),axis=([], false))
p2 = heatmap(y[:,:,1,1];title="Observation",cbar=false,clims = (0, 1),axis=([], false))
p3 = heatmap(X_post_mean[:,:,1,1];title="Posterior mean",cbar=false,clims = (0, 1),axis=([], false))
p4 = heatmap(X_post_var[:,:,1,1];title="Posterior variance",cbar=false,axis=([], false))
p5 = heatmap(X_post[:,:,1,1];title="Posterior sample",cbar=false,clims = (0, 1),axis=([], false))
p6 = heatmap(X_post[:,:,1,2];title="Posterior sample",cbar=false,clims = (0, 1),axis=([], false))

plot(p1,p2,p3,p4,p5,p6,aspect_ratio=:equal, size=(600,400))
savefig("posterior_sampling.png")


