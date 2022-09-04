# Take around 3 minutes on CPU
using InvertibleNetworks
using Flux
using LinearAlgebra
using MLDatasets
using PyPlot
using ProgressMeter: Progress, next!
using Images
using MLUtils
using DrWatson
using Random 
using Suppressor
using Revise

# [x] take off clipnorm 
# [x] batch_size
# [x] proper channel split
# [] do their normalization
# [x] right activation function
# [] learned scaling parameter
# [] zero residual layer last not small
# [] start from 28 size 

matplotlib.pyplot.switch_backend("Agg")

# Training hyperparameters
device = cpu #GPU does not accelerate at this small size. quicker on cpu
#lr     = 6f-3
lr     = 1f-3
noise_lev = 0.00f0
clip_norm = 0f0
epochs = 1
batch_size = 8
#4 137.064012 seconds 
last_act = false
affine = true
zeros_w = true
mask = true
# Load in training data (need more for non-conditional sampling)
n_total = batch_size*4 #512
#n_total = 2048
validation_perc = 0.9
X  = MNIST.traintensor(Float32, 1:n_total) ;

# Resize spatial size to a power of two to make Real-NVP multiscale easier. 
nx = 16; ny = 16;
Xs = zeros(Float32, nx, ny, 1, n_total);
for i in 1:n_total
    Xs[:,:,:,i] = imresize(X[:,:,i]', (nx, ny));
end
#Xs = reshape(X,size(X)[1:end-1]...,1,size(X)[end])

# Use MLutils to split into training and validation set
Random.seed!(123);
X_train, X_val = splitobs(Xs; at=validation_perc, shuffle=true);
train_loader   = DataLoader(X_train, batchsize=batch_size, shuffle=true, partial=false);

Random.seed!(123);
ZX_noise = randn(Float32, nx, ny, 1, 16) |> device;

# Number of training batches 
n_train  = numobs(X_train)
n_val    = numobs(X_val)
batches  = cld(n_train, batch_size)
progress = Progress(epochs*batches);

# Architecture parametrs
chan_x   = 1  # not RGB so chan=1
L        = 3  # Number of multiscale levels
K        = 32 # Number of GLOW layers per multiscale level
n_hidden = 64 # Number of hidden channels in convolutional residual blocks

function SigmoidLayer_glow(;low=0f0, high=1f0)
    fwd_a(x) = Sigmoid(x .+ 2f0; low=low, high=high) .+ 1f-3
    inv_a(y) = SigmoidInv(y .- 1f-3 ; low=low, high=high) .- 2f0
    grad_a(Δy, y; x=nothing) = SigmoidGrad(Δy, y; x=x, low=low, high=high)
    return ActivationFunction(fwd_a, inv_a, grad_a)
end

activation = SigmoidLayer(low=0.5f0)
act = "invnets"
# Create network
Random.seed!(123);
G = NetworkGlow(chan_x, n_hidden,  L, K;activation=activation, freeze_conv=false, affine=affine, split_scales=true) |> device;

X = X_train[:,:,:,1:1]
y, _ = G.forward(X)
X_ = G.inverse(y)
norm(X_ - X)


X = G.squeezer.forward(X)

j=1
T= Float32
num_chan = size(X)[end-1]
mask = ones(T,num_chan)
if mod(j,2) == 0
    mask[1:2:end] = -1f0 .* ones(T,length(mask[1:2:end]))
else
    mask[2:2:end] = -1f0 .* ones(T,length(mask[2:2:end]))
end 

X_1,X_2 = tensor_split(X;mask=mask)
X_ = tensor_cat(X_1,X_2;mask=mask)
#norm(X_ - X)

#x = randn(8,8,2,1)
#x_ = G.CL[1].RB.forward(x)
# Optimizer
#opt = Flux.Optimiser(ClipNorm(clip_norm),ADAM(lr))
opt = ADAM(lr)

# Training logs 
loss_train = [];
loss_val   = [];

@time begin
for e=1:epochs # epoch loop
    for (X) in train_loader #batch loop
        X .+= noise_lev*randn(Float32, size(X))
        X = X |> device;

        ZX, logdet_i = @suppress G.forward(X)
        #ZX, logdet_i =  G.forward(X)

        @suppress G.backward(ZX / batch_size, ZX)

        for p in get_params(G) 
            Flux.update!(opt, p.data, p.grad)
        end
        clear_grad!(G) # clear gradients unless you need to accumulate

        #Progress meter
        N = prod(size(X)[1:end-1])
        append!(loss_train, norm(ZX)^2 / (N*batch_size) - logdet_i / N)  # normalize by image size and batch size
        #println( norm(ZX)^2 / (N*batch_size) - logdet_i / N)
        next!(progress; showvalues=[
            (:objective, loss_train[end])])
    end

    # Evaluate network on validation set 
    X = getobs(X_val) 
    #ZX, logdet_i = G.forward(X); 
    X .+= noise_lev*randn(Float32, size(X))
    X = X |> device
    ZX, logdet_i = @suppress G.forward(X)

    N = prod(size(X)[1:end-1])
    append!(loss_val, norm(ZX)^2 / (N*n_val) - logdet_i / N)  # normalize by image size and batch size
  
    # Make and plot generative samples
    X_gen = @suppress G.inverse(ZX_noise);

    fig = figure()
    subplot(3,4,1); imshow(X_train[:,:,1,1],vmin=0, vmax=1)
    axis("off"); title("training sample");

    subplot(3,4,2); imshow(X_train[:,:,1,2],vmin=0, vmax=1)
    axis("off"); title("training sample");

    subplot(3,4,3); imshow(X_train[:,:,1,3],vmin=0, vmax=1)
    axis("off"); title("training sample");

    subplot(3,4,4); imshow(X_train[:,:,1,4],vmin=0, vmax=1)
    axis("off"); title("training sample");

    subplot(3,4,5); imshow(X_gen[:,:,1,1],vmin=0, vmax=1)
    axis("off"); title("generative sample");

    subplot(3,4,6); imshow(X_gen[:,:,1,2],vmin=0, vmax=1)
    axis("off"); title("generative sample");

    subplot(3,4,7); imshow(X_gen[:,:,1,3],vmin=0, vmax=1)
    axis("off"); title("generative sample");

    subplot(3,4,8); imshow(X_gen[:,:,1,4],vmin=0, vmax=1)
    axis("off"); title("generative sample");

    subplot(3,4,9); imshow(X_gen[:,:,1,5],vmin=0, vmax=1)
    axis("off"); title("generative sample");

    subplot(3,4,10); imshow(X_gen[:,:,1,6],vmin=0, vmax=1)
    axis("off"); title("generative sample");

    subplot(3,4,11); imshow(X_gen[:,:,1,7],vmin=0, vmax=1)
    axis("off"); title("generative sample");

    subplot(3,4,12); imshow(X_gen[:,:,1,8],vmin=0, vmax=1)
    axis("off"); title("generative sample");

    tight_layout()
    fig_name = @strdict mask zeros_w act last_act affine e epochs n_train clip_norm lr noise_lev n_hidden L K batch_size
    safesave(joinpath("plots",savename(fig_name; digits=6)*"_gen.png"), fig); close(fig)

    # Training logs
    final_obj_train = round(loss_train[end];digits=3)
    final_obj_val = round(loss_val[end];digits=3)

    fig = figure()
    title("Objective value: train=$(final_obj_train) validation=$(final_obj_val)")
    plot(loss_train;label="Train"); 
    plot(batches:batches:batches*(e), loss_val;label="Validation"); 
    xlabel("Parameter update"); ylabel("Negative log likelihood objective") ;
    legend()
    ylim(bottom=-4.0)

    fig_name = @strdict mask zeros_w act last_act affine e epochs n_train clip_norm lr noise_lev n_hidden L K batch_size
    safesave(joinpath("plots",savename(fig_name; digits=6)*"_log.png"), fig); close(fig)
end
end


