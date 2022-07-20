using PyPlot
using InvertibleNetworks
using Flux
using LinearAlgebra
using Flux.Optimise: Optimiser, update!, ADAM
using MLDatasets
using Random
using Images
using Statistics
using DrWatson

# Reshape function to see latent variable in spatial dimensions when split scales turned on
function z_shape_simple(Zx, Net) 
    XY_save, Zx = split_states(Zx[:], Net.Z_dims)

    for i=Net.L:-1:1
        if i < Net.L
            Zx = tensor_cat(Zx, XY_save[i, 1])
        end
        Zx = Net.squeezer.inverse(Zx) 
    end
    return Zx
end

# Plotting dir
exp_name = "train-noncond-mnist"
save_dict  = @strdict exp_name
save_path = ""

# Training hyperparameters
T = Float32
device = cpu
plot_every  = 1
noise_lev   = 0.01f0 # Additive noise
lr          = 4f-3
n_epochs    = 10
batch_size  = 256

n_train = 2048
n_test  = batch_size

# Load in training and test data
train_x, train_y = MNIST(split=:train)[1:n_train];
train_x = Float32.(train_x);

test_x, test_y =  MNIST(split=:test)[1:n_test];
test_x = Float32.(test_x);

# Subset of digist to train and test on (All of the digits)
digits = [0,1,2,3,4,5,6,7,8,9]

inds = findall(x -> x in digits, train_y);
train_x_digits = train_x[:,:,inds[1:n_train]];

inds_test = findall(x -> x in digits, test_y);
test_x_digits = test_x[:,:,inds_test[1:n_test]];

# Resize to a power of two to make multiscale splitting easier. 
nx = 16; ny = 16
N = nx*ny;
X_train = zeros(Float32, nx, ny, 1, n_train)
X_test = zeros(Float32, nx, ny, 1, n_test)
for i in 1:n_train
	X_train[:,:,:,i] = imresize(train_x_digits[:,:,i]', (nx, ny))
end
for i in 1:n_test
	X_test[:,:,:,i] = imresize(test_x_digits[:,:,i]', (nx, ny))
end

# Testing batches  
X_train_batch = X_train[:,:,:,1:batch_size];
X_test_batch  = X_test[:,:,:,1:batch_size];

X_train_batch .+= noise_lev*randn(Float32, size(X_train_batch));
X_test_batch  .+= noise_lev*randn(Float32, size(X_test_batch));

X_train_batch = X_train_batch |> device;
X_test_batch  = X_test_batch |> device;

# Test generative samples 
gen_sample_size = batch_size
ZX_noise = randn(Float32, nx, ny, 1, gen_sample_size) |> device;

# Architecture parameters
L = 4                # multiscale levels
K = 10				 # coupling layers at each level 
n_hidden = 64	     # hidden channels in coupling layers conv nets

# Create network
G = NetworkGlow(1, n_hidden,  L, K; split_scales=true, activation=SigmoidLayer(low=0.5f0,high=1.0f0));
G = G |> device;

# training indexes 
n_batches = cld(n_train, batch_size)
idx_e = reshape(1:n_train, batch_size, n_batches)

# Optimizer
opt = ADAM(lr)

# Training logs 
loss = [];
logdet = [];

loss_test = [];
logdet_test = [];

for e=1:n_epochs # epoch loop
    for b = 1:n_batches # batch loop
        X = X_train[:, :, :, idx_e[:,b]]
        X .+= noise_lev*randn(T, size(X))
        X = X |> device

        Zx,  lgdet = G.forward(X)

        # Loss function is l2 norm 
        append!(loss, norm(Zx)^2 / (N*batch_size))  # normalize by image size and batch size
        append!(logdet, -lgdet / N) # logdet is internally normalized by batch size

        G.backward((Zx / batch_size), (Zx))

        print("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, 
            "; l2 = ",  loss[end], 
            "; lgdet = ", logdet[end], "; f = ", loss[end] + logdet[end], "\n")

        for p in get_params(G) 
          update!(opt,p.data,p.grad)
        end
        clear_grad!(G)

        Base.flush(Base.stdout)
    end

    # Evaluate network on train and test batch for visualization
    ZX_test, lgdet_test_i = G.forward(X_test_batch) |> cpu;
    ZX_test_sq = z_shape_simple(ZX_test, G);

    append!(logdet_test, -lgdet_test_i / N)
    append!(loss_test, norm(ZX_test)^2f0 / (N*batch_size));

    ZX_train = G.forward(X_train_batch)[1] |> cpu;
    ZX_train_sq = z_shape_simple(ZX_train, G);

    mean_train_1 = round(mean(ZX_train_sq[:,:,1,1]),digits=2)
    std_train_1 = round(std(ZX_train_sq[:,:,1,1]),digits=2)

    mean_test_1 = round(mean(ZX_test_sq[:,:,1,1]),digits=2)
    std_test_1 = round(std(ZX_test_sq[:,:,1,1]),digits=2)

    mean_train_2 = round(mean(ZX_train_sq[:,:,1,2]),digits=2)
    std_train_2 = round(std(ZX_train_sq[:,:,1,2]),digits=2)

    mean_test_2 = round(mean(ZX_test_sq[:,:,1,2]),digits=2)
    std_test_2 = round(std(ZX_test_sq[:,:,1,2]),digits=2)

    # Make generative samples
    X_gen = G.inverse(ZX_noise[:])
  
    if mod(e,plot_every) == 0
	    fig = figure(figsize=(10, 10)); suptitle("MNIST Glow: epoch = $(e) ")

	    subplot(4,4,1); imshow(X_train_batch[:,:,1,1] |> cpu, interpolation="none", cmap="gray")
	    axis("off"); title(L"$x_{train1} \sim p(x)$")

	    subplot(4,4,2); imshow(ZX_train_sq[:,:,1,1],interpolation="none", vmin=-3, vmax=3, cmap="seismic"); axis("off"); 
	    title(L"$z_{train1} = G_{\theta}^{-1}(x_{train1})$ "*string("\n")*" mean "*string(mean_train_1)*" std "*string(std_train_1));

	    subplot(4,4,3); imshow(X_train_batch[:,:,1,2]|> cpu, aspect=1, interpolation="none",  cmap="gray")
	    axis("off"); title(L"$x_{train2} \sim p(x)$")

	    subplot(4,4,4) ;imshow(ZX_train_sq[:,:,1,2],   interpolation="none", 
	                                    vmin=-3, vmax=3, cmap="seismic"); axis("off"); 
	    title(L"$z_{train2} = G_{\theta}^{-1}(x_{train2})$ "*string("\n")*" mean "*string(mean_train_2)*" std "*string(std_train_2));

	    subplot(4,4,5); imshow(X_test_batch[:,:,1,1]|> cpu,  interpolation="none",  cmap="gray")
	    axis("off"); title(L"$x_{test1} \sim p(x)$")

	    subplot(4,4,6) ;imshow(ZX_test_sq[:,:,1,1],  interpolation="none", 
	                                    vmin=-3, vmax=3, cmap="seismic"); axis("off"); 
	    title(L"$z_{test1} = G_{\theta}^{-1}(x_{test1})$ "*string("\n")*" mean "*string(mean_test_1)*" std "*string(std_test_1));
	          
	    subplot(4,4,7); imshow(X_test_batch[:,:,1,2]|> cpu,  interpolation="none", cmap="gray")
	    axis("off"); title(L"$x_{test2} \sim p(x)$")
	      
	    subplot(4,4,8); imshow(ZX_test_sq[:,:,1,2], interpolation="none", vmin=-3, vmax=3, cmap="seismic");
	    axis("off"); title(L"$z_{test2} = G_{\theta}^{-1}(x_{test2})$ "*string("\n")*" mean "*string(mean_test_2)*" std "*string(std_test_2));
	    
	    # plot generative samples
	    for sample_i = 1:8
		    subplot(4,4,8+sample_i); imshow(X_gen[:,:,1,sample_i], aspect=1, vmin=0,vmax=1,interpolation="none",cmap="gray")
		    axis("off");  title(L"$x\sim p_{\theta}(x)$")

		end
	    tight_layout()

	    fig_name = @strdict n_epochs n_train e lr noise_lev n_hidden L K batch_size
	    safesave(joinpath(save_path, savename(fig_name; digits=6)*"mnsit_glow.png"), fig); close(fig)
	    close(fig)

	    ############# Training metric logs
	    sum = loss + logdet
	    sum_test = loss_test + logdet_test

	    fig = figure("training logs ", figsize=(7,10))
	    subplot(3,1,1); title("L2 term: train="*string(loss[end])*" test="*string(loss_test[end]))
	    plot(loss, label="train");
	    plot(n_batches:n_batches:n_batches*(e), loss_test, label="test"); 
	    axhline(y=1,color="red",linestyle="--",label="Normal Noise")
	    xlabel("Parameter Update"); legend();
	    
	    subplot(3,1,2); title("Logdet term: train="*string(logdet[end])*" test="*string(logdet_test[end]))
	    plot(logdet);
	    plot(n_batches:n_batches:n_batches*(e), logdet_test);
	    xlabel("Parameter Update") ;

	    subplot(3,1,3); title("Total objective: train="*string(sum[end])*" test="*string(sum_test[end]))
	    plot(sum); 
	    plot(n_batches:n_batches:n_batches*(e), sum_test); 
	    xlabel("Parameter Update") ;

	    tight_layout()
	    fig_name = @strdict n_epochs n_train e lr noise_lev n_hidden L K  batch_size
	    safesave(joinpath(save_path, savename(fig_name; digits=6)*"mnsit_glow_log.png"), fig); close(fig)
	    close(fig)
	  end

end


