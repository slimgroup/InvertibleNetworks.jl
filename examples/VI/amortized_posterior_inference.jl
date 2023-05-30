using Distributions

function sample_prior()
    μ = rand(Normal(0, 1))
    σ = rand(Uniform(0, 10))
    return [μ,σ]
end

function generate_data(num_obs)
    θ = sample_prior()
    y = rand(Normal(θ...), num_obs)
    return [θ...,y...] 
end

# train using samples from joint distribution x,y ~ p(x,y) where x=[μ, σ] -> y = N(μ, σ)
# rows: μ, σ, y
num_params = 2; num_obs = 51;
n_c_params = 1; n_c_obs = 1;
n_train = 10000
training_data = mapreduce(x -> generate_data(num_obs), hcat, 1:n_train);

############ train some data
X_train = reshape(training_data[1:num_params,:], (num_params,n_c_params,:));
Y_train = reshape(training_data[(num_params+1):end,:], (num_obs,n_c_obs,:));

n_epochs   = 2
batch_size = 200
n_batches = div(n_train,batch_size)

# make conditional normalizing flow
using InvertibleNetworks, LinearAlgebra, Flux

L = 3 # RealNVP multiscale layers
K = 4 # Coupling layers per scale
n_hidden = 32 # Hidden channels in coupling layers' neural network
G = NetworkConditionalGlow(n_c_params, n_c_obs, n_hidden,  L, K; nx=nx, ndims=1);
opt = ADAM(4f-3)

# Training logs 
loss_l2   = []; logdet_train = [];

for e=1:n_epochs # epoch loop
    idx_e = reshape(1:n_train, batch_size, n_batches) 
    for b = 1:n_batches # batch loop
        X = X_train[:, :,  idx_e[:,b]];
        Y = Y_train[:, :,  idx_e[:,b]];
     
        # Forward pass of normalizing flow
        Zx, Zy, lgdet = G.forward(X, Y)

        # Loss function is l2 norm - logdet
        append!(loss_l2, norm(Zx)^2 / prod(size(X)))  # normalize by image size and batch size
        append!(logdet_train, -lgdet / prod(size(X)[1:end-1])) # logdet is already normalized by batch size

        # Set gradients of flow
        G.backward(Zx / batch_size, Zx, Zy)

        # Update parameters of flow
        for p in get_params(G) 
          Flux.update!(opt,p.data,p.grad)
        end; 
        clear_grad!(G)

        print("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, "; f l2 = ",  loss_l2[end], 
            "; lgdet = ", logdet_train[end], "; full objective = ", loss_l2[end] + logdet_train[end], "\n")
    end
end

# posterior inference on unseen observation
x_ground_truth = [1,1] # mu=1, sigma=1
observed_data  =  reshape(rand(Normal(x_ground_truth[1], x_ground_truth[2]), num_obs), (1,1,num_obs,:))

# posterior sampling with conditional normalizing flow
num_post_samps = 1000
ZX_noise = randn(1,1,num_params,num_post_samps) 
Y_forward = repeat(observed_data,  1, 1, 1, num_post_samps) 

_, Zy_fixed_train, _ = G.forward(ZX_noise, Y_forward); #needed to set the proper transforms on inverse

X_post = G.inverse(ZX_noise, Zy_fixed_train );

# plot results
using PyPlot

X_prior = mapreduce(x -> sample_prior(), hcat, 1:num_post_samps)
X_prior = reshape(X_prior[1:num_params,:], (1,1,num_params,:))

fig = figure()
subplot(1,2,1)
hist(X_prior[1,1,1,:];alpha=0.7,density=true,label="Prior")
hist(X_post[1,1,1,:];alpha=0.7,density=true,label="Posterior")
axvline(x_ground_truth[1], color="k", linewidth=1,label="Ground truth")
xlabel(L"\mu"); ylabel("Density"); 
legend()

subplot(1,2,2)
hist(X_prior[1,1,2,:]; alpha=0.7,density=true,label="Prior")
hist(X_post[1,1,2,:]; alpha=0.7,density=true,label="Posterior")
axvline(x_ground_truth[2], color="k", linewidth=1,label="Ground truth")
xlabel(L"\sigma"); ylabel("Density");
legend()
tight_layout()


# Look at training curve
fig = figure("training logs ", figsize=(10,12))
subplot(3,1,1); title("L2 Term")
plot(loss_l2, label="train"); 
axhline(y=1,color="red",linestyle="--",label="Standard Normal")
xlabel("Parameter Update"); legend();

subplot(3,1,2); title("Logdet Term")
plot(logdet_train);
xlabel("Parameter Update") ;

subplot(3,1,3); title("Total Objective")
plot(loss_l2 + logdet_train); 
xlabel("Parameter Update") ;
tight_layout()

