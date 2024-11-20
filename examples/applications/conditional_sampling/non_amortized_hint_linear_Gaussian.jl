# Example for HINT scienario 2 (Kruse et al, 2020)
# Obtaining samples from posterior for the following problem:
# y = Ax + ϵ, x ~ N(μ_x, Σ_x), ϵ ~ N(μ_ϵ, Σ_ϵ), A ~ N(0, I/|x|)

using Pkg
Pkg.add("InvertibleNetworks"); Pkg.add("Flux"); Pkg.add("PyPlot");
Pkg.add("Distributions"); Pkg.add("Printf")

using InvertibleNetworks, LinearAlgebra, Test
using Distributions
import Flux.Optimise.update!, Flux
using PyPlot
using Printf
using Random

Random.seed!(19)

# Model and data dimension
dim_model = 8
dim_data = 6


# Prior distribution
μ_x = 3.14f0*ones(Float32, dim_model)
σ_x = 3.0f0
Σ_x = σ_x^2*I
Λ_x = inv(Σ_x)

π_x = MvNormal(μ_x, Σ_x)
x = rand(π_x)


# Distribution of noise
μ_ϵ = 0.0f0*ones(Float32, dim_data)
σ_ϵ = sqrt(0.1f0)
Σ_ϵ = σ_ϵ^2*I
Λ_ϵ = inv(Σ_ϵ)

π_ϵ = MvNormal(μ_ϵ, Σ_ϵ)
ϵ = rand(π_ϵ)


# Forward operator
A = randn(Float32, dim_data, dim_model)/sqrt(dim_model*1.0f0)


# Observation
y = A*x + ϵ


# Analytic posterior distribution
Σ_post = inv(Λ_x + A'*Λ_ϵ*A)
μ_post = Σ_post*A'*Λ_ϵ*(y - μ_ϵ) + Λ_x*μ_x


# Analytic posterior distribution sampler
R = cholesky(Σ_post, check=false).L
standard_normal = MvNormal(zeros(Float32, dim_model), 1.0f0*I)

function post_dist_sample()
	return R*rand(standard_normal) + μ_post
end


# Invertible neural net
nx = 1
ny = 1
n_in = dim_model
n_hidden = 16
batchsize = 8
depth = 4
AN = Array{ActNorm}(undef, depth)
L = Array{CouplingLayerHINT}(undef, depth)
Params = Array{Parameter}(undef, 0)

# Create layers
for j=1:depth
    AN[j] = ActNorm(n_in; logdet=true)
    L[j] = CouplingLayerHINT(n_in, n_hidden; logdet=true, 
    						 permute="both", k1=1, k2=1, p1=0, p2=0)

    # Collect parameters
    global Params = cat(Params, get_params(AN[j]); dims=1)
    global Params = cat(Params, get_params(L[j]); dims=1)
end

# Forward pass
function forward(X)
    logdet = 0f0
    for j=1:depth
        X_, logdet1 = AN[j].forward(X)
        X, logdet2 = L[j].forward(X_)
        logdet += (logdet1 + logdet2)
    end
    return X, logdet
end

# Backward pass
function backward(ΔX, X)
    for j=depth:-1:1
        ΔX_, X_ = L[j].backward(ΔX, X)
        ΔX, X = AN[j].backward(ΔX_, X_)
    end
    return ΔX, X
end


# Optimizer
η = 0.01
opt = Flux.Optimiser(Flux.ExpDecay(η, .3, 2000, 0.), Flux.ADAM(η))

# Loss function
function loss(z_in, y_in)
	x̂, logdet = forward(z_in)

	f = (1f0/2)*(sum((A*x̂[1, 1, :, :] .- y_in)'*Λ_ϵ*(A*x̂[1, 1, :, :] .- y_in)) + 
		sum((x̂[1, 1, :, :] .- μ_x)'*Λ_x*(x̂[1, 1, :, :] .- μ_x)))/batchsize - logdet

	ΔY = (A'*Λ_ϵ*(A*x̂[1, 1, :, :] .- y_in) + Λ_x*(x̂[1, 1, :, :] .- μ_x))/batchsize
	ΔX = backward(permutedims(repeat(ΔY, 1, 1, 1, 1), [3, 4, 1, 2]), x̂)[1]
	return f, ΔX
end


# Implementation of scenario 2 at HINT paper
N = 5000
z = randn(Float32, 1, 1, dim_model, N)
max_itr = 5000
fval = zeros(Float32, max_itr)

@printf " [*] Beginning training loop\n"
for j = 1:max_itr

	# Evaluate objective and gradients
	idx = sample(1:N, batchsize; replace=false)
	fval[j] = loss(z[:, :, :, idx], y)[1]
	mod(j, 10) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))

	# Update params
	for p in Params
		update!(opt, p.data, p.grad)
        update!(lr_decay_fn, p.data, p.grad)
	end
	clear_grad!(Params)

end


# Validation

x̂ = forward(z)[1][1, 1, :, :]

μ_est = mean(x̂; dims=2)
Σ_est = cov(x̂; dims=2, corrected=true)



# Training loss
fig = plt.figure("training logs - net", dpi=150, figsize=(7, 2.5))
plt.plot(fval); plt.title("training loss")
plt.grid()

# Comparing estimated and true mean
fig = plt.figure("Posterior mean", dpi=150, figsize=(12, 2.5))
plt.plot(μ_est, label="Estimated posterior mean")
plt.plot(μ_post, label="True posterior mean")
plt.legend(loc="upper right")
plt.grid()


# Comparing estimated and true covariance matrix
fig = plt.figure("Posterior covariance", dpi=150, figsize=(8, 4)); 
plt.subplot(121); plt.imshow(Σ_post, vmin=-maximum(Σ_post), vmax=maximum(Σ_post), 
	cmap="RdBu")
plt.title("True posterior covariance")
plt.colorbar(fraction=0.0475, pad=0.03)
plt.subplot(122); plt.imshow(Σ_est, vmin=-maximum(Σ_post), vmax=maximum(Σ_post), 
	cmap="RdBu")
plt.title("Estimated posterior covariance")
plt.colorbar(fraction=0.0475, pad=0.03)


# Sammpling from true posterior
true_samples = zeros(Float32, dim_model, 100)
for j = 1:100
	true_samples[:, j] = post_dist_sample()
end

# Samples from estimated and true posterior
fig = plt.figure("samples from posterior", dpi=150, figsize=(7, 6))
plt.subplot(211); 
for j = 1:100
	plt.plot(true_samples[:, j], linewidth=0.8, alpha=0.5, color="#22c1d6")
plt.grid()
end
plt.title("Samples from true posterior")
plt.subplot(212); 
for j =1:100
	plt.plot(x̂[:, j], linewidth=0.8, alpha=0.5, color="k")
plt.grid()
plt.title("Samples from estimated posterior")
end
