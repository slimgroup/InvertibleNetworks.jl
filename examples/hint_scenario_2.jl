# Example for HINT scienario 2 (Kruse et al, 2020)
# Obtaining samples from posterior for the following problem:
# y = Ax + ϵ, x ~ N(μ_x, σ_x^2), ϵ ~ N(μ_ϵ, σ_ϵ^2), A ~ N(0, I/|x|^2)

using InvertibleNetworks, LinearAlgebra, Test
using Distributions
import Flux.Optimise.update!, Flux
using PyPlot
using Printf
using Random
Random.seed!(19)

##### Model and data dimension #####
dim_model = 8
dim_data = 6


##### Prior distribution #####
μ_x = 3.14f0
σ_x = 3.0f0

prior_dist = MvNormal(μ_x*ones(dim_model), σ_x*ones(dim_model))
x = convert(Array{Float32}, rand(prior_dist))


##### Distribution of noise #####
μ_ϵ = 0.0f0
σ_ϵ = convert(Float32, sqrt(0.1))

ϵ_dist = MvNormal(μ_ϵ*ones(dim_data), σ_ϵ*ones(dim_data))
ϵ = convert(Array{Float32}, rand(ϵ_dist))


##### Forward operator #####
A = convert(Array{Float32}, randn(dim_data, dim_model)/sqrt(dim_model))


##### Observation #####
y = A*x + ϵ


##### Analytic posterior distribution #####
Σ2_post = inv(I/σ_x^2 + transpose(A)*(I/σ_ϵ^2)*A)
μ_post = convert(Array{Float32}, Σ2_post*transpose(A)*(I/σ_ϵ^2)*y + 
	(I/σ_x^2)*μ_x*ones(dim_model))

R = cholesky(Σ2_post, check=false).L
standard_normal = MvNormal(zeros(dim_model), ones(dim_model))

function post_dist_sample()
	return R*convert(Array{Float32}, rand(standard_normal)) + μ_post
end

##### Invertible neural net #####

depth = 4
AN = Array{ActNorm}(undef, depth)
L = Array{CouplingLayerHINT}(undef, depth)
Params = Array{Parameter}(undef, 0)

# Create layers
for j=1:depth
	AN[j] = ActNorm(dim_model; logdet=true)
	L[j] = CouplingLayerHINT(1, 1, dim_model, 32, 32; logdet=true, permute="lower", 
		k1=1, k2=1, p1=0, p2=0)

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


##### Optimizer #####
η = 0.001
opt = Flux.ADAM(η)

##### Loss function #####
function loss(z_in, y_in)
	x̂, logdet = forward(z_in)
	f = 1.0f0/(2.0f0*σ_ϵ^2) * sum(A*x̂[1, 1, :, :] .- y_in)^2/(1.0f0*batch_size) + 
		1.0f0/(2.0f0*σ_x^2) * sum(x̂ .- μ_x)^2/(1.0f0*batch_size) - logdet

	ΔY = 1.0f0/(σ_ϵ^2) * transpose(A)*(A*x̂[1, 1, :, :] .- y_in)/(1.0f0*batch_size) + 
		1.0f0/(σ_x^2) * (x̂[1, 1, :, :] .- μ_x)/(1.0f0*batch_size)
	ΔX = backward(permutedims(repeat(ΔY, 1, 1, 1, 1), [3, 4, 1, 2]), x̂)[1]
	return f, ΔX
end


##### Implementation of scenario 2 at HINT paper #####
N = 1000
batch_size = 1000
z = randn(Float32, 1, 1, dim_model, N)
max_itr = 1000
fval = zeros(Float32, max_itr)

@printf " [*] Beginning training loop\n"
for j = 1:max_itr

	# Evaluate objective and gradients
	idx = sample(1:N, batch_size; replace=false)
	fval[j] = loss(z[:, :, :, idx], y)[1]
	mod(j, 10) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))

	# Update params
	for p in Params
		update!(opt, p.data, p.grad)
	end
	clear_grad!(Params)
end


##### Validation #####

x̂ = forward(z)[1][1, 1, :, :]

μ_est = mean(x̂; dims=2)
Σ2_est = cov(x̂; dims=2, corrected=true)


# Training loss
fig = figure("training logs - net", dpi=150, figsize=(7, 2.5))
plot(fval); title("training loss")
grid()

# Comparing estimated and true mean
figure("Posterior mean", dpi=150, figsize=(12, 2.5))
plot(μ_est, label="Estimated posterior mean")
plot(μ_post, label="True posterior mean")
legend(loc="upper right")
grid()

# Comparing estimated and true covariance matrix
figure("Posterior covariance", dpi=150, figsize=(8, 4)); 
subplot(121); imshow(Σ2_post, vmin=-maximum(Σ2_post), vmax=maximum(Σ2_post), 
	cmap="RdBu")
title("True posterior covariance")
colorbar(fraction=0.0475, pad=0.03)
subplot(122); imshow(Σ2_est, vmin=-maximum(Σ2_post), vmax=maximum(Σ2_post), 
	cmap="RdBu")
title("Estimated posterior covariance")
colorbar(fraction=0.0475, pad=0.03)

# Sammpling from true posterior
true_samples = zeros(Float32, dim_model, max_itr)
for j = 1:N
	true_samples[:, j] = post_dist_sample()
end

# Samples from estimated and true posterior
figure("samples from posterior", dpi=150, figsize=(7, 6))
subplot(211); 
for j = 1:100
	plot(true_samples[:, j], alpha=0.5)
grid()
end
title("Samples from true posterior")
subplot(212); 
for j =1:100
	plot(x̂[:, j], alpha=0.5)
grid()
title("Samples from estimated posterior")
end
