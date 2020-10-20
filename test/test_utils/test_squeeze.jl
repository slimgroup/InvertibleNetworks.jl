
using InvertibleNetworks, Test, LinearAlgebra, Random

Random.seed!(11)

##############################################################################################################################
# 4D Tensor
X = randn(Float32, 28, 28, 2, 4)

# Squeeze and unsqueeze
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="column"); pattern="column")), 0f0; atol=1f-6)
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="patch"); pattern="patch")), 0f0; atol=1f-6)
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="checkerboard"); pattern="checkerboard")), 0f0; atol=1f-6)

# Wavelet transform invertibility
Y = wavelet_squeeze(X)
X_ = wavelet_unsqueeze(Y)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

# Wavelet transform adjoint test
X = randn(Float32, 28, 28, 2, 4)
Y = randn(Float32, 14, 14, 8, 4)
a = dot(Y, wavelet_squeeze(X))
b = dot(X, wavelet_unsqueeze(Y))
@test isapprox(a/b - 1f0, 0f0; atol=1f-5)

# Split and cat
@test isapprox(norm(X - tensor_cat(tensor_split(X))), 0f0; atol=1f-5)


##############################################################################################################################
# 5D Tensor
X = randn(Float32, 16, 16, 16, 2, 4)

# Squeeze and unsqueeze
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="column"); pattern="column")), 0f0; atol=1f-6)
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="patch"); pattern="patch")), 0f0; atol=1f-6)

# Wavelet transform invertibility
Y = wavelet_squeeze(X)
X_ = wavelet_unsqueeze(Y)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)

# Wavelet transform adjoint test
X = randn(Float32, 16, 16, 16, 2, 4)
Y = randn(Float32, 8, 8, 8, 16, 4)
a = dot(Y, wavelet_squeeze(X))
b = dot(X, wavelet_unsqueeze(Y))
@test isapprox(a/b - 1f0, 0f0; atol=1f-5)

# Split and cat
@test isapprox(norm(X - tensor_cat(tensor_split(X))), 0f0; atol=1f-5)
