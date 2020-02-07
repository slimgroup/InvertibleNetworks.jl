
using InvertibleNetworks, Test, LinearAlgebra

X = randn(Float32, 24, 24, 2, 10)

# Squeeze and unsqueeze
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="column"); pattern="column")), 0f0; atol=1f-6)
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="patch"); pattern="patch")), 0f0; atol=1f-6)
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="checkerboard"); pattern="checkerboard")), 0f0; atol=1f-6)

# Wavelet squeeze and unsqueeze
Y = wavelet_squeeze(X)
X_ = wavelet_unsqueeze(Y)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-6)

# Split and cat
@test isapprox(norm(X - tensor_cat(tensor_split(X))), 0f0; atol=1f-6)
