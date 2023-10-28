using InvertibleNetworks, Test, LinearAlgebra, Random

Random.seed!(11)

##############################################################################################################################
# 3D Tensor
X = randn(Float32, 28, 2, 4)

# Squeeze and unsqueeze
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="column"); pattern="column")), 0f0; atol=1f-6)
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="checkerboard"); pattern="checkerboard")), 0f0; atol=1f-6)

#test norm preservation of orthogonal transforms
@test norm(X) ≈ norm(Haar_squeeze(X))

# Wavelet transform invertibility
@test X ≈ invHaar_unsqueeze(Haar_squeeze(X))

# Wavelet transform adjoint test
X = randn(Float32, 28, 2, 4)
Y = randn(Float32, 14, 4, 4)
a = dot(Y, Haar_squeeze(X))
b = dot(X, invHaar_unsqueeze(Y))
@test isapprox(a/b - 1f0, 0f0; atol=1f-5)

# Split and cat
@test isapprox(norm(X - tensor_cat(tensor_split(X))), 0f0; atol=1f-5)

# Unsqueeze works for any size as long as channel size divisible by 2 (for 3D Tensor)
x_size  = 27
ch_size = 8
batch_size = 4
X = randn(Float32, x_size, ch_size, batch_size)
@test isequal(size(unsqueeze(X)), (x_size*2, div(ch_size,2), batch_size))

##############################################################################################################################
# 4D Tensor
X = randn(Float32, 28, 28, 2, 4)

# Squeeze and unsqueeze
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="column"); pattern="column")), 0f0; atol=1f-6)
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="patch"); pattern="patch")), 0f0; atol=1f-6)
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="checkerboard"); pattern="checkerboard")), 0f0; atol=1f-6)

#test norm preservation of orthogonal transforms
@test norm(X) ≈ norm(Haar_squeeze(X))

# Wavelet transform invertibility
Y = wavelet_squeeze(X)
X_ = wavelet_unsqueeze(Y)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)
@test X ≈ invHaar_unsqueeze(Haar_squeeze(X))

# Wavelet transform adjoint test
X = randn(Float32, 28, 28, 2, 4)
Y = randn(Float32, 14, 14, 8, 4)
a = dot(Y, wavelet_squeeze(X))
b = dot(X, wavelet_unsqueeze(Y))
@test isapprox(a/b - 1f0, 0f0; atol=1f-5)
a = dot(Y, Haar_squeeze(X))
b = dot(X, invHaar_unsqueeze(Y))
@test isapprox(a/b - 1f0, 0f0; atol=1f-5)

# Split and cat
@test isapprox(norm(X - tensor_cat(tensor_split(X))), 0f0; atol=1f-5)

# Unsqueeze works for any size as long as channel size divisible by 4 (for 4D Tensor)
x_size  = 27
y_size  = 27
ch_size = 8
batch_size = 4
X = randn(Float32, x_size, y_size, ch_size, batch_size)
@test isequal(size(unsqueeze(X)), (x_size*2, y_size*2, div(ch_size,4), batch_size))


##############################################################################################################################
# 5D Tensor
X = randn(Float32, 16, 16, 16, 2, 4)

# Squeeze and unsqueeze
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="column"); pattern="column")), 0f0; atol=1f-6)
@test isapprox(norm(X - unsqueeze(squeeze(X; pattern="patch"); pattern="patch")), 0f0; atol=1f-6)

#test norm preservation of orthogonal transforms
@test norm(X) ≈ norm(Haar_squeeze(X))

# Wavelet transform invertibility
Y = wavelet_squeeze(X)
X_ = wavelet_unsqueeze(Y)
@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-5)
@test X ≈ invHaar_unsqueeze(Haar_squeeze(X))

# Wavelet transform adjoint test
X = randn(Float32, 16, 16, 16, 2, 4)
Y = randn(Float32, 8, 8, 8, 16, 4)
a = dot(Y, wavelet_squeeze(X))
b = dot(X, wavelet_unsqueeze(Y))
@test isapprox(a/b - 1f0, 0f0; atol=1f-5)
a = dot(Y, Haar_squeeze(X))
b = dot(X, invHaar_unsqueeze(Y))
@test isapprox(a/b - 1f0, 0f0; atol=1f-5)

# Split and cat
@test isapprox(norm(X - tensor_cat(tensor_split(X))), 0f0; atol=1f-5)

# Unsqueeze works for any size as long as channel size divisible by 8 (for 5D Tensor)
x_size  = 27
y_size  = 27
z_size  = 27
ch_size = 8
batch_size = 4
X = randn(Float32, x_size, y_size, z_size, ch_size, batch_size)
@test isequal(size(unsqueeze(X)), (x_size*2, y_size*2, z_size*2, div(ch_size,8), batch_size))