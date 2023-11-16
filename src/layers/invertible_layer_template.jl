# Template for implementing an invertible neural network layer and its logdet.
# We will take an affine layer as an example to explain all necessary steps.
# The affine layer is defined as:
#
# Y = f(X) = S .* X .+ B
#
# Here, X is the 4D input tensor of dimensions (nx, ny, num_channel, batchsize).
# S and B are a scaling and bias term, both with dimensions (nx, ny, num_channel).
#
# The trainable parameters of this layer are S, B. The input is X and the output is Y.
#


# You need to expoert whichever functions you want to be able to access
export AffineLayer

# This immutable structure defines our network layer. The structure contains the
# parameters of our layer (in this case S and B) or any other building blocks that
# you want to use in your layer (for example 1x1 convolutions). However, in this
# case we only have parameters S and B.
struct AffineLayer <: InvertibleNetwork
    S::Parameter    # trainable parameters are defined as Parameters.
    B::Parameter    # both S and B have two fields: S.data and S.grad
    logdet::Bool    # bool to indicate whether you want to compute the logdet
end

# Functor the layer for gpu/cpu offloading
@Flux.functor AffineLayer

# The constructor builds and returns a new network layer for given input dimensions.
function AffineLayer(nx, ny, nc; logdet=false)

    # Create S and B
    S = Parameter(glorot_uniform(nx, ny, nc))   # initiliaze S with random values and make it a parameter
    B = Parameter(zeros(Float32, nx, ny, nc))   # initilize B with zeros and make it a parameter

    # Build an affine layer
    return AffineLayer(S, B, logdet)
end

# Foward pass: Input X, Output Y
# The forward pass for the affine layer is:
# Y = X .* S .+ B
function forward(X::AbstractArray{T, N}, AL::AffineLayer) where {T, N}

    Y = X .* AL.S.data .+ AL.B.data   # S and B are Parameters, so access their values via the .data field

    # If logdet is true, also compute the logdet and return it as a second output argument.
    # Otherwise only return Y.
    if AL.logdet == true
        return Y, logdet_forward(S)
    else
        return Y
    end
end

# Inverse pass: Input Y, Output X
# The inverse pass for our affine layer is:
# X = (Y .- B) ./ S
# To avoid division by zero, we add numerical noise to S in the division.
function inverse(Y::AbstractArray{T, N}, AL::AffineLayer) where {T, N}
    X = (Y .- AL.B.data) ./ (AL.S.data .+ eps(T))   # avoid division by 0
    return X
end

# Backward pass: Input (ΔY, Y), Output (ΔY, Y)
# Assuming that the layer is invertible, the backward function takes
# 2 input arguments: the data residual ΔY and the original output Y.
# The first step is to recompute the original input X by calling the
# inverse function on Y. If the layer is not invertible, this layer
# needs X as an input instead of Y.
# Second of all, we compute the partial derivatives of our layer
# with respect to X, S, and B:
# ΔX = S .* ΔY (corresponds to df/dX * dY)
# ΔS = X .* ΔY (corresponds to df/dS * dY)
# ΔB = 1 .* ΔY (corresponds to df/dB * dY)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, AL::AffineLayer) where {T, N}
    nx, ny, n_in, batchsize = size(Y)

    # Recompute X from Y
    X = inverse(Y, AL)

    # Gradient w.r.t. X
    ΔX = ΔY .* AL.S.data

    # Gradient w.r.t. S (sum over the batchsize, as S is only a 3D tensor)
    ΔS = sum(ΔY .* X, dims=4)[:,:,:,1]

    # If the logdet is computed, in the forward pass, also compute the gradient of the
    # logdet term and subtract from S (assuming that the logdet term is subtracted in the
    # objective function)
    AL.logdet == true && (ΔS -= logdet_backward(S))

    # Gradient w.r.t. B (sum over the batchsize)
    ΔB = sum(ΔY, dims=4)[:,:,:,1]

    # Set the gradient fields of the parameters S and B
    AL.S.grad = ΔS
    AL.B.grad = ΔB

    # Return the backpropagated data residual and the re-computed X
    return ΔX, X
end

# Function for the logdet and for computing the gradient of the logdet.
# For our affine layer consisting of an element-wise multiplication of S
# and X, the Jacobian is given by S, and the logdet is the sum of the logarithm
# of the (absolute) values.
logdet_forward(S) = sum(log.(abs.(S.data)))

# The gradient of the forward logdet function is given by 1/S
logdet_backward(S) = 1f0 ./ S.data
