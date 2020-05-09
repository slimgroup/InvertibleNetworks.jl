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
# Furthermore, the function contains functions for the three main functionalities:
# `forward` (define the forward pass)
# `inverse` (defines the inverse pass)
# `backward` (defines the backward pass, i.e. we backpropagate the data residual
# and compute the gradients with respect to our parameters)
struct AffineLayer <: NeuralNetLayer
    S::Parameter    # trainable parameters are defined as Parameters.
    B::Parameter    # both S and B have two fields: S.data and S.grad
    logdet::Bool    # bool to indicate whether you want to compute the logdet
    forward::Function   # forward function
    inverse::Function   # inverse function
    backward::Function  # backward function
end

# Functor the layer for gpu/cpu offloading
@Flux.functor AffineLayer

# The constructor builds and returns a new network layer for given input dimensions.
function AffineLayer(nx, ny, nc; logdet=false)

    # Create S and B
    S = Parameter(glorot_uniform(nx, ny, nc))   # initiliaze S with random values and make it a parameter
    B = Parameter(zeros(Float32, nx, ny, nc))   # initilize B with zeros and make it a parameter
    
    # Build an affine layer
    return AffineLayer(S, B, logdet,
        X -> affine_forward(X, S, B, logdet),
        Y -> affine_inverse(Y, S, B),
        (ΔY, Y) -> affine_backward(ΔY, Y, S, B, logdet)
    )
end

# Foward pass: Input X, Output Y
# The forward pass for the affine layer is:
# Y = X .* S .+ B
function affine_forward(X, S, B, logdet)

    Y = X .* S.data .+ B.data   # S and B are Parameters, so access their values via the .data field
    
    # If logdet is true, also compute the logdet and return it as a second output argument.
    # Otherwise only return Y.
    if logdet == true
        return Y, logdet_forward(S)
    else
        return Y
    end
end

# Inverse pass: Input Y, Output X
# The inverse pass for our affine layer is:
# X = (Y .- B) ./ S
# To avoid division by zero, we add numerical noise to S in the division.
function affine_inverse(Y, S, B)
    X = (Y .- B.data) ./ (S.data + randn(Float32, size(S.data)) .* eps(1f0))   # avoid division by 0
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
function affine_backward(ΔY, Y, S, B, logdet)
    nx, ny, n_in, batchsize = size(Y)
    
    # Recompute X from Y
    X = affine_inverse(Y, S, B)

    # Gradient w.r.t. X
    ΔX = ΔY .* S.data

    # Gradient w.r.t. S (sum over the batchsize, as S is only a 3D tensor)
    ΔS = sum(ΔY .* X, dims=4)[:,:,:,1]

    # If the logdet is computed, in the forward pass, also compute the gradient of the
    # logdet term and subtract from S (assuming that the logdet term is subtracted in the 
    # objective function)
    logdet == true && (ΔS -= logdet_backward(S))

    # Gradient w.r.t. B (sum over the batchsize)
    ΔB = sum(ΔY, dims=4)[:,:,:,1]

    # Set the gradient fields of the parameters S and B
    S.grad = ΔS
    B.grad = ΔB

    # Return the backpropagated data residual and the re-computed X
    return ΔX, X
end

# For optimization, we need a function that clears all the gradients.
# I.e. we set the .grad fields of all parameters to nothing.
function clear_grad!(AL::AffineLayer)
    AL.S.grad = nothing
    AL.B.grad = nothing
end

# Also we define a get_params function that returns an array of all
# the parameters. In this case, our parameters are S and B
get_params(AL::AffineLayer) = [AL.S, AL.B]

# Function for the logdet and for computing the gradient of the logdet.
# For our affine layer consisting of an element-wise multiplication of S
# and X, the Jacobian is given by S, and the logdet is the sum of the logarithm
# of the (absolute) values.
logdet_forward(S) = sum(log.(abs.(S.data))) 

# The gradient of the forward logdet function is given by 1/S
logdet_backward(S) = 1f0 ./ S.data
