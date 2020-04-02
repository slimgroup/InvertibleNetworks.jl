# Example how to use the residual block 
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using Test, LinearAlgebra, InvertibleNetworks, NNlib

# Input
nx1 = 32
nx2 = 32
nx_in = 8
n_hidden = 16   # same for x and y
batchsize = 2

ny1 = 64
ny2 = 44
ny_in = 4

# Input image
X = glorot_uniform(nx1, nx2, nx_in, batchsize)
D = glorot_uniform(ny1, ny2, ny_in, batchsize)

# # Residual blocks
# RB = ConditionalResidualBlock(nx1, nx2, nx_in, ny1, ny2, ny_in, n_hidden, batchsize)

# # Observed data
# Y, D_ = RB.forward(X, D)

# # Set data residual to zero
# ΔY = Y.*0f0; ΔD = D.*0f0

# # Backward pass
# ΔX, ΔD = RB.backward(ΔY, ΔD, X, D)


#######################################################################################################################


# Dense map
W0 = glorot_uniform(nx1*nx2*nx_in, ny1*ny2*ny_in)
b0 = zeros(Float32, nx1*nx2*nx_in)
Y0 = W0*reshape(D, :, batchsize) .+ b0

# Conv map
k1 = 3
s1 = 1
p1 = 1
W1 = glorot_uniform(k1, k1, nx_in, n_hidden)
b1 = zeros(Float32, n_hidden)
cdims1 = DenseConvDims((nx1, nx2, nx_in, batchsize), (k1, k1, nx_in, n_hidden); stride=(s1,s1), padding=(p1,p1))
X_ = conv(X, W1, cdims1)


#######################################################################################################################


function get_stride_padding(nx, ny, k)
    
    # Check dimensions
    if mod(nx, 2) != 0 || mod(ny, 2) != 0
        throw("Dimensions must be even integers.")
    end
    
    if nx == ny
        stride = 1
        pad_y = 0
        pad_x = Int((k-1)/2)
        is_transpose = false
    elseif nx > ny
        stride = Int(floor(nx/ny))
        pad_y = Int((nx - ny*stride)/2)
        pad_x = Int((k - mod(k,2))/2)
        is_transpose = false
    elseif nx < ny
        stride = Int(floor(ny/nx))
        pad_x = Int((ny - nx*stride)/2) + Int((k - mod(k,2))/2)
        pad_y = 0
        is_transpose = true
    else
        throw("Encountered exception during dimensionality check.")
    end

    return stride, pad_y, pad_x, is_transpose
end

function get_conv_dims(nx1, nx2, nx_in, ny1, ny2, ny_in, batchsize; k1=3, k2=3)

    s1, py1, px1, is_transpose = get_stride_padding(nx1, ny1, k1)
    s2, py2, px2, is_transpose = get_stride_padding(nx2, ny2, k2)

    cdims = DenseConvDims((nx1, nx2, nx_in, batchsize), (k1, k2, nx_in, ny_in); stride=(s1, s2), padding=(px1, px2))
    return cdims, is_transpose
end

cdims, is_transpose = get_conv_dims(nx1, nx2, nx_in, ny1, ny2, ny_in, batchsize; k1=3, k2=3)
W1 = randn(Float32, 3, 3, nx_in, ny_in)
Y = conv(X, W1, cdims)
