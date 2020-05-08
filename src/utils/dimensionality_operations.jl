# Dimensionality operations for 4D Tensors
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export squeeze, unsqueeze, wavelet_squeeze, wavelet_unsqueeze, tensor_split, tensor_cat

####################################################################################################
# Squeeze and unsqueeze

"""
    Y = squeeze(X; pattern="column")

 Reshape input image such that each spatial dimension is reduced by a factor
 of 2, while the number of channels is increased by a factor of 4. 

 *Input*: 
 
 - `X`: 4D input tensor of dimensions `nx` x `ny` x `n_channel` x `batchsize`

 - `pattern`: Squeezing pattern

        1 2 3 4        1 1 3 3        1 3 1 3
        1 2 3 4        1 1 3 3        2 4 2 4
        1 2 3 4        2 2 4 4        1 3 1 3
        1 2 3 4        2 2 4 4        2 4 2 4

        column          patch       checkerboard

 *Output*:
 
 - `Y`: Reshaped tensor of dimensions `nx/2` x `ny/2` x `n_channel*4` x `batchsize`

 See also: [`unsqueeze`](@ref), [`wavelet_squeeze`](@ref), [`wavelet_unsqueeze`](@ref)
"""
function squeeze(X::AbstractArray{T,4}; pattern="column") where T

    # Dimensions
    nx_in, ny_in, nc_in, batchsize = size(X)
    if mod(nx_in, 2) == 1 || mod(ny_in, 2) == 1 
        throw("Input dimensions must be multiple of 2")
    end
    nx_out = Int(round(nx_in/2))
    ny_out = Int(round(ny_in/2))
    nc_out = Int(round(nc_in*4))
    Y = zeros(T, nx_out, ny_out, nc_out, batchsize)

    if pattern == "column"
        Y = reshape(X, nx_out, ny_out, nc_out, batchsize)
    elseif pattern == "patch"
        Y[:, :, 1: nc_in, :] = X[1:nx_out, 1:ny_out, 1:nc_in, :]
        Y[:, :, nc_in+1: 2*nc_in, :] = X[nx_out+1:end, 1:ny_out, 1:nc_in, :]
        Y[:, :, 2*nc_in+1: 3*nc_in, :] = X[1:nx_out, ny_out+1:end, 1:nc_in, :]
        Y[:, :, 3*nc_in+1: 4*nc_in, :] = X[nx_out+1:end, ny_out+1:end, 1:nc_in, :]
    elseif pattern == "checkerboard"
        Y[:, :, 1: nc_in, :] = X[1:2:nx_in, 1:2:ny_in, 1:nc_in, :]
        Y[:, :, nc_in+1: 2*nc_in, :] = X[2:2:nx_in, 1:2:ny_in, 1:nc_in, :]
        Y[:, :, 2*nc_in+1: 3*nc_in, :] = X[1:2:nx_in, 2:2:ny_in, 1:nc_in, :]
        Y[:, :, 3*nc_in+1: 4*nc_in, :] = X[2:2:nx_in, 2:2:ny_in, 1:nc_in, :]
    else
        throw("Specified pattern not defined.")
    end
    return Y
end


"""
    X = unsqueeze(Y; pattern="column")

 Undo squeezing operation by reshaping input image such that each spatial dimension is 
 increased by a factor of 2, while the number of channels is decreased by a factor of 4. 

 *Input*: 
 
 - `Y`: 4D input tensor of dimensions `nx` x `ny` x `n_channel` x `batchsize`

 - `pattern`: Squeezing pattern
 
            1 2 3 4        1 1 3 3        1 3 1 3
            1 2 3 4        1 1 3 3        2 4 2 4
            1 2 3 4        2 2 4 4        1 3 1 3
            1 2 3 4        2 2 4 4        2 4 2 4

            column          patch       checkerboard

 *Output*:
 
 - `X`: Reshaped tensor of dimensions `nx*2` x `ny*2` x `n_channel/4` x `batchsize`

 See also: [`squeeze`](@ref), [`wavelet_squeeze`](@ref), [`wavelet_unsqueeze`](@ref)
"""
function unsqueeze(Y::AbstractArray{T,4}; pattern="column") where T

    # Dimensions
    nx_in, ny_in, nc_in, batchsize = size(Y)
    if mod(nx_in, 2) == 1 || mod(ny_in, 2) == 1 
        throw("Input dimensions must be multiple of 2")
    end
    nx_out = Int(round(nx_in*2))
    ny_out = Int(round(ny_in*2))
    nc_out = Int(round(nc_in/4))
    X = zeros(T, nx_out, ny_out, nc_out, batchsize)

    if pattern == "column"
        X = reshape(Y, nx_out, ny_out, nc_out, batchsize)
    elseif pattern == "patch"
        X[1:nx_in, 1:ny_in, 1:nc_out, :] = Y[:, :, 1: nc_out, :]
        X[nx_in+1:end, 1:ny_in, 1:nc_out, :] = Y[:, :, nc_out+1: 2*nc_out, :]
        X[1:nx_in, ny_in+1:end, 1:nc_out, :] = Y[:, :, 2*nc_out+1: 3*nc_out, :]
        X[nx_in+1:end, ny_in+1:end, 1:nc_out, :] = Y[:, :, 3*nc_out+1: 4*nc_out, :]
    elseif pattern == "checkerboard"
        X[1:2:nx_out, 1:2:ny_out, 1:nc_out, :] = Y[:, :, 1: nc_out, :]
        X[2:2:nx_out, 1:2:ny_out, 1:nc_out, :] = Y[:, :, nc_out+1: 2*nc_out, :]
        X[1:2:nx_out, 2:2:ny_out, 1:nc_out, :] = Y[:, :, 2*nc_out+1: 3*nc_out, :]
        X[2:2:nx_out, 2:2:ny_out, 1:nc_out, :] = Y[:, :, 3*nc_out+1: 4*nc_out, :]
    else
        throw("Specified pattern not defined.")
    end
    return X
end

function unsqueeze(X::AbstractArray{T,4}, Y::AbstractArray{T,4}; pattern="column") where T
    return unsqueeze(X; pattern=pattern), unsqueeze(Y; pattern=pattern)
end


####################################################################################################
# Squeeze and unsqueeze using the wavelet transform

"""
    Y = wavelet_squeeze(X; type=WT.db1)

 Perform a 1-level channelwise 2D wavelet transform of X and squeeze output of each
 transform into 4 channels (per 1 input channel).

 *Input*: 
 
 - `X`: 4D input tensor of dimensions `nx` x `ny` x `n_channel` x `batchsize`

 - `type`: Wavelet filter type. Possible values are `WT.haar` for Haar wavelets, 
    `WT.coif2`, `WT.coif4`, etc. for Coiflet wavelets, or `WT.db1`, `WT.db2`, etc. 
    for Daubechies wavetlets. See *https://github.com/JuliaDSP/Wavelets.jl* for a 
    full list.

 *Output*:
 
 - `Y`: Reshaped tensor of dimensions `nx/2` x `ny/2` x `n_channel*4` x `batchsize`

 See also: [`wavelet_unsqueeze`](@ref), [`squeeze`](@ref), [`unsqueeze`](@ref)
"""
function wavelet_squeeze(X::AbstractArray{T,4}; type=WT.db1) where T
    nx_in, ny_in, nc_in, batchsize = size(X)
    nx_out = Int(round(nx_in/2))
    ny_out = Int(round(ny_in/2))
    nc_out = Int(round(nc_in*4))
    Y = zeros(T, nx_out, ny_out, nc_out, batchsize)
    for i=1:batchsize
        for j=1:nc_in
            Ycurr = dwt(X[:,:,j,i], wavelet(type), 1)
            Y[:, :, (j-1)*4 + 1: j*4, i] = squeeze(reshape(Ycurr, nx_in, ny_in, 1, 1); pattern="patch")
        end
    end
    return Y
end


"""
    X = wavelet_unsqueeze(Y; type=WT.db1)

 Perform a 1-level inverse 2D wavelet transform of Y and unsqueeze output.
 This reduces the number of channels by 4 and increases each spatial
 dimension by a factor of 2. Inverse operation of `wavelet_squeeze`.

 *Input*: 
 
 - `Y`: 4D input tensor of dimensions `nx` x `ny` x `n_channel` x `batchsize`

 - `type`: Wavelet filter type. Possible values are `haar` for Haar wavelets,
  `coif2`, `coif4`, etc. for Coiflet wavelets, or `db1`, `db2`, etc. for Daubechies
  wavetlets. See *https://github.com/JuliaDSP/Wavelets.jl* for a full list.

 *Output*:
 
 - `X`: Reshaped tensor of dimenions `nx*2` x `ny*2` x `n_channel/4` x `batchsize`

 See also: [`wavelet_squeeze`](@ref), [`squeeze`](@ref), [`unsqueeze`](@ref)
"""
function wavelet_unsqueeze(Y::AbstractArray{T,4}; type=WT.db1) where T
    nx_in, ny_in, nc_in, batchsize = size(Y)
    nx_out = Int(round(nx_in*2))
    ny_out = Int(round(ny_in*2))
    nc_out = Int(round(nc_in/4))
    X = zeros(T, nx_out, ny_out, nc_out, batchsize)
    for i=1:batchsize
        for j=1:nc_out
            Ycurr = unsqueeze(Y[:, :, (j-1)*4 + 1: j*4, i:i]; pattern="patch")[:, :, 1, 1]
            Xcurr = idwt(Ycurr, wavelet(type), 1)
            X[:, :, j, i] = reshape(Xcurr, nx_out, ny_out, 1, 1)
        end
    end
    return X
end


####################################################################################################
# Split and concatenate

"""
    Y, Z = tensor_split(X)

 Split 4D input tensor in half along the channel dimension. Inverse operation
 of `tensor_cat`.

 *Input*: 
 
 - `X`: 4D input tensor of dimensions `nx` x `ny` x `n_channel` x `batchsize`

 *Output*:
 
 - `Y`, `Z`: 4D output tensors, each of dimensions `nx` x `ny` x `n_channel/2` x `batchsize`

 See also: [`tensor_cat`](@ref)
"""
function tensor_split(X::AbstractArray{T,4}; split_index=nothing) where T
    if isnothing(split_index)
        k = Int(round(size(X, 3)/2))
    else
        k = split_index
    end
    return X[:, :, 1:k, :], X[:, :, k+1:end, :]
end

function tensor_split(X::AbstractArray{T,5}; split_index=nothing) where T
    if isnothing(split_index)
        k = Int(round(size(X, 4)/2))
    else
        k = split_index
    end
    return X[:, :, :, 1:k, :], X[:, :, :, k+1:end, :]
end

function tensor_split(X::AbstractArray{T,1}; split_index=nothing) where T
    if isnothing(split_index)
        k = Int(round(size(X, 1)/2))
    else
        k = split_index
    end
    return X[1:k], X[k+1:end]
end

"""
    X = tensor_cat(Y, Z)

 Concatenate 4D input tensors along the channel dimension. Inverse operation
 of `tensor_split`.

 *Input*: 
 
 - `Y`, `Z`: 4D input tensors, each of dimensions `nx` x `ny` x `n_channel` x `batchsize`

 *Output*:
 
 - `X`: 4D output tensor of dimensions `nx` x `ny` x `n_channel*2` x `batchsize`

 See also: [`tensor_split`](@ref)
"""
function tensor_cat(X::AbstractArray{T,4}, Y::AbstractArray{T,4}) where T 
    if size(X, 3) == 0
        return Y
    elseif size(Y, 3) == 0
        return X
    else
        return cat(X, Y; dims=3)
    end
end

function tensor_cat(X::AbstractArray{T,5}, Y::AbstractArray{T,5}) where T
    if size(X, 4) == 0
        return Y
    elseif size(Y, 4) == 0
        return X
    else
        return cat(X, Y; dims=4)
    end
end

function tensor_cat(X::AbstractArray{T,1}, Y::AbstractArray{T,1}) where T
    if size(X, 1) == 0
        return Y
    elseif size(Y, 1) == 0
        return X
    else
        return cat(X, Y; dims=1)
    end
end

tensor_cat(X::Tuple{AbstractArray{T,4}, AbstractArray{T,4}}) where T = tensor_cat(X[1], X[2])
