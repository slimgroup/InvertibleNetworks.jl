
# Dimensionality operations for 4D Tensors
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export general_squeeze, general_unsqueeze, squeeze, unsqueeze, wavelet_squeeze, wavelet_unsqueeze, Haar_squeeze, invHaar_unsqueeze, tensor_split, tensor_cat

####################################################################################################
# General Squeeze and unsqueeze for user selection
function general_squeeze(X::AbstractArray{T, N}; squeeze_type="normal", pattern="column") where {T, N}
    if squeeze_type == "normal"
        Y = squeeze(X; pattern=pattern)
    elseif squeeze_type == "wavelet"
        Y = wavelet_squeeze(X)
    elseif squeeze_type == "Haar"
        Y = Haar_squeeze(X)
    else
        throw("Specified squeeze not defined.")
    end
    return Y
end

function general_unsqueeze(X::AbstractArray{T, N}; squeeze_type="normal", pattern="column") where {T, N}
    if squeeze_type == "normal"
        Y = unsqueeze(X; pattern=pattern)
    elseif squeeze_type == "wavelet"
        Y = wavelet_unsqueeze(X)
    elseif squeeze_type == "Haar"
        Y = invHaar_unsqueeze(X)
    else
        throw("Specified unsqueeze not defined.")
    end
    return Y
end

####################################################################################################
# Squeeze and unsqueeze
function patch_inds(N::NTuple{n, Int}, d::Integer) where n
    indsX = [ix*N[1]+1:(ix+1)*N[1] for ix=[(i+1)%2 for i=1:2^(d-2)]]
    indsY = [iy*N[2]+1:(iy+1)*N[2] for iy=[div(i-1,2)%2 for i=1:2^(d-2)]]
    d == 4 && (return zip(indsX, indsY))
    indsZ = [iz*N[3]+1:(iz+1)*N[3] for iz=[div(i-1,4)%2 for i=1:2^(d-2)]]
    return zip(indsX, indsY, indsZ)
end

function checkboard_inds(N::NTuple{n, Int}, d::Integer) where n
    indsX = [ix+1:2:N[1] for ix=[(i+1)%2 for i=1:2^(d-2)]]
    indsY = [iy+1:2:N[2] for iy=[div(i-1,2)%2 for i=1:2^(d-2)]]
    d == 4 && (return zip(indsX, indsY))
    indsZ = [iz:2:N[3] for iz=[div(i-1,4)%2 for i=1:2^(d-2)]]
    return zip(indsX, indsY, indsZ)
end

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
function squeeze(X::AbstractArray{T, N}; pattern="column") where {T, N}

    # Dimensions
    nc_in, batchsize = size(X)[N-1:N]
    if any([mod(nn, 2) == 1 for nn=size(X)[1:N-2]])
        throw("Input dimensions must be multiple of 2")
    end
    N_out = Tuple(nn÷2 for nn=size(X)[1:N-2])
    nc_out = size(X, N-1) * 2^(N-2)
    cinds = Tuple((:) for i=1:N-2)

    if pattern == "column"
        Y = reshape(X, N_out..., nc_out, batchsize)
    elseif pattern == "patch"
        Y = cuzeros(X, N_out..., nc_out, batchsize)
        iX = patch_inds(N_out, N)
        for (i, ix)=enumerate(iX)
            Y[cinds..., (i-1)*nc_in+1:i*nc_in, :] = X[ix..., :, :]
        end
    elseif pattern == "checkerboard"
        Y = cuzeros(X, N_out..., nc_out, batchsize)
        iX = checkboard_inds(size(X), N)
        for (i, ix)=enumerate(iX)
            Y[cinds..., (i-1)*nc_in+1:i*nc_in, :] = X[ix..., :, :]
        end
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
function unsqueeze(Y::AbstractArray{T,N}; pattern="column") where {T, N}

    # Dimensions
    batchsize = size(Y, N)
    if any([mod(nn, 2) == 1 for nn=size(Y)[1:N-2]])
        throw("Input dimensions must be multiple of 2")
    end
    N_out = Tuple(nn*2 for nn=size(Y)[1:N-2])
    nc_out = size(Y, N-1) ÷ 2^(N-2)
    cinds = Tuple((:) for i=1:N-2)

    if pattern == "column"
        X = reshape(Y, N_out..., nc_out, batchsize)
    elseif pattern == "patch"
        X = cuzeros(Y, N_out..., nc_out, batchsize)
        iX = patch_inds(size(Y), N)
        for (i, ix)=enumerate(iX)
            X[ix..., :, :] = Y[cinds..., (i-1)*nc_out+1:i*nc_out, :]
        end
    elseif pattern == "checkerboard"
        X = cuzeros(Y, N_out..., nc_out, batchsize)
        iX = checkboard_inds(N_out, N)
        for (i, ix)=enumerate(iX)
            X[ix..., :, :] = Y[cinds..., (i-1)*nc_out+1:i*nc_out, :]
        end
    else
        throw("Specified pattern not defined.")
    end
    return X
end

function unsqueeze(X::AbstractArray{T,N}, Y::AbstractArray{T,4}; pattern="column") where {T,N}
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
function wavelet_squeeze(X::AbstractArray{T, N}; type=WT.db1) where {T, N}
    batchsize = size(X, N)
    N_in = size(X)[1:N-2]
    N_out = Tuple(nn÷2 for nn=size(X)[1:N-2])
    nd = 2^(N-2)
    nc_out = size(X, N-1) * nd
    cinds = Tuple((:) for i=1:N-2)

    Y = cuzeros(X, N_out..., nc_out, batchsize)
    for i=1:batchsize
        for j=1:size(X, N-1)
            Ycurr = dwt(X[cinds..., j, i], wavelet(type), 1)
            Y[cinds..., (j-1)*nd + 1: j*nd, i] = squeeze(reshape(Ycurr, N_in..., 1, 1); pattern="patch")
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
function wavelet_unsqueeze(Y::AbstractArray{T,N}; type=WT.db1) where {T, N}
    N_out = Tuple(nn*2 for nn=size(Y)[1:N-2])
    nd = 2^(N-2)
    nc_out = size(Y, N-1) ÷ nd
    cinds = Tuple((:) for i=1:N-2)

    X = cuzeros(Y, N_out..., nc_out, size(Y, N))
    for i=1:size(Y, N)
        for j=1:nc_out
            Ycurr = unsqueeze(Y[cinds..., (j-1)*nd + 1: j*nd, i:i]; pattern="patch")[cinds..., 1, 1]
            Xcurr = idwt(Ycurr, wavelet(type), 1)
            X[cinds..., j, i] = reshape(Xcurr, N_out..., 1, 1)
        end
    end
    return X
end


########## Haaar wavelet, GPU supported #####################

function HaarLift(x::AbstractArray{T,N}, dim) where {T, N}
    dim > N - 2 && return (x,)
    #Haar lifting
    inds = [i==dim ? (1:2:size(x, dim)) : (:) for i=1:N];
    # Splitting
    H = x[inds...]
    inds[dim] = 2:2:size(x, dim)
    L = x[inds...]

    # predict
    H .-= L

    #update
    L .+= H ./ T(2.0)

    #normalize
    H ./= sqrt(T(2.0))
    L .*= sqrt(T(2.0))

    return L, H
end

function invHaarLift(x::AbstractArray{T,N}, dim) where {T, N}
    dim > N - 2 && return x
    return invHaarLift(tensor_split(x)..., dim)
end

function invHaarLift(L::AbstractArray{T,N}, H::AbstractArray{T,N},dim) where {T, N}
    #inverse Haar lifting

    #inv normalize
    H .*= sqrt(T(2.0))
    L ./= sqrt(T(2.0))

    #inv update & predict
    L .-= H ./ T(2.0)
    H .+= L

    #allocate output:
    x = cat(cuzeros(L,size(L)...), cuzeros(H,size(H)...), dims=dim)
    inds = [i==dim ? (1:2:size(x, dim)) : (:) for i=1:N];
    x[inds...] .= H
    inds[dim] = 2:2:size(x, dim)
    x[inds...] .= L

    return x
end

"""
    Y = Haar_squeeze(X)

 Perform a 1-level channelwise 2D/3D (lifting) Haar transform of X and squeeze output of each
 transform into 8 channels (per 1 input channel).

 *Input*:

 - `X`: 4D/5D input tensor of dimensions `nx` x `ny` (x `nz`) x `n_channel` x `batchsize`

 *Output*:

 - `Y`: Reshaped tensor of dimensions `nx/2` x `ny/2` (x `nz/2`) x `n_channel*8` x `batchsize`

 See also: [`wavelet_unsqueeze`](@ref), [`Haar_unsqueeze`](@ref), [`HaarLift`](@ref), [`squeeze`](@ref), [`unsqueeze`](@ref)
"""
function Haar_squeeze(x::AbstractArray{T, N}) where {T, N}

        L, H = HaarLift(x, 2)

        a, h = HaarLift(L ,1)
        v, d = HaarLift(H, 1)

        a = HaarLift(a, 3)
        v = HaarLift(v, 3)
        h = HaarLift(h, 3)
        d = HaarLift(d, 3)

        return cat(a..., v..., h..., d...,dims=N-1)
end


"""
    X = invHaar_unsqueeze(Y)

 Perform a 1-level inverse 2D/3D Haar transform of Y and unsqueeze output.
 This reduces the number of channels by 8 and increases each spatial
 dimension by a factor of 2. Inverse operation of `Haar_squeeze`.

 *Input*:

 - `Y`: 4D/5D input tensor of dimensions `nx` x `ny` (x `nz`) x `n_channel` x `batchsize`

 *Output*:

 - `X`: Reshaped tensor of dimenions `nx*2` x `ny*2` (x `nz*2`) x `n_channel/8` x `batchsize`

 See also: [`wavelet_unsqueeze`](@ref), [`Haar_unsqueeze`](@ref), [`HaarLift`](@ref), [`squeeze`](@ref), [`unsqueeze`](@ref)
"""
function invHaar_unsqueeze(x::AbstractArray{T, N}) where {T, N}

        s = size(x, N-1)
        a, h = tensor_split(x)
        a, v, h, d = tensor_split(a)..., tensor_split(h)...

        a = invHaarLift(a, 3)
        v = invHaarLift(v, 3)
        h = invHaarLift(h, 3)
        d = invHaarLift(d, 3)

        L = invHaarLift(a, h, 1)
        H = invHaarLift(v, d, 1)

        x = invHaarLift(L, H, 2)

        return x
end

####################################################################################################
# Split and concatenate

"""
    Y, Z = tensor_split(X)

 Split ND input tensor in half along the channel dimension. Inverse operation
 of `tensor_cat`.

 *Input*:

 - `X`: ND input tensor of dimensions `nx` [x `ny` [x `nz`]] x `n_channel` x `batchsize`

 *Output*:

 - `Y`, `Z`: ND output tensors, each of dimensions `nx` [x `ny` [x `nz`]] x `n_channel/2` x `batchsize`

 See also: [`tensor_cat`](@ref)
"""
function tensor_split(X::AbstractArray{T,N}; split_index=nothing) where {T, N}
    d = max(1, N-1)
    if isnothing(split_index)
        k = Int(round(size(X, d)/2))
    else
        k = split_index
    end

    indsl = [i==d ? (1:k) : (:) for i=1:N]
    indsr = [i==d ? (k+1:size(X, d)) : (:) for i=1:N]

    return X[indsl...], X[indsr...]
end

"""
    X = tensor_cat(Y, Z)

 Concatenate ND input tensors along the channel dimension. Inverse operation
 of `tensor_split`.

 *Input*:

 - `Y`, `Z`: ND input tensors, each of dimensions `nx` [x `ny` [x `nz`]] x `n_channel` x `batchsize`

 *Output*:

 - `X`: ND output tensor of dimensions `nx` [x `ny` [x `nz`]] x `n_channel*2` x `batchsize`

 See also: [`tensor_split`](@ref)
"""
function tensor_cat(X::AbstractArray{T,N}, Y::AbstractArray{T,N}) where {T, N}
    d = max(1, N-1)
    if size(X, d) == 0
        return Y
    elseif size(Y, d) == 0
        return X
    else
        return cat(X, Y; dims=d)
    end
end

tensor_cat(X::Tuple{AbstractArray{T,N}, AbstractArray{T,N}}) where {T, N} = tensor_cat(X[1], X[2])
