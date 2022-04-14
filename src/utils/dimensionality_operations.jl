# Dimensionality operations for 4D Tensors
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export squeeze, unsqueeze, wavelet_squeeze, wavelet_unsqueeze, Haar_squeeze, invHaar_unsqueeze 
export tensor_split, tensor_cat
export cat_states, split_states
export ShuffleLayer, WaveletLayer, HaarLayer
###############################################################################
# Custom type for squeezer functions

struct Squeezer
    forward::Function
    inverse::Function
end

function ShuffleLayer(;pattern="checkerboard")
    return Squeezer(x -> squeeze(x;pattern=pattern), x -> unsqueeze(x;pattern=pattern))
end

function WaveletLayer(;type=WT.db1)
    return Squeezer(x -> wavelet_squeeze(x;type=type), x -> wavelet_unsqueeze(x;type=type))
end

function HaarLayer()
    return Squeezer(x -> Haar_squeeze(x), x -> invHaar_unsqueeze(x))
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
 
 Squeeze operation that is only a reshape. 

 Reshape input image such that each spatial dimension is reduced by a factor
 of 2, while the number of channels is increased by a factor of 4 if 4D tensor 
 and increased by a factor of 8 if 5D tensor.

 *Input*:

 - `X`: 4D/5D  input tensor of dimensions `nx` x `ny` (x `nz`) x `n_channel` x `batchsize`

 - `pattern`: Squeezing pattern

        1 2 3 4        1 1 3 3        1 3 1 3
        1 2 3 4        1 1 3 3        2 4 2 4
        1 2 3 4        2 2 4 4        1 3 1 3
        1 2 3 4        2 2 4 4        2 4 2 4

        column          patch       checkerboard

 *Output*:
 if 4D tensor:
 - `Y`: Reshaped tensor of dimensions `nx/2` x `ny/2` x `n_channel*4` x `batchsize`
 or if 5D tensor:
 - `Y`: Reshaped tensor of dimensions `nx/2` x `ny/2` x `nz/2` x `n_channel*8` x `batchsize`

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
    cinds = Tuple(Colon() for i=1:N-2)

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
 increased by a factor of 2, while the number of channels is decreased by a factor of 4
 if 4D tensor of decreased by a factor of 8 if a 5D tensor.

 *Input*:

 - `Y`: 4D/5D input tensor of dimensions `nx` x `ny` (x `nz`) x `n_channel` x `batchsize`

 - `pattern`: Squeezing pattern

            1 2 3 4        1 1 3 3        1 3 1 3
            1 2 3 4        1 1 3 3        2 4 2 4
            1 2 3 4        2 2 4 4        1 3 1 3
            1 2 3 4        2 2 4 4        2 4 2 4

            column          patch       checkerboard

 *Output*:
 If 4D tensor:
 - `X`: Reshaped tensor of dimensions `nx*2` x `ny*2` x `n_channel/4` x `batchsize`
 If 5D tensor:
 - `X`: Reshaped tensor of dimensions `nx*2` x `ny*2` x `nz*2` x `n_channel/8` x `batchsize`

 See also: [`squeeze`](@ref), [`wavelet_squeeze`](@ref), [`wavelet_unsqueeze`](@ref)
"""
function unsqueeze(Y::AbstractArray{T,N}; pattern="column") where {T, N}

    # Dimensions
    batchsize = size(Y, N)
    if mod(size(Y, N-1), 2^(N-2)) != 0
        throw("With tensor of dimension N, number of channels must be divisible by 2^(N-2)")
    end
    N_out = Tuple(nn*2 for nn=size(Y)[1:N-2])
    nc_out = size(Y, N-1) ÷ 2^(N-2)
    cinds = Tuple(Colon() for i=1:N-2)

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
 transform to increase number of channels by a factor of 4 if input is 4D tensor or by a factor of 
 8 if a 5D tensor.

 *Input*:

 - `X`: 4D/5D input tensor of dimensions `nx` x `ny` (x `nz`) x `n_channel` x `batchsize`

 - `type`: Wavelet filter type. Possible values are `WT.haar` for Haar wavelets,
    `WT.coif2`, `WT.coif4`, etc. for Coiflet wavelets, or `WT.db1`, `WT.db2`, etc.
    for Daubechies wavetlets. See *https://github.com/JuliaDSP/Wavelets.jl* for a
    full list.

 *Output*:
  if 4D tensor:
  - `Y`: Reshaped tensor of dimensions `nx/2` x `ny/2` x `n_channel*4` x `batchsize`
  or if 5D tensor:
  - `Y`: Reshaped tensor of dimensions `nx/2` x `ny/2` x `nz/2` x `n_channel*8` x `batchsize`
 See also: [`wavelet_unsqueeze`](@ref), [`squeeze`](@ref), [`unsqueeze`](@ref)
"""
function wavelet_squeeze(X::AbstractArray{T, N}; type=WT.db1) where {T, N}
  
    batchsize = size(X, N)
    N_in = size(X)[1:N-2]
    N_out = Tuple(nn÷2 for nn=size(X)[1:N-2])
    nd = 2^(N-2)
    nc_out = size(X, N-1) * nd
    cinds = Tuple(Colon() for i=1:N-2)

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
 This reduces the number of channels by factor of 4 if 4D tensor or by a 
 factor of 8 if 5D tensor and increases each spatial dimension by a factor of 2.
 Inverse operation of `wavelet_squeeze`.

 *Input*:

 - `Y`: 4D/5D input tensor of dimensions `nx` x `ny` (x `nz`) x `n_channel` x `batchsize`

 - `type`: Wavelet filter type. Possible values are `haar` for Haar wavelets,
  `coif2`, `coif4`, etc. for Coiflet wavelets, or `db1`, `db2`, etc. for Daubechies
  wavetlets. See *https://github.com/JuliaDSP/Wavelets.jl* for a full list.

 *Output*:
 If 4D tensor:
 - `X`: Reshaped tensor of dimensions `nx*2` x `ny*2` x `n_channel/4` x `batchsize`
 If 5D tensor:
 - `X`: Reshaped tensor of dimensions `nx*2` x `ny*2` x `nz*2` x `n_channel/8` x `batchsize`

 See also: [`wavelet_squeeze`](@ref), [`squeeze`](@ref), [`unsqueeze`](@ref)
"""
function wavelet_unsqueeze(Y::AbstractArray{T,N}; type=WT.db1) where {T, N}

    N_out = Tuple(nn*2 for nn=size(Y)[1:N-2])
    nd = 2^(N-2)
    nc_out = size(Y, N-1) ÷ nd
    cinds = Tuple(Colon() for i=1:N-2)

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
    inds = [i==dim ? (1:2:size(x, dim)) : Colon() for i=1:N];
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
    inds = [i==dim ? (1:2:size(x, dim)) : Colon() for i=1:N];
    x[inds...] .= H
    inds[dim] = 2:2:size(x, dim)
    x[inds...] .= L

    return x
end

"""
    Y = Haar_squeeze(X)

 Perform a 1-level channelwise 2D/3D (lifting) Haar transform of X and squeeze output of each
 transform to increase channels by factor of 4 in 4D tensor or by factor of 8 in 5D channels.

 *Input*:

 - `X`: 4D/5D input tensor of dimensions `nx` x `ny` (x `nz`) x `n_channel` x `batchsize`

 *Output*:

 if 4D tensor:
 - `Y`: Reshaped tensor of dimensions `nx/2` x `ny/2` x `n_channel*4` x `batchsize`
  or if 5D tensor:
 - `Y`: Reshaped tensor of dimensions `nx/2` x `ny/2` x `nz/2` x `n_channel*8` x `batchsize`

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
 This reduces the number of channels by factor of 4 in 4D tensors or by factor
 of 8 in 5D tensors and increases each spatial dimension by a factor of 2.
 Inverse operation of `Haar_squeeze`.

 *Input*:

 - `Y`: 4D/5D input tensor of dimensions `nx` x `ny` (x `nz`) x `n_channel` x `batchsize`

 *Output*:

 If 4D tensor:
 - `X`: Reshaped tensor of dimensions `nx*2` x `ny*2` x `n_channel/4` x `batchsize`
 If 5D tensor:
 - `X`: Reshaped tensor of dimensions `nx*2` x `ny*2` x `nz*2` x `n_channel/8` x `batchsize`

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
function tensor_split(X::AbstractArray{T, N}; split_index=nothing) where {T, N}
    d = max(1, N-1)
    if isnothing(split_index)
        k = Int(round(size(X, d)/2))
    else
        k = split_index
    end

    indsl = [i==d ? (1:k) : Colon() for i=1:N]
    indsr = [i==d ? (k+1:size(X, d)) : Colon() for i=1:N]

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
function tensor_cat(X::AbstractArray{T, N}, Y::AbstractArray{T, N}) where {T, N}
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

# In place cat
function tensor_cat!(out::AbstractArray{T, N}, X::AbstractArray{T, N}, Y::AbstractArray{T, N}) where {T, N}
    d = max(1, N-1)
    if size(X, d) == 0
        copyto!(out, Y)
    elseif size(Y, d) == 0
        copyto!(out, X)
    else
        k = size(X, d)
        indsl = [i==d ? (1:k) : Colon() for i=1:N]
        indsr = [i==d ? (k+1:size(out, d)) : Colon() for i=1:N]
        out[indsl...] .= X
        out[indsr...] .= Y
    end
end

@inline xy_dims(dims::Array, ::Val{false}) = tuple(dims...)
@inline xy_dims(dims::Array, ::Val{true}) = tuple(Int.(dims .* (.5, .5, 4, 1))...)

# Concatenate states Zi and final output
function cat_states(XY_save::AbstractMatrix{<:AbstractArray}, X::AbstractArray{T, 4}, Y::AbstractArray{T, 4}) where T
    return cat_states(XY_save[:, 1], X), cat_states(XY_save[:, 2], Y)
end
# Concatenate states Zi and final output
function cat_states(Z_save::Vector{<:AbstractArray}, X::AbstractArray{T, N}) where {T, N}
    return cat([[vec(Z_save[j]) for j=1:length(Z_save)]..., vec(X)]..., dims=1)
end

# Split and reshape 1D vector Y in latent space back to states Zi
# where Zi is the split tensor at each multiscale level.
function split_states(Y::AbstractVector{T}, Z_dims) where {T, N}
    L = length(Z_dims) + 1
    inds = cumsum([1, [prod(Z_dims[j]) for j=1:L-1]...])
    Z_save = [reshape(Y[inds[j]:inds[j+1]-1], xy_dims(Z_dims[j], Val(j==L))) for j=1:L-1]
    X = reshape(Y[inds[L]:end], xy_dims(Z_dims[end], Val(true)))
    return Z_save, X
end

# Split and reshape 1D vector X_full and Y_vull in latent space back to states Zi
# where Zi is the split tensor at each multiscale level.
function split_states(X_full::AbstractArray{T, 1}, Y_full::AbstractArray{T, 1}, XY_dims::AbstractArray{Array, 1}) where T
    c1, X  = split_states(X_full, XY_dims)
    c2, Y  = split_states(Y_full, XY_dims)
    return hcat(c1, c2), X, Y
end
