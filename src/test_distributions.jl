# 2D test distributions for generative models with the change of variables formula
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export sample_banana, sample_swirl


# Banana distribution
"""
    S = sample_banana(batchsize; c=[1f0, 4f0])

Generate samples from a 2D banana distribution. Samples S have
dimensions `1 x 1 x 2 x batchsize`.
"""
function sample_banana(batchsize; c=[1f0, 4f0])
    x = randn(Float32, 2, batchsize)
    y = zeros(Float32, 1, 1, 2, batchsize)
    y[1,1,1,:] = x[1,:] ./ c[1]
    y[1,1,2,:] = x[2,:].*c[1] + c[1].*c[2].*(x[1,:].^2 .+ c[1]^2)
    return y
end


# Swirl distribution
"""
    S = sample_swirl(batchsize; noise=1f0)

Generate samples from a 2D swirl distribution. Samples S have
dimensions `1 x 1 x 2 x batchsize`.
"""
function sample_swirl(batchsize; noise=1f0)
    n = sqrt.(rand(Float32, batchsize)) .* 800f0 .* (2f0*pi) ./ 360f0
    d1x = -cos.(n).*n .+ rand(Float32, batchsize) .* noise
    d1y = sin.(n).*n .+ rand(Float32, batchsize) .* noise
    return reshape(cat(d1x', d1y'; dims=1), 1, 1, 2, batchsize)
end