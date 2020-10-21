# Original ResNet from He et al. (2015): https://arxiv.org/pdf/1512.03385.pdf with optional normalization (BatchNorm, Ioffe et al. (2015), https://arxiv.org/pdf/1502.03167.pdf)
# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: October 2020

export ResNet


function ResNet(n_in::Int64, n_hidden::Int64, nblocks::Int64; k::Int64=3, p::Int64=1, s::Int64=1, norm=BatchNorm, n_out::Union{Nothing, Int64}=nothing)

    resnet_blocks = Array{Any, 1}(undef, nblocks+!isnothing(n_out))
    for i = 1:nblocks
        if isnothing(n_out)
            resnet_blocks[i] = SkipConnection(Chain(Conv((k, k), n_in => n_hidden; stride = s, pad = p), Conv((k, k), n_hidden => n_in; stride = s, pad = p)), +)
        else
            resnet_blocks[i] = SkipConnection(Chain(Conv((k, k), n_in => n_hidden; stride = s, pad = p), norm(n_hidden, relu), Conv((k, k), n_hidden => n_in; stride = s, pad = p)), +)
        end
    end
    !isnothing(n_out) && (resnet_blocks[end] = Conv((k, k), n_in => n_out; stride = s, pad = p))
    model = Chain(resnet_blocks...)

    return FluxBlock(model)

end