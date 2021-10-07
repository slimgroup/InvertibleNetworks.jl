export ConvolutionalBlock

struct ConvolutionalBlock{T} <: NeuralNetLayer
    CL1::ConvolutionalLayer{T}
    A1::Union{Nothing,ActNormPar{T}}
    CL2::ConvolutionalLayer{T}
    A2::Union{Nothing,ActNormPar{T}}
    CL3::ConvolutionalLayer0{T}
end

@Flux.functor ConvolutionalBlock

function ConvolutionalBlock(nc_in, nc_out, nc_hidden; k1=3, p1=1, s1=1, actnorm1::Bool=true, k2=1, p2=0, s2=1, actnorm2::Bool=true, k3=3, p3=1, s3=1, weight_std1::Real=0.05, weight_std2::Real=0.05, logscale_factor::Real=3.0, init_zero::Bool=true, T::DataType=Float32)

    CL1 = ConvolutionalLayer(nc_in, nc_hidden; k=k1, p=p1, s=s1, bias=~actnorm1, weight_std=weight_std1, T=T)
    actnorm1 ? (A1 = ActNormPar(nc_hidden; logdet=false, T=T)) : (A1 = nothing)
    CL2 = ConvolutionalLayer(nc_hidden, nc_hidden; k=k2, p=p2, s=s2, bias=~actnorm2, weight_std=weight_std2, T=T)
    actnorm2 ? (A2 = ActNormPar(nc_hidden; logdet=false, T=T)) : (A2 = nothing)
    if init_zero
        CL3 = ConvolutionalLayer0(nc_hidden, nc_out; k=k3, p=p3, s=s3, logscale_factor=logscale_factor, T=T)
    else
        CL3 = ConvolutionalLayer0(nc_hidden, nc_out; k=k3, p=p3, s=s3, logscale_factor=logscale_factor, weight_std=weight_std2, T=T)
    end

    return ConvolutionalBlock{T}(CL1, A1, CL2, A2, CL3)

end

function forward(X::AbstractArray{T,N}, CB::ConvolutionalBlock{T}; save::Bool=false) where {N,T}

    C1 = CB.CL1.forward(X)
    CB.A1 !== nothing ? (A1 = CB.A1.forward(C1)) : (A1 = C1)
    H1 = ReLU(A1)

    C2 = CB.CL2.forward(H1)
    CB.A2 !== nothing ? (A2 = CB.A2.forward(C2)) : (A2 = C2)
    H2 = ReLU(A2)

    Y = CB.CL3.forward(H2)
    # C3 = CB.CL3.forward(H2)
    # Y = X+C3

    ~save ? (return Y) : (return A1, H1, A2, H2, Y)#(return A1, H1, A2, H2, C3, Y)#

end

function backward(ΔY::AbstractArray{T, N}, X::AbstractArray{T,N}, CB::ConvolutionalBlock{T}) where {N,T}

    A1, H1, A2, H2, Y = CB.forward(X; save=true) # Recompute forward states from input
    # A1, H1, A2, H2, C3, Y = CB.forward(X; save=true) # Recompute forward states from input

    # ΔH2 = CB.CL3.backward(ΔY, H2; Z=C3)
    ΔH2 = CB.CL3.backward(ΔY, H2; Z=Y)

    ΔA2 = ReLUgrad(ΔH2, A2)
    CB.A2 !== nothing ? ((ΔC2, _) = CB.A2.backward(ΔA2, A2)) : (ΔC2 = ΔA2)
    ΔH1 = CB.CL2.backward(ΔC2, H1)

    ΔA1 = ReLUgrad(ΔH1, A1)
    CB.A1 !== nothing ? ((ΔC1, _) = CB.A1.backward(ΔA1, A1)) : (ΔC1 = ΔA1)
    ΔX = CB.CL1.backward(ΔC1, X)
    # ΔX = ΔY+CB.CL1.backward(ΔC1, X)

    return ΔX

end

function clear_grad!(CB::ConvolutionalBlock)
    clear_grad!(CB.CL1)
    CB.A1 !== nothing && clear_grad!(CB.A1)
    clear_grad!(CB.CL2)
    CB.A2 !== nothing && clear_grad!(CB.A2)
    clear_grad!(CB.CL3)
end

function get_params(CB::ConvolutionalBlock)
    p = get_params(CB.CL1)
    CB.A1 !== nothing && (p = cat(p, get_params(CB.A1); dims=1))
    p = cat(p, get_params(CB.CL2); dims=1)
    CB.A2 !== nothing && (p = cat(p, get_params(CB.A2); dims=1))
    p = cat(p, get_params(CB.CL3); dims=1)
    return p
end

gpu(CB::ConvolutionalBlock{T}) where T = ConvolutionalBlock{T}(gpu(CB.CL1),gpu(CB.A1),gpu(CB.CL2),gpu(CB.A2),gpu(CB.CL3))
cpu(CB::ConvolutionalBlock{T}) where T = ConvolutionalBlock{T}(cpu(CB.CL1),cpu(CB.A1),cpu(CB.CL2),cpu(CB.A2),cpu(CB.CL3))