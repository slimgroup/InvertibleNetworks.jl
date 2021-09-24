# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Jacobian utilities

export Jacobian


## Jacobian types

struct JacobianInvertibleNetwork{T} <: joAbstractLinearOperator{T, T}
    name::String
    n::Int64
    m::Int64
    fop::Function
    fop_T::Function
    N::Union{NeuralNetLayer, InvertibleNetwork}
    X::AbstractArray{T}
    Y::Union{Nothing, AbstractArray{T}}
end

struct JacobianInvertibleNetworkAdjoint{T} <: joAbstractLinearOperator{T, T}
    name::String
    n::Int64
    m::Int64
    fop::Function
    fop_T::Function
    N::Union{NeuralNetLayer, InvertibleNetwork}
    X::AbstractArray{T}
    Y::Union{Nothing, AbstractArray{T}}
end


## Constructor

function Jacobian(N::Union{InvertibleNetwork, NeuralNetLayer}, X::AbstractArray{T};
                  Y::Union{Nothing, AbstractArray{T}}=nothing, save_Y::Bool=true,
                  io_mode::String="θ↦Y", name::String="Jacobian") where T

    # Computing & storing Y=f(X) if requested
    if save_Y && isnothing(Y)
        Y = N.forward(X)
        isa(Y, Tuple) && (Y = Y[1])
    end

    # Input/output dimensions
    m = length(X)
    n = length(par2vec(get_params(N))[1])
    ((io_mode == "X×θ↦Y") || (io_mode == "X×θ↦Y")) && (n += m)

    # Forward evaluation
    function fop(ΔX::Union{Nothing, AbstractArray{T}}, Δθ::Array{Parameter, 1}) where T
        isnothing(ΔX) && (ΔX = cuzeros(X, size(X)...))
        out = N.jacobian(ΔX, Δθ, X)
        ((io_mode == "θ↦Y") || (io_mode == "X×θ↦Y")) && (return out[1])
        io_mode == "full"                            && (return out)
    end

    # Adjoint evaluation
    function fop_adj(ΔY::AbstractArray{T}; Y::Union{Nothing, AbstractArray{T}}=Y) where T
        if isnothing(Y)
            Y = N.forward(X)
            isa(Y, Tuple) && (Y = Y[1])
        end
        out = N.adjointJacobian(ΔY, Y)
        io_mode == "θ↦Y"   && (return out[2])
        io_mode == "X×θ↦Y" && (return out[1:2])
        io_mode == "full"  && (return out)
    end

    # Jacobian JOLI
    joJ = joLinearFunctionFwd_T(n, m, fop, fop_adj, Float32, Float32; name=name)

    # Output
    return JacobianInvertibleNetwork{T}(name, n, m, fop, fop_adj, N, X, Y)

end


## Algebra

function adjoint(J::JacobianInvertibleNetwork{T}) where T
    return JacobianInvertibleNetworkAdjoint{T}(string("adjoint(", J.name, ")"), J.m, J.n, J.fop_T, J.fop, J.N, J.X, J.Y)
end

function adjoint(JT::JacobianInvertibleNetworkAdjoint{T}) where T
    return JacobianInvertibleNetwork{T}(JT.name[9:end-1], JT.m, JT.n, JT.fop_T, JT.fop, JT.N, JT.X, JT.Y)
end

*(J::JacobianInvertibleNetwork{T}, Δθ::Array{Parameter,1}) where T = J.fop(nothing, Δθ)

function *(J::JacobianInvertibleNetwork{T}, input::Tuple{AbstractArray{T2}, Array{Parameter,1}}) where {T, T2}
    return J.fop(jo_convert(T, input[1], false), input[2])
end

function *(JT::JacobianInvertibleNetworkAdjoint{T}, ΔY::AbstractArray{T2}) where {T, T2}
    return JT.fop(jo_convert(T, ΔY, false))
end