# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Jacobian utilities

export Jacobian


## Jacobian types

struct JacobianInvertibleNetwork <: joAbstractLinearOperator{Float32, Float32}
    name::String
    n::Int64
    m::Int64
    fop::Function
    fop_T::Function
    N::Union{NeuralNetLayer, InvertibleNetwork}
    X::Array{Float32}
    Y::Union{Nothing, Array{Float32}}
end

struct JacobianInvertibleNetworkAdjoint <: joAbstractLinearOperator{Float32, Float32}
    name::String
    n::Int64
    m::Int64
    fop::Function
    fop_T::Function
    N::Union{NeuralNetLayer, InvertibleNetwork}
    X::Array{Float32}
    Y::Union{Nothing, Array{Float32}}
end


## Constructor

function Jacobian(N::Union{InvertibleNetwork, NeuralNetLayer}, X::Array{Float32}; Y::Union{Nothing, Array{Float32}}=nothing, save_Y::Bool=true, io_mode::String="θ↦Y", name::String="Jacobian")

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
    function fop(ΔX::Union{Nothing, Array{Float32}}, Δθ::Array{Parameter, 1})
        isnothing(ΔX) && (ΔX = cuzeros(X, size(X)...))
        out = N.jacobian(ΔX, Δθ, X)
        ((io_mode == "θ↦Y") || (io_mode == "X×θ↦Y")) && (return out[1])
        io_mode == "full"                            && (return out)
    end

    # Adjoint evaluation
    function fop_adj(ΔY::Array{Float32}; Y::Union{Nothing, Array{Float32}}=Y)
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
    return JacobianInvertibleNetwork(name, n, m, fop, fop_adj, N, X, Y)

end


## Algebra

function adjoint(J::JacobianInvertibleNetwork)
    return JacobianInvertibleNetworkAdjoint(string("adjoint(", J.name, ")"), J.m, J.n, J.fop_T, J.fop, J.N, J.X, J.Y)
end

function adjoint(JT::JacobianInvertibleNetworkAdjoint)
    return JacobianInvertibleNetwork(JT.name[9:end-1], JT.m, JT.n, JT.fop_T, JT.fop, JT.N, JT.X, JT.Y)
end

function *(J::JacobianInvertibleNetwork, Δθ::Array{Parameter,1})
    return J.fop(nothing, Δθ)
end

function *(J::JacobianInvertibleNetwork, input::Tuple{Array{Float32}, Array{Parameter,1}})
    return J.fop(input[1], input[2])
end

function *(JT::JacobianInvertibleNetworkAdjoint, ΔY::Array{Float32})
    return JT.fop(ΔY)
end