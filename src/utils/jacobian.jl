# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Jacobian utilities

export Jacobian, JacobianAdjoint
import Base.*, LinearAlgebra.adjoint

struct Jacobian
    N::Union{InvertibleNetwork,NeuralNetLayer}
    X::Array{Float32}
    out_std::Bool
end

function Jacobian(N::Union{InvertibleNetwork,NeuralNetLayer}, X::Array{Float32})
    return Jacobian(N, X, true)
end

function *(J::Jacobian, input::Tuple{Array{Float32}, Array{Parameter, 1}})
    J.out_std ? (return J.N.jacobian_forward(input[1], input[2], J.X)[1]) : (return J.N.jacobian_forward(input[1], input[2], J.X))
end

function *(J::Jacobian, input::Array{Parameter, 1})
    return J*(zeros(Float32, size(J.X)), input)
end

mutable struct JacobianAdjoint
    J::Jacobian
    Y::Union{Nothing, Array{Float32}}
end

function adjoint(J::Jacobian; Y::Union{Nothing, Array{Float32}}=nothing)
    return JacobianAdjoint(J, Y)
end

function *(JT::JacobianAdjoint, ΔY::Array{Float32}; Y::Union{Nothing, Array{Float32}}=nothing)
    if JT.Y != nothing
        JT.J.out_std ? (return JT.J.N.jacobian_backward(ΔY, JT.Y)[1:2]) : (return JT.J.N.jacobian_backward(ΔY, JT.Y))
    elseif Y == nothing
        JT.J.N.logdet ? (JT.Y = JT.J.N.forward(JT.J.X)[1]) : (JT.Y = JT.J.N.forward(JT.J.X))
    end
    return JT*ΔY
end
