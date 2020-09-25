# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Jacobian utilities

export InvertibleNetworkLinearOperator#, solveGaussNewton


## Invertible network specific linear operator type

# Type
struct InvertibleNetworkLinearOperator{RDT,DDT}
    fop::Function
    fop_adj::Function
end

# Adjoint
function adjoint(A::InvertibleNetworkLinearOperator{RDT,DDT}) where {RDT,DDT}
    return InvertibleNetworkLinearOperator{DDT,RDT}(A.fop_adj, A.fop)
end

# Mat-vec product
function *(A::InvertibleNetworkLinearOperator{RDT,DDT}, x::DDT) where {RDT,DDT}
    return A.fop(x)
end

# Mat-mat product
function *(A::InvertibleNetworkLinearOperator{DT1,DT2}, B::InvertibleNetworkLinearOperator{DT2,DT3}) where {DT1,DT2,DT3}
    return InvertibleNetworkLinearOperator{DT1,DT3}(x -> A.fop(B.fop(x)), y -> B.fop_adj(A.fop_adj(y)))
end

# Constructor
function JacobianInvNet(N::Union{InvertibleNetwork, NeuralNetLayer}, X::Array{Float32}; Y::Union{Nothing, Array{Float32}}=nothing)

    n = length(X)
    m = 0
    for p in get_params(N)
        m += length(p)
    end
    fop(ΔX, Δθ) = N.jacobian_forward(ΔX, Δθ, X)
    if Y == nothing # recomputing Y=f(X) if not provided
        Y = N.forward(X)
        isa(Y, Tuple) && (Y = Y[1])
    end
    fop_T(ΔY) = N.jacobian_backward(ΔY, Y)
    joLF = joLinearFunctionFwd_T(n, m, fop, fop_T, Float32, Float32; name="Jacobian")

    return JacobianInvNet(N, X, Y, joLF)

end

# Utilities
function size(J::JacobianInvNet)
    return size(J.joLF)
end
function show(J::JacobianInvNet)
    return show(J.joLF)
end

# Adjoint
function adjoint(J::JacobianInvNet)
    return adjoint(J.joLF)
end

# Algebra
function *(J::JacobianInvNet, input::Tuple{Array{Float32, 4}, Array{Parameter, 1}})
    return J.joLF(input)
end
function *(J::JacobianInvNet, Δθ::Array{Parameter, 1})
    return J.joLF.fop(zeros(Float32, size(J.X)), Δθ)
end


## Gauss-Newton solve utilities

# function solveGaussNewton(J::Jacobian, JT::JacobianAdjoint, b::Array{Float32})
#     ;
# end
