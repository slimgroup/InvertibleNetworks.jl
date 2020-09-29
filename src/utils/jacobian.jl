# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Jacobian utilities

export Jacobian

function Jacobian(N::Union{InvertibleNetwork,NeuralNetLayer}, X::Array{Float32,4}; Y::Union{Nothing,Array{Float32,4}}=nothing)

    # Domain and range data types
    DDT = Tuple{Array{Float32,4}, Array{Parameter,1}}
    RDT = Array{Float32,4}

    # Forward evaluation
    fop(Δ::Tuple{Array{Float32,4}, Array{Parameter,1}}) = N.jacobian_forward(Δ[1], Δ[2], X)[1]
    fop(Δθ::Array{Parameter,1}) = N.jacobian_forward(zeros(Float32, size(X)), Δθ, X)[1]

    # Adjoint evaluation
    if Y == nothing # recomputing Y=f(X) if not provided
        Y = N.forward(X)
        isa(Y, Tuple) && (Y = Y[1])
    end
    fop_adj(ΔY::Array{Float32,4}) = N.jacobian_backward(ΔY, Y)

    return InvertibleNetworkLinearOperator{RDT,DDT}(fop, fop_adj)

end

# Special input case
function *(J::InvertibleNetworkLinearOperator{Array{Float32,4},Tuple{Array{Float32,4}, Array{Parameter,1}}}, Δθ::Array{Parameter,1})
    return J.fop(Δθ)
end
