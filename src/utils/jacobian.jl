# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Jacobian utilities

export Jacobian

function Jacobian(Net::Union{InvertibleNetwork,NeuralNetLayer}, X::Array{Float32,N}; Y::Union{Nothing,Array{Float32,N}}=nothing) where N

    # Domain and range data types
    DDT = Tuple{Array{Float32,N}, Array{Parameter,1}}
    RDT = Array{Float32,N}

    # Forward evaluation
    fop(ΔX::Array{Float32,N}, Δθ::Array{Parameter,1}) = Net.jacobian_forward(ΔX, Δθ, X)[1]
    fop(Δθ::Array{Parameter,1}) = Net.jacobian_forward(zeros(Float32, size(X)), Δθ, X)[1]

    # Adjoint evaluation
    if Y == nothing # recomputing Y=f(X) if not provided
        Y = N.forward(X)
        isa(Y, Tuple) && (Y = Y[1])
    end
    fop_adj(ΔY::Array{Float32,N}) = Net.jacobian_backward(ΔY, Y)

    return InvertibleNetworkLinearOperator{RDT,DDT}(fop, fop_adj)

end

# Special case for coupling layer
function Jacobian(Net::CouplingLayerBasic, X1::Array{Float32,N}, X2::Array{Float32,N}; Y1::Union{Nothing,Array{Float32,N}}=nothing, Y2::Union{Nothing,Array{Float32,N}}=nothing) where N

    # Domain and range data types
    DDT = Tuple{Array{Float32,N}, Array{Parameter,1}}
    RDT = Array{Float32,N}

    # Forward evaluation
    fop(ΔX1::Array{Float32,N}, ΔX2::Array{Float32,N}, Δθ::Array{Parameter,1}) = Net.jacobian_forward(ΔX1, ΔX2, Δθ, X1, X2)[1:2]
    fop(Δθ::Array{Parameter,1}) = Net.jacobian_forward(zeros(Float32, size(X)), zeros(Float32, size(X)), Δθ, X1, X2)[1:2]

    # Adjoint evaluation
    if Y == nothing # recomputing Y=f(X) if not provided
        Y1, Y2 = N.forward(X1, X2)
        isa(Y2, Tuple) && (Y2 = Y2[1])
    end
    fop_adj(ΔY1::Array{Float32,N}, ΔY2::Array{Float32,N}) = Net.jacobian_backward(ΔY1, ΔY2, Y1, Y2)

    return InvertibleNetworkLinearOperator{RDT,DDT}(fop, fop_adj)

end

# Special input case
function *(J::InvertibleNetworkLinearOperator{Array{Float32,N},Tuple{Array{Float32,N}, Array{Parameter,1}}}, Δθ::Array{Parameter,1}) where N
    return J.fop(Δθ)
end
