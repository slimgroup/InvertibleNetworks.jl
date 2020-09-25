# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Invertible network specific linear operator type

export InvertibleNetworkLinearOperator

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
