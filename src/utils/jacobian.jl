# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Jacobian utilities

export Jacobian, Jacobian!

function Jacobian(Net::Union{InvertibleNetwork,NeuralNetLayer}, X::Array{Float32,N}; Y::Union{Nothing,Array{Float32,N}}=nothing, opt_output=false) where N

    # Domain and range data types
    DDT = Tuple{Array{Float32,N}, Array{Parameter,1}}
    RDT = Array{Float32,N}

    # Clean output option
    if !opt_output

        # Forward evaluation
        fop1(input::Tuple{Array{Float32,N}, Array{Parameter,1}}) = Net.jacobian(input[1], input[2], X)[1]
        fop1(Δθ::Array{Parameter,1}) = Net.jacobian(zeros(Float32, size(X)), Δθ, X)[1]

        # Adjoint evaluation
        if isnothing(Y) # recomputing Y=f(X) if not provided
            Y = Net.forward(X)
            isa(Y, Tuple) && (Y = Y[1])
        end
        fop1_adj(ΔY::Array{Float32,N}) = Net.adjointJacobian(ΔY, Y)[1:2]

        return InvertibleNetworkLinearOperator{RDT,DDT}(fop1, fop1_adj)

    else

        # Forward evaluation
        fop(input::Tuple{Array{Float32,N}, Array{Parameter,1}}) = Net.jacobian(input[1], input[2], X)
        fop(Δθ::Array{Parameter,1}) = Net.jacobian(zeros(Float32, size(X)), Δθ, X)
        fop(input::Tuple{Array{Float32,N}, Array{Parameter,1}, Array{Float32,N}, Array{Parameter,1}}) = Net.jacobian(input[1], input[2], X) # when input = full output from adjoint eval (logdet=true)
        fop(input::Tuple{Array{Float32,N}, Array{Parameter,1}, Array{Float32,N}}) = Net.jacobian(input[1], input[2], X) # when input = full output from adjoint eval (logdet=false)

        # Adjoint evaluation
        if isnothing(Y) # recomputing Y=f(X) if not provided
            Y = Net.forward(X)
            isa(Y, Tuple) && (Y = Y[1])
        end
        fop_adj(ΔY::Array{Float32,N}) = Net.adjointJacobian(ΔY, Y)
        fop_adj(input::Tuple{Array{Float32,N}, Array{Float32,N}, Float32, Array{Parameter,1}}) = Net.adjointJacobian(input[1], input[2]) # when input = full output from forward eval (logdet=true)
        fop_adj(input::Tuple{Array{Float32,N}, Array{Float32,N}}) = Net.adjointJacobian(input[1], input[2]) # when input = full output from forward eval (logdet=true)

        return InvertibleNetworkLinearOperator{RDT,DDT}(fop, fop_adj)

    end

end

function Jacobian!(Net::Union{InvertibleNetwork,NeuralNetLayer}, X::Array{Float32,N}; θ::Union{Nothing,Array{Parameter,1}}=nothing, Y::Union{Nothing,Array{Float32,N}}=nothing, opt_output=false) where N

    # Set parameters if provided
    if !isnothing(θ)
        set_params!(Net, θ)
    end

    # Domain and range data types
    DDT = Tuple{Array{Float32,N}, Array{Parameter,1}}
    RDT = Array{Float32,N}

    # Clean output option
    if !opt_output

        # Forward evaluation
        fop1(input::Tuple{Array{Float32,N}, Array{Parameter,1}}) = Net.jacobian(input[1], input[2], X)[1]
        fop1(Δθ::Array{Parameter,1}) = Net.jacobian(zeros(Float32, size(X)), Δθ, X)[1]

        # Adjoint evaluation
        if isnothing(Y) # recomputing Y=f(X) if not provided
            Y = Net.forward(X)
            isa(Y, Tuple) && (Y = Y[1])
        end
        fop1_adj(ΔY::Array{Float32,N}) = Net.adjointJacobian(ΔY, Y)[1:2]

        return InvertibleNetworkLinearOperator{RDT,DDT}(fop1, fop1_adj)

    else

        # Forward evaluation
        fop(input::Tuple{Array{Float32,N}, Array{Parameter,1}}) = Net.jacobian(input[1], input[2], X)
        fop(Δθ::Array{Parameter,1}) = Net.jacobian(zeros(Float32, size(X)), Δθ, X)
        fop(input::Tuple{Array{Float32,N}, Array{Parameter,1}, Array{Float32,N}, Array{Parameter,1}}) = Net.jacobian(input[1], input[2], X) # when input = full output from adjoint eval (logdet=true)
        fop(input::Tuple{Array{Float32,N}, Array{Parameter,1}, Array{Float32,N}}) = Net.jacobian(input[1], input[2], X) # when input = full output from adjoint eval (logdet=false)

        # Adjoint evaluation
        if isnothing(Y) # recomputing Y=f(X) if not provided
            Y = Net.forward(X)
            isa(Y, Tuple) && (Y = Y[1])
        end
        fop_adj(ΔY::Array{Float32,N}) = Net.adjointJacobian(ΔY, Y)
        fop_adj(input::Tuple{Array{Float32,N}, Array{Float32,N}, Float32, Array{Parameter,1}}) = Net.adjointJacobian(input[1], input[2]) # when input = full output from forward eval (logdet=true)
        fop_adj(input::Tuple{Array{Float32,N}, Array{Float32,N}}) = Net.adjointJacobian(input[1], input[2]) # when input = full output from forward eval (logdet=true)

        return InvertibleNetworkLinearOperator{RDT,DDT}(fop, fop_adj)

    end

end

# Special input case
function *(J::InvertibleNetworkLinearOperator{Array{Float32,N},Tuple{Array{Float32,N}, Array{Parameter,1}}}, Δθ::Array{Parameter,1}) where N
    return J.fop(Δθ)
end
