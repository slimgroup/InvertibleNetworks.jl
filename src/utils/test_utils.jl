export grad_test_input, grad_test_input_loss

using Printf, Random, LinearAlgebra, Test

function grad_test_input(L, X::AbstractArray{T}, loss::Function; rtol::T=nothing) where T
	hasfield(typeof(L), :logdet) ? (logdet = L.logdet) : (logdet = false)
	isnothing(rtol) && (rtol = sqrt(eps(T)))
	
	# Generate arbitrary starting point X0
	X0 = randn(T, size(X));

	# Generate objective Y
	logdet ? ((Y, _) = L.forward(X)) : (Y = L.forward(X))

	# Use loss function with grad=true to get gradient 
	f_init, ΔX = loss(L, X0, Y; grad=true)
	dX = T.(ΔX .* ( 1 .+ rand(size(ΔX))/64))
	gdx = dot(dX, ΔX)

	# Gradient test with 1st and 2nd order linearization error
	h = T(1f0)
	h_init = h
	h_step = T(sqrt(2))

	maxiter = 7
	err1 = zeros(T, maxiter)
	err2 = zeros(T, maxiter)

	@printf("%8s %8s %8s %8s %8s %8s \n", "h", "f_init", "fk", "e1", "h*gdx", "e2")
	for k=1:maxiter
	    fk = loss(L, X0 + h*dX, Y; grad=false)
	    err1[k] = abs(fk - f_init)
	    err2[k] = abs(fk - f_init - h*gdx)
	    @printf("%2.2e %2.2e %2.2e %2.2e %2.2e %2.2e \n", h, f_init, fk, fk-f_init, h*gdx, fk - f_init - h*gdx)
	    h = h / h_step
	end
	
	h = h_init / h_step
	@printf("%8s %8s %8s\n", "h", "p1", "p2")
	for i in 2:maxiter
	    @printf("%2.2e %2.2e %2.2e \n",h, err1[i-1]/err1[i], err2[i-1]/err2[i])
	    h = h / h_step
	end

	# First test average of slopes
	rate_1 = sum(err1[1:end-1]./err1[2:end])/(maxiter - 1);
	rate_2 = sum(err2[1:end-1]./err2[2:end])/(maxiter - 1);
	
	@test isapprox(rate_1, h_step; rtol=rtol)
	@test isapprox(rate_2, h_step^2; rtol=rtol)

	# Then test individual slopes with larger tol
	rates_1 = err1[1:end-1]./err1[2:end]
	rates_2 = err2[1:end-1]./err2[2:end]
	
	@test all(isapprox.(rates_1, h_step; rtol=2*rtol))
	@test all(isapprox.(rates_2, h_step^2; rtol=2*rtol))
end

function grad_test_input_loss(L, X, Y; grad::Bool=true)
	hasfield(typeof(L), :logdet)     ? (logdet     = L.logdet)     : (logdet     = false)
	hasfield(typeof(L), :invertible) ? (invertible = L.invertible) : (invertible = true)

	logdet ? (Y_, logdet_val) = L.forward(X) : (Y_ = L.forward(X))

    f = mse(Y_, Y)
    logdet && (f -= logdet_val)

    # Just pass back f
    !grad && return f

    ΔY = ∇mse(Y_, Y) 
    invertible ? ((ΔX, _) = L.backward(ΔY, Y_)) : (ΔX = L.backward(ΔY, X)) 

    # Pass back f and gradients w.r.t. input X
    return f, ΔX
end
