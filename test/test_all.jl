using InvertibleNetworks, LinearAlgebra, Test, Random, Printf

function grad_test_input(L, X::AbstractArray{T}, loss::Function; rtol::T=nothing) where T
	hasfield(typeof(L), :logdet) ? (logdet = L.logdet) : (logdet = false)
	isnothing(rtol) && (rtol = sqrt(eps(T)))

	logdet ? ((Y, _) = L.forward(X)) : (Y = L.forward(X))

	X0   = randn(T, size(X));
	dX   = X - X0 
	#dX .*= norm(X) / norm(dX)
	dX .*= 1 / norm(X0)

	f_init, ΔX = loss(L, X0, Y; grad=true)
	gdx = dot(dX, ΔX)

	h = 5f-1
	maxiter = 7
	err1 = zeros(Float32, maxiter)
	err2 = zeros(Float32, maxiter)

	@printf("%8s %8s %8s %8s %8s %8s \n", "h", "f_init", "fk", "e1", "h*gdx", "e2")
	for k=1:maxiter
	    fk = loss(L, X0 + h*dX, Y; grad=false)
	    err1[k] = abs(fk - f_init)
	    err2[k] = abs(fk - f_init - h*gdx)
	    @printf("%2.2e %2.2e %2.2e %2.2e %2.2e %2.2e \n", h, f_init, fk, fk-f_init, h*gdx, fk - f_init - h*gdx)
	    global h = h/2f0
	end
	
	global h = 2.5f-1
	@printf("%8s %8s %8s\n", "h", "p1", "p2")
	for i in 2:maxiter
	    @printf("%2.2e %2.2e %2.2e \n",h, err1[i-1]/err1[i], err2[i-1]/err2[i])
	    global h = h / 2
	end

	rate_1 = sum(err1[1:end-1]./err1[2:end])/(maxiter - 1);
	rate_2 = sum(err2[1:end-1]./err2[2:end])/(maxiter - 1);

	@test isapprox(rate_1, 2f0; rtol=rtol)
	@test isapprox(rate_2, 4f0; rtol=rtol)
end

function grad_test_input_loss(L, X, Y; grad::Bool=true)
	hasfield(typeof(L), :logdet)     ? (logdet     = L.logdet)     : (logdet     = false)
	hasfield(typeof(L), :invertible) ? (invertible = L.invertible) : (invertible = true)
	logdet ? (Y_, logdet_val) = L.forward(X) : (Y_ = L.forward(X))

    f = mse(Y_, Y)
    logdet && (f -= logdet_val)

    !grad && return f

    ΔY = ∇mse(Y_, Y) 
    invertible ? ((ΔX, _) = L.backward(ΔY, Y_)) : (ΔX = L.backward(ΔY, X)) 

    # Pass back f and gradients w.r.t. input X
    return f, ΔX
end


# Random seed
Random.seed!(11)

###################################################################################################
for nx in [8,16,32]
	for n_in in [2,4,8]
		for batch_size in [1,2,4]
			for n_hidden in [8,16,32]
				for logdet in [true]
					# Input images
					print("\nCouplingLayerGlow gradient test wrt input | nx=$(nx) n_in=$(n_in) n_hidden=$(n_hidden) logdet=$(logdet)\n")
		    
					X = randn(Float32, nx, nx, n_in, batch_size)
					L = CouplingLayerGlow(n_in, n_hidden; logdet=logdet)
					grad_test_input(L, grad_test_input_loss, X; rtol=5f-2)
				end
			end
		end
	end
end


for nx in [28,16,32]
	for n_in in [4,8]
		for batch_size in [2,4]
			for n_hidden in [8,16,32]
				for k in [3,4]
					for p in [1,2]
						for s in [1,2,3,4]
							for fan in [false]
								# Input images
								print("\nResidualBlock gradient test wrt input | nx=$(nx) n_in=$(n_in)",
									" batch_size=$(batch_size) n_hidden=$(n_hidden) k=$(k) ",
									" p=$(p) s=$(s) fan=$(fan) \n")
					    
								X = randn(Float32, nx, nx, n_in, batch_size)
								L = ResidualBlock(n_in, n_hidden; k1=k, k2=k, p1=p, p2=p, s1=s, s2=s, fan=fan)
								grad_test_input(L, X, grad_test_input_loss; rtol=5f-2)
							end
						end
					end
				end
			end
		end
	end
end
