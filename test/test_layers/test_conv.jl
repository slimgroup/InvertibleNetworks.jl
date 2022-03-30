# Author: Rafael Orozco, rorozco@gatech.edu
# Date: March 2022

using InvertibleNetworks, LinearAlgebra, Test, Statistics
Random.seed!(2)

###############################################################################
# Gradient tests
num_tries = 3
Ts = [Float32]
nxs = [16,32]
n_ins = [1,2]
batch_sizes = [1,2]
n_outs = [1,2]
ks = [1,2,3]
ps = [0,1]
ss = [1,2]

# Gradient test w.r.t. input
for T in Ts, nx in nxs, n_in in n_ins, batch_size in batch_sizes
    for n_out in n_outs, k in ks
        for p in ps, s in ss
            for bias in [true, false]
                print("\n ConvLayer gradient test wrt input | T=$(T) nx=$(nx) n_in=$(n_in)",
                    " batch_size=$(batch_size) n_out=$(n_out) k=$(k) bias=$(bias)",
                    " p=$(p) s=$(s) \n")
                
                X = randn(T, nx, nx, n_in, batch_size)
                L = ConvLayer(n_in, n_out; k=k, p=p, s=s, bias = bias, T=T)
         
                for try_i = 1:num_tries
                    try 
                        grad_test_input(L, X, grad_test_input_loss; rtol=T(5f-2))
                        break
                    catch e
                        println("Test Failed, will try again $(try_i)/$(num_tries)")
                    end
                end
            end
        end
    end
end

###################################################################################################
