# Test memory usage of InvertibleNetworks as network depth increases


using InvertibleNetworks, LinearAlgebra, Flux
import Flux.Optimise.update!

device = InvertibleNetworks.CUDA.functional() ? gpu : cpu

#turn off JULIA cuda optimization to get raw peformance
ENV["JULIA_CUDA_MEMORY_POOL"] = "none"

using CUDA, Printf

export @gpumem

get_mem() = NVML.memory_info(NVML.Device(0)).used/(1024^3)

function montitor_gpu_mem(used::Vector{T}, status::Ref) where {T<:Real}
    while status[]
        #cleanup()
        push!(used, get_mem())
    end
    nothing
end

cleanup() = begin GC.gc(true); CUDA.reclaim(); end

macro gpumem(expr)
    return quote
        # Cleanup
        cleanup()
        monitoring = Ref(true)
        used = [get_mem()]
        Threads.@spawn montitor_gpu_mem(used, monitoring)
        val = $(esc(expr))
        monitoring[] = false
        cleanup()
        @printf("Min memory: %1.3fGiB , Peak memory: %1.3fGiB \n",
                extrema(used)...)
        used
    end
end

# Objective function
function loss(G,X)
    Y, logdet = G.forward(X)
    #cleanup()
    f = .5f0/batchsize*norm(Y)^2 - logdet
    ΔX, X_ = G.backward(1f0./batchsize*Y, Y)
    return f
end

# size of network input 
nx = 256 
ny = nx
n_in = 3
batchsize = 8
X = rand(Float32, nx, ny, n_in, batchsize) |> device

# Define network
n_hidden = 256
L = 3   # number of scales

num_retests=1
mems_max= []
mem_tested = [4,8,16,32,48,64,80]
for K in mem_tested
    G = NetworkGlow(n_in, n_hidden, L, K; split_scales=true)  |> device

    loss(G,X)
    curr_maxes = []
    for i in 1:num_retests
        usedmem = @gpumem begin
            loss(G,X)
        end
        append!(curr_maxes,maximum(usedmem))
    end
    append!(mems_max, minimum(curr_maxes))
    println(mems_max)
end

# control for memory of storing the network parameters on GPU, not relevant to backpropagation
mems_model = []
for K in mem_tested
    G = NetworkGlow(n_in, n_hidden, L, K; split_scales=true) |> device
    G(X)
    G = G |> cpu
    usedmem = @gpumem begin
     G = G |> device
    end

    append!(mems_model, maximum(usedmem)) 
end
mems_model_norm = mems_model .- mems_model[1]


mem_used_invnets = mems_max .- mems_model_norm
mem_used_pytorch = [5.0897216796875, 7.8826904296875, 13.5487060546875, 24.8709716796875, 36.2010498046875, 40, NaN]
mem_ticks = mem_tested


using PyPlot

font_size=15
PyPlot.rc("font", family="serif"); 
PyPlot.rc("font", family="serif", size=font_size); PyPlot.rc("xtick", labelsize=font_size); PyPlot.rc("ytick", labelsize=font_size);
PyPlot.rc("axes", labelsize=font_size)    # fontsize of the x and y labels

#nice plot 
fig = figure(figsize=(14,6))
plot(log.(mem_tested),mem_used_pytorch; color="black", linestyle="--",markevery=collect(range(0,4,step=1)),label="PyTorch package",marker="o",markerfacecolor="r")
plot(log.(mem_tested),mem_used_invnets; color="black", label="InvertibleNetworks.jl package",marker="o",markerfacecolor="b")
axvline(log.(mem_tested)[end-1], linestyle="--",color="red",label="PyTorch out of memory error")
grid()
xticks(log.(mem_ticks), [string(i) for i in mem_ticks],rotation=60)
legend()
xlabel("Network depth [# of layers]");
ylabel("Peak memory used [GB]");
fig.savefig("mem_used_new_depth_new.png", bbox_inches="tight", dpi=400)
