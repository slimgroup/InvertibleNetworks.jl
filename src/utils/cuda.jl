using CUDA

export convert_cu, cuzeros, cuones

convert_cu(in_a, X) =  X isa CuArray ? cu(in_a) : in_a

cuzeros(X, args...) = X isa CuArray ? CuArray.fill(0f0, args) : zeros(Float32, args)
cuones(X, args...) = X isa CuArray ? CuArray.fill(1f0, args) : ones(Float32, args)
