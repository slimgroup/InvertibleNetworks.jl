
## Invertible Layers

### Types

```@autodocs
Modules = [InvertibleNetworks]
Order  = [:type]
Filter = t -> t<:NeuralNetLayer
```

## Invertible Networks

### Types

```@autodocs
Modules = [InvertibleNetworks]
Order   = [:type]
Filter = t -> t<:InvertibleNetwork
```

## Activations functions

```@autodocs
Modules = [InvertibleNetworks]
Order   = [:function]
Pages = ["activation_functions.jl"]
```

## Dimensions manipulation

```@autodocs
Modules = [InvertibleNetworks]
Order   = [:function]
Pages = ["dimensionality_operations.jl"]
```