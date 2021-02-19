using Documenter, InvertibleNetworks

makedocs(sitename="Invertible Networks",
         doctest=false, clean=true,
         authors="Philipp Witte, Ali Siahkoohi, Gabbrio Rizzuti, Mathias Louboutin, Felix J. Herrmann",
         format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
         pages = Any[
             "Home" => "index.md",
             "Examples" => "examples.md",
             "API Reference" => "api.md",
             "LICENSE" => "LICENSE.md",
         ])

deploydocs(repo="github.com/slimgroup/InvertibleNetworks.jl")
