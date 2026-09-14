using DistributionsParameterSpaces
using Documenter

DocMeta.setdocmeta!(DistributionsParameterSpaces, :DocTestSetup, :(using DistributionsParameterSpaces); recursive=true)

makedocs(;
    modules=[DistributionsParameterSpaces],
    authors="Oskar Laverny <oskar.laverny@univ-amu.fr> and contributors",
    sitename="DistributionsParameterSpaces.jl",
    format=Documenter.HTML(;
        canonical="https://lrnv.github.io/DistributionsParameterSpaces.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lrnv/DistributionsParameterSpaces.jl",
    devbranch="main",
)
