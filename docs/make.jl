using ParameterSpaces
using Documenter

DocMeta.setdocmeta!(ParameterSpaces, :DocTestSetup, :(using ParameterSpaces); recursive=true)

makedocs(;
    modules=[ParameterSpaces],
    authors="Oskar Laverny <oskar.laverny@univ-amu.fr> and contributors",
    sitename="ParameterSpaces.jl",
    format=Documenter.HTML(;
        canonical="https://lrnv.github.io/ParameterSpaces.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lrnv/ParameterSpaces.jl",
    devbranch="main",
)
