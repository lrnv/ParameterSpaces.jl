using Paramoprh
using Documenter

DocMeta.setdocmeta!(Paramoprh, :DocTestSetup, :(using Paramoprh); recursive=true)

makedocs(;
    modules=[Paramoprh],
    authors="Oskar Laverny <oskar.laverny@univ-amu.fr> and contributors",
    sitename="Paramoprh.jl",
    format=Documenter.HTML(;
        canonical="https://lrnv.github.io/Paramoprh.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lrnv/Paramoprh.jl",
    devbranch="main",
)
