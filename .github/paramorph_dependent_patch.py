from pathlib import Path
import textwrap

wf = Path('.github/workflows/tmp-dependent-spaces.yml').read_text()
source = wf.split("cat > src/DependentSpaces.jl <<'EOF'\n", 1)[1].split("\n          EOF", 1)[0]
source = textwrap.dedent(source)

old_struct = '''struct DependentProduct{T<:Tuple} <: AbstractParameterSpace
    spaces::T
end'''
new_struct = '''struct DependentProduct{T<:Tuple} <: AbstractParameterSpace
    spaces::T

    function DependentProduct{T}(spaces::T) where {T<:Tuple}
        all(q -> q isa AbstractParameterSpace, spaces) ||
            throw(ArgumentError("all dependent-product entries must be parameter spaces"))
        seen = Set{Symbol}()
        for q in spaces
            if q isa AbstractDependentScalarSpace
                ref = _dependent_reference(q)
                ref in seen || throw(ArgumentError(
                    "dependent parameter $(only(names(q))) references $ref before it is defined",
                ))
            end
            for name in names(q)
                name in seen && throw(ArgumentError(
                    "dependent products require unique logical parameter names; duplicate $name",
                ))
                push!(seen, name)
            end
        end
        return new{T}(spaces)
    end
end'''
assert old_struct in source
source = source.replace(old_struct, new_struct, 1)

old_outer = '''function DependentProduct(spaces::Tuple)
    all(q -> q isa AbstractParameterSpace, spaces) ||
        throw(ArgumentError("all dependent-product entries must be parameter spaces"))
    seen = Set{Symbol}()
    for q in spaces
        if q isa AbstractDependentScalarSpace
            ref = _dependent_reference(q)
            ref in seen || throw(ArgumentError(
                "dependent parameter $(only(names(q))) references $ref before it is defined",
            ))
        end
        for name in names(q)
            name in seen && throw(ArgumentError(
                "dependent products require unique logical parameter names; duplicate $name",
            ))
            push!(seen, name)
        end
    end
    return DependentProduct{typeof(spaces)}(spaces)
end

DependentProduct(spaces::AbstractParameterSpace...) = DependentProduct(spaces)'''
new_outer = '''DependentProduct(spaces::Tuple) = DependentProduct{typeof(spaces)}(spaces)
DependentProduct(spaces::AbstractParameterSpace...) = DependentProduct(spaces)'''
assert old_outer in source
source = source.replace(old_outer, new_outer, 1)
Path('src/DependentSpaces.jl').write_text(source)

p = Path('src/Paramorph.jl')
s = p.read_text()
old = 'export Id, Pos, NonNeg, Neg, Prob, ProbOpen, ProbOpenLeft, ProbOpenRight, Lower, LowerClosed, Bounded, BoundedOpen, BoundedOpenLeft, BoundedOpenRight, Ordered, PosOrdered, Between, Simplex, SPD, Prefixed, PosVec, NonNegVec, LowerClosedVec, ProbVec, RealVec, RealMat, Correlation'
assert old in s
s = s.replace(old, old + ', GreaterThan, LowerThan, DependentProduct', 1)
marker = '\n\nend # module'
assert marker in s
s = s.replace(marker, '\n\ninclude("DependentSpaces.jl")' + marker, 1)
p.write_text(s)

tests = wf.split("cat >> test/runtests.jl <<'EOF'\n", 1)[1].split("\n          EOF", 1)[0]
tests = textwrap.dedent(tests)
tests = tests.replace('@test ξ ≈ (0.5, 0.25)', '@test ξ == (0.5, 0.25)')
with Path('test/runtests.jl').open('a') as io:
    io.write('\n' + tests + '\n')
