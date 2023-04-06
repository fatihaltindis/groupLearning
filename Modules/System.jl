# v0.0.1
module System

# functions:

# waste | free the memory for all objects passed as arguments

export
  waste,
  charlie,
  remove,
  isSquare

# free memory for all arguments passed as `args...`
# see https://github.com/JuliaCI/BenchmarkTools.jl/pull/22
function waste(args...)
  for a in args a=nothing end
  for i=1:4 GC.gc(true) end
end

# if b is true, print a warning with the `msg` and return true,
# otherwise return false. This is used within functions
# to make a check and if necessary print a message and return.
# Example: charlie(type ≠ :s && type ≠ :t, "my message") && return
charlie(b::Bool, msg::String; fb::Symbol=:warn) =
  if      fb==:warn  b ? (@warn msg; return true) : return false
  elseif fb==:error b ? (@error msg; return true) : return false
  elseif fb==:info b ? (@info msg; return true) : return false
  end

# Remove one or more elements from a vector or one or more
# columns or rows from a matrix.
# If `X` is a Matrix, `dims`=1 (default) remove rows,
# `dims`=2 remove columns.
# If `X` is a Vector, `dims` has no effect.
# The second argument is either an integer or a vector of integers
# EXAMPLES:
# a=randn(5)
# a=remove(a, 2)
# a=remove(a, collect(1:3)) # remove rows 1 to 3
# A=randn(3, 3)
# A=remove(A, 2)
# A=remove(A, 2; dims=2)
# A=randn(5, 5)
# B=remove(A, collect(1:2:5)) # remove rows 1, 3 and 5
# C=remove(A, [1, 4])
# A=randn(10, 10)
# A=remove(A, [collect(2:3); collect(8:10)]; dims=2)
function remove(X::Union{Vector, Matrix}, what::Union{Int, Vector{Int}}; dims=1)
    1<dims<2 && throw(ArgumentError("function `remove`: the `dims` keyword argument must be 1 or 2"))
    di = X isa Vector ? 1 : dims
    d = size(X, di)
    mi, ma = minimum(what), maximum(what)
    (1≤mi≤d && 1≤ma≤d) || throw(ArgumentError("function `remove`: the second argument must holds elements comprised in between 1 and $d. Check also the `dims` keyword"))
    b = filter(what isa Int ? x->x≠what : x->x∉what, 1:d)
    return X isa Vector ? X[b] : X[di==1 ? b : 1:end, di==2 ? b : 1:end]
end

isSquare(X::Matrix)=size(X, 1)==size(X, 2)

end
