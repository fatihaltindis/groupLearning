module EEGprocessing

using StatsBase, Statistics, LinearAlgebra, PosDefManifold

# functions:

export
    evar,
    normalizeMean!



## compute tr(A[i, :]*(B[:, i]'*C*B[:, i])*A[i, :]')
# This is the expected variance for the ith source estimated
# on covariance matrix `C` given mixing (`A`) and demixing (`B`) matrices
function evar(A, B, C, i)
      e=PosDefManifold.quadraticForm(B[:, i], C)
      return sum(A[i, j]^2*e for j=1:size(A, 2))
end

# all expected variances for i=1 size(A, 1)
evar(A, B, C)=[evar(A, B, C, i) for i=1:size(A, 1)]

# same as evar but normalizing the `A` and `B` vectors first
function evarn(A, B, C, i)
      b=B[:, i]
      b=b/norm(b)
      e=qf(b, C)
      a=A[i, :]
      a=a/norm(a)
      return sum(e*(a.^2) )
end

evarn(A, B, C)=[evarn(A, B, C, i) for i=1:size(A, 1)]




## Mean Normalization of real matrix `X`.
# `dims=0` does not do anything (no normalization).
# `dims`=1 normalise the columns, `dims`=2 normalize the rows (default).
# With `dims`=3 normalize iteratively both the columns and the rows
# (complete standardization).
# `maxiter` and `tol` are the maximum number of iterations and the tolerance
# for the iterative algorithm (used only if `dims`=3).
# If `verbose` is true, the convergence reached at each iteration is printed.
function normalizeMean!(X::Matrix{T};
                       dims::Int=2,
                       maxiter::Int=100,
                       tol::Real=0.,
                       verbose::Bool=false) where T<:Real
   if     dims==0 return
   elseif dims∈(1, 2)
       r, c=size(X)
       @views dims==1 ? n=[1.0/sum(X[:, j]) for j=1:c] :
                        n=[1.0/sum(X[i, :]) for i=1:r]

       @inbounds dims==1 ? for j=1:c X[:, j]*=n[j] end :
                           for i=1:r X[i, :]*=n[i] end
   elseif dims==3
       r, c=size(X)
       u=min(r, c)
       tol≈0. ? tolerance = √eps(real(eltype(X))) : tolerance = tol
       iter=1
       verbose && println("running Complete Normalization iterations....")
       while true
           conv=0.
           for d=1:2
               d==1 ?   n=[1.0/mean(X[:, j]) for j=1:c] :
                        n=[1.0/mean(X[i, :]) for i=1:r]

               conv+=mean(abs.(n.-mean(n))) #xxx
               @inbounds d==1 ? for j=1:c X[:, j]*=n[j] end :
                                for i=1:r X[i, :]*=n[i] end
           end
           conv=conv/2
           verbose && println("iteration: ", iter, "; convergence: ", conv)
           (overRun = iter == maxiter) && @warn("complete normalization reached the max number of iterations before convergence:", iter)
           conv <= tolerance || overRun==true ? break : nothing
           iter+=1
       end # while
   else throw(ArgumentError, "the `dims` argument must be 0 (do nothing), 1, 2 or 3 (complete normalization)")
   end
end



end # module
