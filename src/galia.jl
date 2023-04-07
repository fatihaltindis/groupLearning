# Unit galia.jl, part of groupLearning Package for julia
#
# MIT License 
# version: 10 Sept 2022
# Copyright (c) - 2023
# Fatih Altindis and Marco Congedo
# Abdullah Gul University, Kayseri
# GIPSA-lab, CNRS, University Grenoble Alpes

# This function implements a special joint blind source separation algorithm
# particularly suiting joint alignmment of multiple-dataset feature vectors.

# It takes as input a vector of M matrices, each one holding in columns the 
# feature vectors for each of M datasets (e.g., subjects).
# For the original data, the length of these vectors may be different for each dataset,
# however the number of features vectors must be the same for all datasets
# and should be at least equal to the highest feature vector length.
# If the length of these vectors is different, first one must whiten the M input matrices
# using M (whitening) matrices 𝐖=[W[1], ..., W[m]] as it follows:
# T[m] = W[m]' * X[m], where W[m]' = w^-1 V', X[m] is the original data and X[m] = VwZ its SVD.
# Notice that the number of rows of matrices W[m]' is typically chosen so as to operate
# a dimensionality reduction (efficient Procrustes).  
# Finally, matrices 𝐓=[T[1], ..., T[M]] are given as input of the function `joal`.
# If the length of these vectors is NOT different and no dimensionality reduction is sought,
# then the W[m] matrices are just the identity.

# `joal` outputs M matrices 𝐔=[U[1],..., U[m]]. After running the algorithm all columns of all 
# matrices 𝐔 must be normalized to unit norm. 
# Then, the aligning matrices are obtained as B[m] = 𝐔[m]' * 𝐖[m]' for each m=1:M.

# These matrices are such that B[i] (T[i] T[j]') B[j]' are as diagonal
# as possible for all i ≠ j = 1:M 

# Given a full group learning model obtained on M subjects using the `joal` function,
# function `joal_newss` align a new subject to the M subjects of the full model.
# The function takes as input arguments (𝐓, 𝐔, T), where
# 𝐓 is the input of the `joal` function,
# 𝐔 is the output of the `joal` function and
# T is the whitened feature vectors matrix for the new subject.
# Let us denote the whitening matrix W.
# Note that the whitened T must have the same dimension as all matrices in 𝐓.
# Function `joal_newss` outputs matrix U, which columns must be normalized to unit norm
# and from which then the alignment matrix for the new subject is computed as B = U' * W' 

# See : 
# Congedo M., Bleuzé A., Mattout J. (2022)
# Group Learning by Joint Alignment in the Riemannian Tangent Space
# GRETSI conference, 6-9 September 2022, Nancy, France.

# uses trifact! instead of julia cholesky and use BLAS functions
function jointAlignment(𝐓         ::AbstractArray;
                        init      :: Union{AbstractArray, Nothing} = nothing,
                        tol       :: Real = 0.,
                        maxiter   :: Int  = 1000,
                        verbose   :: Bool = false,
                        threaded  :: Bool = false)

    M, type, D = length(𝐓), eltype(𝐓[1]), size(𝐓[1], 1)
    iter, conv, oldconv, 😋, conv_ = 1, 0., 1.01, false, 0.
    diverging, previous_conv = false, 1.0
    tol==0. ? tolerance = √eps(real(type)) : tolerance = tol            

    # thread operations only if n is at least Threads.nthreads() (no worth otherwise)
    threadedx = threaded && D ≥ Threads.nthreads() # !!!

    # Get input data C_ij,k (compute only lower triangular part of the matrix of matrices)
    𝐂=[[Matrix{type}(undef, D, D) for i=1:j] for j=1:M]
    for j=1:M, i=j+1:M  
        𝐂[i][j] = 𝐓[i]* 𝐓[j]' 
    end

    # use the initialization for matrices U if provided, otherwise initialize them to the identity 
    𝐔 = init===nothing ? [Matrix{type}(I, D, D) for m=1:M] : init

    # pre-allocate memory
    V = Matrix{type}(undef, D, M - 1)
    𝐑 = MatrixVector([Matrix{type}(undef, D, D) for d=1:D])

    verbose && @info("Iterating joal algorithm...")

    # This function is used only if threadedx is true
    function triS!(R, U, L, d)
        # y = L'\(L\BLAS.symv('L', R, U[:, d])) # BLAS call computes 𝐑[η]*𝐔[i][:, η]
        y = L'\(L\R*U[:, d])
        U[:, d] = y/√(PosDefManifold.quadraticForm(y, R))
    end

    # Iterations
    while true
        conv_ =0.
        @inbounds for e2=1:2, i=1:M # m optimizations for updating 𝐔[1]...𝐔[m]
                                    # double loop to avoid oscillating convergence
            for d=1:D
                x = 1   
                for j=1:i-1
                    V[:, x] = 𝐂[i][j] * 𝐔[j][:, d] 
                    x += 1   
                end 
                for j=i+1:M  
                    V[:, x] = 𝐔[j][:, d]' * 𝐂[j][i]
                    x += 1 
                end

                # write the lower triangle of V * V'
                𝐑[d] = threadedx ?  LowerTriangular(BLAS.syrk('L', 'N', 1.0, V)) : (V * V')
            end

            # do 1 power iteration
            L=trifact!(threadedx ? PosDefManifold.fVec(sum, 𝐑) : sum(𝐑)) # Cholesky LL'of 𝐑[1]+...+𝐑[n]

            # Solve Lx=𝐑[η]*𝐔[i][:, η] for x and L'*y=x for y,
            # then updates the ηth column of 𝐔[i] as 𝐔[i][:, η] <- y/sqrt(y'*𝐑[η]*y)
            if threadedx 
                @threads for d=1:D triS!(𝐑[d], 𝐔[i], L, d) end 
            else
                for d=1:D
                    # computee 𝐑[d]*𝐔[i][:, d]
                    y =  threadedx ?    L'\(L\BLAS.symv('L', 𝐑[d], 𝐔[i][:, d])) :
                                        L'\(L\(𝐑[d]*𝐔[i][:, d]))
                    𝐔[i][:, d] = y/√(PosDefManifold.quadraticForm(y, 𝐑[d])) # qf accepts low. tri. matrices
                end    
            end       

            # update convergence in a quick way # improve this !!
            conv_+=PosDefManifold.ss(𝐔[i])   
        end

        conv_/=(2*D^2*M) # 2 because of the double loop
        iter==1 ? conv=1. : conv = abs((conv_-oldconv)/oldconv)  # relative change
        
        verbose && println("iteration: ", iter, "; convergence: ", conv)
        previous_conv < 1e-5 && !diverging ? (diverging = previous_conv < conv) && verbose && @warn("NoJoB diverged at:", iter) : nothing
        (overRun = iter == maxiter) && @warn("NoJoB: reached the max number of iterations before convergence:", iter)
        (😋 = 0. <= conv <= tolerance) || overRun == true ? break : nothing
        previous_conv = deepcopy(conv)
        oldconv=conv_
        iter += 1
    end # while

    return 𝐔, iter, conv, diverging # NB: columns have not unit norm
end
joal = jointAlignment; # alias

function fastAlignment(  𝐓::AbstractArray, 𝐔::AbstractArray, T::Matrix;
                                threaded  :: Bool = false)

    M, type, D = length(𝐓), eltype(𝐓[1]), size(𝐓[1], 1)

    # thread operations only if n is at least Threads.nthreads() (no worth otherwise)
    threadedx = threaded && D ≥ Threads.nthreads() # !!!

    # Get input data C_ij
    𝐂 = [T * 𝐓[j]' for j=1:M]

     # pre-allocate memory
    V = Matrix{type}(undef, D, M)
    𝐑 = MatrixVector([Matrix{type}(undef, D, D) for d=1:D])
    U = Matrix{type}(undef, D, D) 

    for d=1:D
        for j=1:M
            V[:, j] = 𝐂[j]*𝐔[j][:, d]
        end  
        𝐑[d] = Hermitian(V * V')
    end

    H = invsqrt(Hermitian(sum(𝐑)))

    for d=1:D    
        ev=eigvecs(Hermitian(H * 𝐑[d] * H'))
        U[:, d] = H * ev[:, end]
    end  # NB: the columns of U are not in unit norm   

    return U
end
faal = fastAlignment; # alias

function trifact!(	P::Matrix{T};
                    check::Bool = true,
                    tol::Real = √eps(real(T))) where T<:Union{Real}

    LinearAlgebra.require_one_based_indexing(P)
    n = LinearAlgebra.checksquare(P)

    @inbounds for j=1:n-1
        check && abs2(P[j, j])<tol && throw(LinearAlgebra.PosDefException(1))
        f = P[j, j]
        P[j, j] = √(f)
        g = P[j, j]
        for i=j+1:n
            θ = P[i, j] / f
            cθ = conj(θ)
            for k=i:n P[k, i] -= cθ * P[k, j] end # update P and write D
            P[i, j] = θ * g # write L factor
        end
    end
    P[n, n] = √(P[n, n]) # write last diagonal element of L factor

    return LowerTriangular(P)
end

function joalInitializer(𝐓     :: Vector{Matrix{Float64}};
                         type  :: Symbol = :smart)

    M = length(𝐓)
    U_init = [Matrix{Float64}(I, size(𝐓[1], 1), size(𝐓[1], 1)) for m=1:M]
    if type == :smart
        𝐂 = [[Matrix{}(undef, size(𝐓[1], 1), size(𝐓[1], 1)) for i=1:j] for j=1:M]
        for j=1:M, i=j+1:M  
            𝐂[i][j] = 𝐓[i] * 𝐓[j]' 
        end

        H = Matrix{}(undef, size(𝐓[1], 1), size(𝐓[1], 1))
        for m=1:M
            fill!(H,0)
            for j=1:M
                if m>j 
                    H += 𝐂[m][j]
                end 
                if m<j 
                    H += 𝐂[j][m]'
                end 
            end 
            F = svd(H, alg=LinearAlgebra.QRIteration())
            U_init[m] = F.U
        end
        return U_init
    else
        return nothing
    end
end
