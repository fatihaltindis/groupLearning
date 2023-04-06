module ERPsML

using   LinearAlgebra,
        CovarianceEstimation,
        Diagonalizations,
        PosDefManifold,
        PosDefManifoldML,
        Random,
        JLD

push!(LOAD_PATH, homedir()*"/src/julia/Modules")
using   EEGio,
        FileSystem,
        EEGpreprocessing,
        System,
        FileSystem,
        ERPs,
        Tyler

export  trialsCovP300,
        trialsTangVec,
        getDbInfo,
        writeCovTangVec,
        alignTangVec,
        alignTangVecTest,
        trainTestSplit,
        balanceERP,
        trainTestAcc,
        bestSourceSbj,
        bestTargetSbj,
        intraSessionAcc

"""
Estimate covariance matrices of single-trials ERPs from an EEG structure `o`
(see EEGio.jl). The extended covariance matrix for ERP is estimated
(Congedo et al., 2017), which is the covariance matrix of the trial
augmented with a prototype. As prototype, the first `ncomp` principal
components of the TARGET ERP mean is used (by default, `ncomp` is equal to 4).

RETURN
======
The covariance matrix for all concerned classes (see `mark` argument)
are returned concatenated in an ℍVector. See PosDefManifold.jl for
the type ℍVector.

OPTIONAL KEYWORD ARGUMENTS
==========================
- `mark` : by default the covariance matrices are estimated for the trials
    pointed by the `o.mark` field. You can request to compute them only
    for ceratin classes passing as optional keyword argument `mark`
    a vector of integers of a vector of vectors of integers. See the function
    `trials` in unit ERPs.jl.
- `estimator`: (default:lw) Covariance matrix estimator. The options are:
      `:scm` (sample covariance matrix),
      `:lw` (Ledoit-Wolf),
      `:tme` (Tyler M-estimator),
      `:nrtme` (normalized and regularized tme).
  For :scm, the covariance matrices are regularized.
  The amount of regularization is given by argument `scmRegSNR`, see
  function `regularize!` in PosDefManifold.jl package for its meaning.
- `ncomp`: (default 4)
    number of components for PCA on the TARGET ERP average
- `verbose`: (default false)
- `normalization`: (default nothing)
    normalization to unit determinant (:det) or unit trace (:tr)

"""
function trialsCovP300(o :: EEG;
            mark            :: Union{Vector{S}, Vector{Vector{S}}, Nothing} = nothing,
            estimator       :: Symbol = :lw,
            tylerReg        :: Symbol = :rmt,
            scmRegSNR       :: Union{Int, R} = 1e5,
            verbose         :: Bool = false,
            ncomp           :: Int = 4,
            normalization   :: Union{Symbol, Nothing} = nothing) where {R<:Real, S<:Int}

    m = mark===nothing ? o.mark : mark
    if m===o.mark
        𝐗 = o.trials===nothing ? trials(o.X, m, o.wl; offset=o.offset) : o.trials
    else
        𝐗 = trials(o.X, m, o.wl; offset=o.offset)
    end

    # PCA to keep only ncomp components for the TARGET ERP mean
    TAlabel = findfirst(isequal("Target"), o.clabels)
    X̄ = mean(o; overlapping=true, weights=:a)[TAlabel]
    X̃ₘ = X̄ * eigvecs(X̄'*X̄)[:, o.ne-ncomp+1:o.ne]

    if estimator == :nrtme
        # normalized and regularized Tyler M-estimator
        𝐂 = ℍVector([ℍ(Tyler.nrtme([X X̃ₘ]'; reg=tylerReg, verbose=verbose)) for X ∈ 𝐗])
    elseif estimator == :lw
        # Ledoit-Wolf shrinkage
        𝐂 = ℍVector([ℍ(cov(LinearShrinkage(ConstantCorrelation()), [X X̃ₘ])) for X ∈ 𝐗])
    elseif estimator == :tme
        # Tyler M-Estimator
        𝐂 =ℍVector([ℍ(Tyler.tme([X X̃ₘ]')) for X ∈ 𝐗])
    elseif estimator == :scm
        # Sample Covarience Estimator
        𝐂 = ℍVector([ℍ(cov(SimpleCovariance(), [X X̃ₘ])) for X ∈ 𝐗])
        regularize!(𝐂; SNR=scmRegSNR)
    else
        throw(ArgumentError("unknown covariance estimator "*string(estimator)*" : possible choices are :lw, :tme, :nrtme, :scm"))
    end

    if normalization == :det
        # det normalization
        for i=1:length(𝐂) 𝐂[i] = det1(𝐂[i]) end
    elseif normalization == :tr
        ### trace normalization
        for i=1:length(𝐂) 𝐂[i] = tr1(𝐂[i]) end
    end
    return 𝐂
end



"""
Return vectorized tangent vectors for the trials in EEG structure
`o` (see EEGio.jl). This function calls functions `trialsCovP300`
(from this module) and `tsMap` (from package PosDefManifoldML.jl).

RETURN
======
- T: a real matrix with vectorized tangent vectors in its colums.
    The size of the vectors (rows of T) depends on argument `vecRange`
    (see function `vecP` of package PosDefManifold.jl, which is called
    by the `tsMap` function). The number of trials (columns of T)
    is the number of trials pointed to by `mark`.
- 𝐂: the output of function `trialsCovP300` (if `returnCov` is true).

OPTIONAL KEYWORD ARGUMENTS
==========================
If `returnCov` is true return the matrix of tangent vectors and the
ℍVector of covariance matrices (the output of function `trialsCovP300`),
otherwise (default) return only the matrix of tangent vectors.

Arguments `mark`, `estimator`, `tylerReg`, `scmRegSNR`, `verbose`, `ncomp`
and `normalization` are passed to function `trialsCovP300`.

Arguments `metric`, `w`, `checkw`, `threaded`, `tol`, `transpose`
and `vecRange` are passed to function `tsMap`.

Nota Bene: By default the matrix of tangent vectors holds the tangent
vectors in its columns and the weights for estimating the base point
on the manifold for passing in the tangent space are inversely proportional
to the class numerosity (using function PosDefManifoldML.tsWeights).
If you use other weights AND do not normalize them to sum up to 1,
you must use optional keyword argument `checkw=true`.
If you wish not to use weights, use `w=[]`.
"""
function trialsTangVec(o :: EEG;
            # OKA passed to trialsCovP300
            mark            :: Union{Vector{S}, Vector{Vector{S}}, Nothing} = nothing,
            estimator       :: Symbol = :lw,
            tylerReg        :: Symbol = :rmt,
            scmRegSNR       :: Union{S, R} = 1e5,
            verbose         :: Bool = false,
            ncomp           :: S = 4,
            normalization   :: Union{Symbol, Nothing} = nothing,
            # OKA passed to tsMap
            metric          :: PosDefManifold.Metric = Fisher,
            w    	        :: Vector{R} = PosDefManifoldML.tsWeights(o.y),
            checkw   	    :: Bool = false,
            threaded	    :: Bool = true,
            tol             :: Real = 0.,
            transpose       :: Bool = false,
            vecRange        :: UnitRange = 1:o.ne,

            # OKA for this function
            returnCov       :: Bool = false) where {R<:Real, S<:Int}

    𝐂 = trialsCovP300(o;
            mark            = mark,
            estimator       = estimator,
            tylerReg        = tylerReg,
            scmRegSNR       = scmRegSNR,
            verbose         = verbose,
            ncomp           = ncomp,
            normalization   = normalization)

    T, _ = tsMap(metric, 𝐂;
            w               = w,
            ✓w              = checkw,
            ⏩              = threaded,
            tol             = tol,
            transpose       = transpose,
            vecRange        = vecRange)

    return returnCov ? (T, 𝐂) : T
end

## xxx

"""
Return Array with subject, session and run information for each file of a db
ARGUMENTS
==========
- db: String
    path to database
- nfiles: Int or nothing (default)
    number of files to process, to limit memory usage
RETURN
======
- dbinfo: Array(length(db), 3) containing subj, sess & run
"""
function getDbInfo(db; nfiles=nothing)
    files = (nfiles ≠ nothing ? loadNYdb(db)[1:nfiles] : loadNYdb(db))
    dbinfo = Array{Int64}(undef, length(files), 3)
    for (i, f) ∈ enumerate(files)
        o = readNY(f)
        dbinfo[i, :] = [o.subject; o.session; o.run]
        waste(o)
    end
    return dbinfo
end


"""
Compute and store all covariance matrices and tangent vectors from database
ARGUMENTS
==========
- db: String
    path to EEG dataset
- estimator: (default:lw) Covariance matrix estimator. The options are
  :scm (sample covariance matrix), :lw (Ledoit-Wolf),
  :tme (Tyler M-estimator), :nrtme (normalized and regularized tme).
- ncomp: Int
    number of components (default: 4)
- preload: Bool
    load Dict dbdata into memory or return name of JLD/HDF5 file for lazy loading
- nfiles: Int or nothing
    compute only for the first nfiles datafiles
Return
======
- dbdata: Dict or String
    return Dict of all covariance and tangent vectors if preload
    or the name of JLD file for lazy loading
"""
function writeCovTangVec(db; estimators=[:lw, :scm], ncomp=4,
                      preload=false, nfiles=nothing)
    if isfile("./dbdata"*basename(db)*".jld")
        println("loading existing precomputed Cov and TV...")
        if preload
            f = jldopen("./dbdata"*basename(db)*".jld", "r")
            dbdata = Dict()
            for n in names(f)
                for e in estimators
                    kt, kc = n*"/TV/"*string(e), n*"/Cov/"*string(e)
                    dbdata[kt], C_ = read(f[kt]), read(f[kc])
                    dbdata[kc] = ℍVector([ℍ(C) for C in C_])
                end
                dbdata[n*"/y"] = read(f[n*"/y"])
            end
            close(f)
            return dbdata
        else
            return "./dbdata" * basename(db) * ".jld"
        end
    else
        dbdata = Dict()
        files = (nfiles ≠ nothing ? loadNYdb(db)[1:nfiles] : loadNYdb(db))
        f = jldopen("./dbdata"*basename(db)*".jld", "w")
        try
            for i in 1:length(files)
                o = readNY(files[i];  bandpass=(1, 16))
                for e in estimators
                    kt, kc = string(i)*"/TV/"*string(e), string(i)*"/Cov/"*string(e)
                    dbdata[kt], dbdata[kc] = trialsTangVec(o; estimator=e, verbose=false, ncomp=ncomp, retCov=true)
                    f[kt] = dbdata[kt]
                    f[kc] = [Matrix(C) for C in dbdata[kc]]
                end
                dbdata[string(i)*"/y"] = f[string(i)*"/y"] = o.y
                waste(o)
            end
        catch ex
            print(ex)
            close(f)
        end
        close(f)
        return preload ? dbdata : "./dbdata"*basename(db)*".jld"
    end
end


"""
Return train test split based on calibration/test EEG data
This function uses the first run as calibration set and others run as test.
ARGUMENTS
==========
- dbinfo: Array of Int
    output of getDbInfo, holds subj, sess & run
Return
======
- trainId: Dict of Int
    indices for the training set for each subject
- testId: Dict of Array of Int
    indices for the test sets for each subject
"""
function trainTestSplit(dbinfo)
    subjects = unique(dbinfo[:,1])
    trainId, testId = Dict(), Dict()
    for s in subjects
        subjId = dbinfo[findall(in(s), dbinfo[:, 1]), :]
        trainId[s] = findall(in(s), dbinfo[:, 1])[1]
        testId[s] = findall(in(s), dbinfo[:, 1])[2:end]
    end
    return trainId, testId
end


"""
return accuracy for train/test split for tangent vectors
From an array 𝐓 of tangent vectors, of size nsplit, train a model on the first
nsplit-1 data and test on the last one.
Parameters
==========
- 𝐓 :: Vector{Array{Float64,2}}
    array of Tangent vector
- 𝐲 :: Vector{IntVector} ;
    associated labels
- model :: MLmodel = ENLR(Fisher)
    Model to train on tangent vector
Return
======
- acc: Float
    Accuracy
"""
function trainTestAcc(
            𝐓 :: Vector{Matrix{R}},
            𝐲 :: Vector{Vector{S}};
            model :: MLmodel = ENLR(Fisher)) where {R<:Real, S<:Int}

    nsplit = length(𝐓)
    𝐓Tr = vcat([Matrix(𝐓[i]') for i in 1:nsplit-1]...)
    𝐓Te = Matrix(𝐓[nsplit]')
    𝐲Tr = vcat(𝐲[1:nsplit-1]...)
    𝐲Te = 𝐲[nsplit]
    model = fit(ENLR(Fisher), 𝐓Tr, 𝐲Tr)
    𝐲Pr = predict(model, 𝐓Te)
    predErr = predictErr(𝐲Te, 𝐲Pr)
    #TODO: use balanced accuracy
    acc = 100. - predErr
    return acc
end


"""
return accuracy for train/test split for covariance matrices
From an array 𝐓 of tangent vectors, of size nsplit, train a model on the first
nsplit-1 data and test on the last one.
Parameters
==========
- 𝐏 :: Vector{ℍVector}
    array of covariance matrices
- 𝐲 :: Vector{IntVector} ;
    associated labels
- model :: MLmodel = ENLR(Fisher)
    Model to train on covariance matrices
Return
======
- acc: Float
    Accuracy
"""
function trainTestAcc(
            𝐏 :: Vector{ℍVector},
            𝐲 :: Vector{Vector{S}};
            model :: MLmodel = ENLR(Fisher)) where S<:Int

    nsplit = length(𝐏)
    𝐏Tr = vcat([𝐏[i] for i in 1:nsplit-1]...)
    𝐲Tr = vcat(𝐲[1:nsplit-1]...)
    model = fit(model, 𝐏Tr, 𝐲Tr)
    𝐏Te = 𝐏[nsplit]
    𝐲Te = 𝐲[nsplit]
    𝐲Pr = predict(model, 𝐏Te)
    predErr = predictErr(𝐲Te, 𝐲Pr)
    #TODO: use balanced accuracy
    acc = 100. - predErr
    return acc
end


"""
Return aligned tangent vectors
From a target subject, whom tangent vectors are stored as the
first element of 𝐓 array, and tangent vectors of source subjects,
stored next in 𝐓, this function align subjects with Maximum Covariance Analysis if there is only one source subject or Generalized Maximum Covariance Analysis if there is more than one source subject.
Make sure that in each aggregated tangent vectors, the label are also aligned.
Parameters
==========
- 𝐓: Vector{Array{Float64,2}}
    Array of tangent vectors
- align_args: nothing (default) or Array
    arguments for MCA/GMCA
Return
======
- 𝐓aligned: Array{Float64, 2}
    Aligned tangent vectors
"""
function alignTangVec(𝐓; align_args=nothing)
    if length(𝐓) == 2
        gm = (align_args == nothing ? mca(𝐓[1], 𝐓[2]) : mca(𝐓[1], 𝐓[2], align_args...))
    else
        gm = (align_args == nothing ? gmca(𝐓; dims=2) : gmca(𝐓; dims=2, align_args...))
    end
    return [gm.F[i]'*𝐓[i] for i=1:length(𝐓)], gm
end

"""
return aligned tangent vector from a computed model
To align test data with the filter computed on calibration data between a
target subject and some source subjects. The filter for target subject should
be the last gm.F filter.
Parameters
==========
- gm: G/MCA model
    Model for tangent vector alignment
- 𝐓: Vector{Array{Float64,2}}
    Array of tangent vectors for test data
Return
======
- 𝐓aligned: Array{Float64, 2}
    Aligned tangent vectors
"""
function alignTangVecTest(gm, 𝐓)
    return gm.F[length(gm.F)]'*𝐓
end


"""
Return  list of source subjects ordered by alignment with target
The alignment is estimated with the eigenvalues of the CCA computed
on tangent vectors.
Parameters:
- db is path to EEG dataset
- target_s is int, target subject id
- ncomp is Int, number of mean component for extended signal
- estimator is String, default value :lw (Ledoit-Wolf) or :Tyler
- verbose is Bool
"""
function bestSourceSbj(db, target_s, ncomp; verbose=false, estimator=:lw)
    files = loadNYdb(db)
    ot_ = readNY(files[target_s];  bandpass=(1, 16))
    vot_ = trialsTangVec(ot_, verbose=false, ncomp=ncomp, estimator=:lw)
    d_ = size(vot_, 1)
    # evcca = Matrix{Float64}(undef, d_, length(files))
    evcca = zeros(Float64, d_, length(files))
    ⌚ = verbose && now()
    for (i, file) ∈ enumerate(files)
        os_ = readNY(files[i];  bandpass=(1, 16))
        verbose && println("file ", i, ", target ", target_s)
        if os_.subject ≠ ot_.subject
            verbose && println("subject ", os_.subject, ", target ", ot_.subject)
            𝐓, _ = getTV(db, target_s, i; verbose=verbose, ncomp=ncomp, estimator=estimator)
            model = cca(𝐓[1], 𝐓[2]; dims=2, simple=false)
            evcca[1:size(model.D)[1], i] = diag(model.D)
        end
        waste(ot_)
    end
    verbose && println("Estimating CCA done in ", now()-⌚)
    return sortperm(evcca[1, :], rev=true), evcca
end


"""
Return average accuracy for intrasession of subject
Parameters:
- o is EEG
- nFolds is Int for k-fold cross-validation
- shuffle is Bool
"""
function intraSessionAcc(o; nFolds=10, shuffle=false, ncomp=4)
    𝐂lw = trialsCovP300(o, estimator=:lw, ncomp=ncomp, normalization=:det)
    # classification
    args=(shuffle=shuffle, tol=1e-6, verbose=false, nFolds=nFolds, lambda_min_ratio=1e-4)
    cvlw = cvAcc(ENLR(Fisher), 𝐂lw, o.y; args...)

    return cvlw.avgAcc
end

"""
Return subject ordered by intrasession accuracy
Parameters:
- db is path to EEG dataset
- nFolds is Int for number of k-fold cross-validation
- verbose is Bool
"""
function bestTargetSbj(db; nFolds=10, verbose=true, ncomp=4)
    files = loadNYdb(db)
    ⌚ = verbose && now()
    subjAcc = [intraSessionAcc(readNY(files[i];  bandpass=(1, 16)), nFolds=nFolds, ncomp=ncomp) for i in eachindex(files)]
    bestTargetIdx = sortperm(subjAcc, rev=true)
    verbose && println("Estimation of intra-session accuracy done in ", now()-⌚)
    return bestTargetIdx, subjAcc
end


"""
Return a vector of HermitianVector with balanced labels
The min number of class labels 𝐲 on all ℍVector is computed and a new output
Vector 𝐏out is computed with the same number of labels for all input 𝐏.
The first elements of each labels are selected, the extra elements are dropped.
Parameters
==========
- 𝐏: Vector of ℍVector
    Input covariance matrices vector
- 𝐲: Vector of IntVector
    vector of corresponding labels vectors
Return
======
- 𝐏out: Vector of ℍVector
    Output covariance matrices vector, with balanced class
- 𝐲: Vector of IntVector
    array of corresponding labels vectors, with balanced class
"""
function balanceERP(𝐏 :: Vector{ℍVector},  𝐲 :: Vector{Vector{S}}) where S<:Int
    if size(𝐏,1) == size(𝐲,1)
        labels = unique([unique(y) for y in 𝐲])[1]
        count = [[sum(x->x==lab, y) for lab in labels] for y in 𝐲]
        minLabels = [min([count[i][lab] for i in 1:size(count,1)]...) for lab in labels]
        # sum(minLabels)
        𝐏out = Vector{Array{Hermitian,1}}(undef, size(𝐏,1))

        for k in 1:size(𝐏,1)
            labelsplit = [findall(x->x==lab,𝐲[k]) for lab in labels]
            labelresize = [labelsplit[lab][1:minLabels[lab]] for lab in 1:length(labels)]
            𝐏lab = [𝐏[k][i] for i in labelresize]
            𝐏out[k] = []
            for lab in 1:length(labels)
                append!(𝐏out[k],[f for f in 𝐏lab[lab]])
            end
        end

        𝐲 = vcat(fill.(labels, minLabels)...)

        return 𝐏out, 𝐲
    else
        e = error("𝐏 and 𝐲 must be the same size. 𝐏 the must be the vector of ℍVector of Covariances Matrices and 𝐲 the vector of labels corresponding to the ℍVectors")
        throw(e)
    end
end


"""
Return a vector of vectorized tangent vectors with balanced labels
The min number of class labels 𝐲 on all tangent vectors is computed and a new
output Vector 𝐓out is computed with the same number of labels for all input 𝐓.
The first elements of each labels are selected, the extra elements are dropped.
Parameters
==========
- 𝐓: Vector of Array
    Input vector of tangent vectors, stacked in columns
- 𝐲: Vector of IntVector
    vector of corresponding labels vectors
Return
======
- 𝐓out: Vector of Array
    Output vector of tangent vectors, with balanced class
- 𝐲: Vector of IntVector
    array of corresponding labels vectors, with balanced class
"""
function balanceERP(𝐓 :: Vector{Matrix{R}}, 𝐲 :: Vector{Vector{S}}) where {R<:Real, S<:Int}

    if size(𝐓, 1) == size(𝐲, 1)
        labels = unique([unique(y) for y in 𝐲])[1]
        count = [[sum(x->x==lab, y) for lab in labels] for y in 𝐲]
        minLabels = [min([count[i][lab] for i in 1:size(count,1)]...) for lab in labels]
        𝐓out = [Array{Float64,2}(undef,size(𝐓[1], 1),sum(minLabels)) for idx in 𝐓]

        for k in 1:size(𝐓,1)
            labelsplit = [findall(x->x==lab,𝐲[k]) for lab in labels]
            labelresize = [labelsplit[lab][1:minLabels[lab]] for lab in 1:length(labels)]
            𝐓lab = [𝐓[k][:,i] for i in labelresize]
            𝐓out[k] = hcat(𝐓lab...)
        end
        𝐲 = vcat(fill.(labels,minLabels)...)
        return 𝐓out, 𝐲
    else
        e = error("𝐓 and 𝐲 must be the same size. 𝐓 the must be the vector of Tangent Vectors and 𝐲 the vector of corresponding labels ")
        throw(e)
    end
end #balanceERP

end # Module
