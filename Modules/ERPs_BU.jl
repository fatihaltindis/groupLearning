module ERPs_BU

using   LinearAlgebra,
        Statistics,
        Plots,
        StatsPlots,
        StatsBase,
        FourierAnalysis,
        PosDefManifold,
        Diagonalizations,
        Tyler,

        EEGpreprocessing

import Statistics: mean

export
   mymean,
   stim2mark,
   mark2stim,
   groupMark,
   trialsWeights,
   trials,
   reject,
   tfas,
   trialsCospectra,
   bss,
   extractTrials


## Create the Toeplitz matrix for estimating ERP means via
# multivariate regression. See congedo et al.(2016)
# `ns` is the number of samples
# `wl` is the window length, i.e., the ERPs duration, in samples
# `mark` is a vector of nc integer vectors, where nc is the number of classes,
#         holding all samples where there is a stimulation.
#         Typically, it is created from a stimulation vector using the
#         'stim2mark' function of this module.
# Optional keyword arguments:
# `offset` is an offset for determining the trial starting sample
#         with respect to the samples in `mark`. Can be zero, positive or negative.
#         BEWARE! If mark has been created using an offset, you won't need it here!
# `weights` is a vector of non-negative real weights for the trials
#         corresponding to the stimulations in `mark`.
#         The weights must be appropriately normalized (see 'mean' in this module).
#         By default, no weights are used.
function toep(ns::S, wl::S, mark::Vector{Vector{S}};
              offset::S = 0,
              weights::Union{Vector{Vector{R}}, Symbol} = :none
              ) where {R<:Real, S<:Int}
  T=zeros(wl*length(mark), ns)
  nc=length(mark)
  if     weights==:none
           @inbounds for c=1:nc, j=1:wl, i=1:length(mark[c])
                     T[(c-1)*wl+j, mark[c][i]+j-1+offset]=1 end
  else
           @inbounds for c=1:nc, j=1:wl, i=1:length(mark[c])
                    T[(c-1)*wl+j, mark[c][i]+j-1+offset]= weights[c][i] end
  end
  return T
end


## Estimate the mean ERPs using the arithmetic mean if `overlap` is false
# or the multivariate regression method explained # in Congedo et al.(2016)
# if `overlap` is true.
# `X` is the whole EEG recording of size # of samples x # of electrodes
# `wl` is the window length, i.e., the ERPs duration, in samples
# `mark` is a vector of nc integer vectors, where nc is the number of classes,
#         holding all samples where there is a stimulation.
#         Typically, it is created from a stimulation vector using the
#         stim2mark function in EEGPreprocessing.jl.
# Optional keyword arguments:
# `offset` is an offset for determining tha trial starting sample
#         with respect to the samples in `mark`. Can be zero, positive or negative.
#         BEWARE! If mark has been created using an offset, you won't need it here!
# `weights` is a vector of non-negative real weights for the trials
#         corresponding to the stimulations in `mark`. They don't need to be
#         normalized. If `weights=:a` is passed, adaptive weights are
#         computed as the inverse of the squared Frobenius norm of the
#         trials data, along the lines of Congedo et al. (2016).
#         By default, no weights are used.
# REFERENCE:
# Congedo M, Korczowski L, Delorme A, Lopes da Silva F. (2016)
# Spatio-temporal common pattern: A companion method for ERP analysis in
# the time domain. Journal of Neuroscience Methods, 267, 74-88.
# https://hal.archives-ouvertes.fr/hal-01343026/document
function mymean(X::Matrix{R}, wl::S, mark::Vector{Vector{S}}, overlap::Bool=false;
         offset :: S = 0,
         weights:: Union{Vector{Vector{R}}, Symbol}=:none) where {R<:Real, S<:Int}

  if overlap
    if      weights==:none
            T=toep(size(X, 1), wl, mark; offset=offset)
            Xbar=inv(T*T') * (T*X)
    else
            weights==:a ? weights=[[1/(norm(X[mark[i][j]+offset:mark[i][j]+offset+wl-1,:])^2) for j=1:length(mark[i])] for i=1:length(mark)] : nothing
            T=toep(size(X, 1), wl, mark; offset=offset, weights=weights)
            w=[sum(weight.^2)/sum(weight) for weight in weights]
            Xbar=Diagonal(vcat(fill.(w, wl)...)) * (inv(T*T') * (T*X))
    end
    return [Xbar[wl*(c-1)+1:wl*c, :] for c=1:length(mark)]
  else
    if      weights==:none
            return [mean(X[mark[c][j]+offset:mark[c][j]+offset+wl-1, :] for j=1:length(mark[c])) for c=1:length(mark)]
    else
            weights==:a ? weights=[[1/(norm(X[mark[i][j]:mark[i][j]+wl-1,:])^2) for j=1:length(mark[i])] for i=1:length(mark)] : nothing
            w=[weight./sum(weight) for weight in weights]
            return [sum(X[mark[c][j]+offset:mark[c][j]+offset+wl-1, :]*w[c][j] for j=1:length(mark[c])) for c=1:length(mark)]
    end
  end
end

## Convert a vector of stimulations into a vector of vectors of markers;
# `stim` is a ns-vector of stimulations, where ns is the number of
# samples in the data `stim` is coupled with.
# `stim` holds 0 everywhere, but at the samples where a stimulation occurs,
# where it holds an integer coding the stimulation class.
# By default the code for stimulations MUST be the natural numbers 1, 2, 3, ...
# If not, pass a vector of integers with the code for each class as `code`.
# In this case the first element of `code` MUST be 0 (no stimulation),
# then, arbitary integers can be used (even negative).
# The integers provided in `code` must be unique.
# If `code` is provided, the nc-vectors are arranged in the order given there,
# otherwise the first vector correspond to stimulation=1, the second to
# stimulation=2, etc.
# RETURN an nc-vector of vectors, where `nc` is the number of classes, i.e.,
# the non-zero elements in `code` or non-sero stimulations in `stim`.
# In the vector for each class there are as many elements as stimulations for
# that class and the elements are the samples when the stimulations occurred.
# In each vector the samples are sorted in ascending order.
# The window-length `wl` must be provided in order to make sure not to include
# markers that allow the definition of trials going beyond `ns`, where `ns`
# is the number of samples in `stim`, that is,
# a marker can only point to a sample whose index is < ns-wl+offset.
# `offset` can be applied to shift the markers with respect to the stim.
# The `offset` must be provided in samples. It can be positive, to shift the
# onset of the trial forward, or negative, to shift it backward (0 by default).
function stim2mark(stim::Vector{S}, wl::S;
                      offset::S=0, code=nothing) where S <: Int
    unic = code===nothing ? sort(unique(stim)) : code
    numc = length(unic)
    code===nothing && unic ≠ collect(0:numc-1) && throw(ArgumentError("ERPs.jl, `stim2mark` function: the stimulation code is not the sequence 0...i, where i is the number of classes."))
    code≠nothing && length(unique(code))≠length(code) && throw(ArgumentError("ERPs.jl, `stim2mark` function: the integers in the provided `code` vector are not unique."))
    return[[i+offset for i in eachindex(stim) if stim[i]==unic[j+1] && i+offset+wl-1<=length(stim)] for j=1:numc-1]
end


# The reverse transformation of `stim2mark`
# if an offset has been used in `stim2mark`, -offset must be used here
# to get back the original stim vector.
function mark2stim(mark::Vector{Vector{S}}, ns::S;
                      offset::S=0, code=nothing) where S <: Int
    stim=zeros(S, ns)
    unic = code===nothing ? collect(0:length(mark)) : code
    for z=1:length(mark), j ∈ mark[z] stim[j] = unic[z+1]+offset end
    return stim
end


## Group vector of markers and sort them within each class.
# Example: suppose mark holds 4 vectors of markes and mergeInd=[[1, 2], [3, 4]],
# then the result will hold 2 markers vector, the mark vectors 1 and 2
# concatenated and sorted and mark vectors 3 and 4 concatenated and sorted.
groupMark(mark::Vector{Vector{Int}}, mergeInd::Vector{Vector{Int}}) =
    [sort(vcat(mark[m]...)) for m ∈ mergeInd]
    #    𝐑_=trials(X, mark, wl; offset=offset, shape=:byclass)
    #    𝐑=[vcat(𝐑_[c]...) for c ∈ conditionInd]


## Compute adaptive weights for trials as the inverse of their squared
# Frobenius norm, along the lines of Congedo et al. (2016).
# `X` is the whole EEG recording of size # of samples x # of electrodes
# `wl` is the window length, i.e., the ERPs duration, in samples
# `mark` is either:
#       1) a vector of all samples serial number where there is a stimulation.
#       2) a vector of nc vectors as per 1), where nc is the number of classes.
#       Typically `mark` is created from a stimulation vector using the
#       'stim2mark' function of this module.
# if `mark` is a vector and a matrix is passed as the `M` argument, then the
#       weights are computed as the squared norm of (trial-M) for each trial.
#       `M` in general is the mean ERP for the class and can be used
#       to obtain a better weight estimate.
#       if `mark` is a vector of vector and a vector of matrices is passed as
#       the `M` argument, then the estimator here above is used for all classes.
#       By default `M` is equal to nothing.
# Optional keyword arguments:
# `offset` is an offset for determining the trial starting sample
#       with respect to the samples in `mark`. Can be zero, positive or negative.
#       BEWARE! If mark has been created using an offset, you won't need it here!
function trialsWeights(X::Matrix{R}, wl::S, mark::Vector{S};
                        M::Union{Matrix{R}, Nothing} = nothing,
                        offset::S = 0,
                        ) where {R<:Real, S<:Int}
    if M===nothing
        w = [1/(norm(X[m+offset:m+offset+wl-1, :])^2) for m ∈ mark]
    else
        w = [1/(norm(X[m+offset:m+offset+wl-1, :]-M)^2) for m ∈ mark]
    end
    return w./mean(w)
end


function trialsWeights(X::Matrix{R}, wl::S, mark::Vector{Vector{S}};
                        M::Union{Vector{Matrix{R}}, Nothing} = nothing,
                        offset::S = 0) where {R<:Real, S<:Int}
    M===nothing || length(M)==length(mark) || throw(ArgumentError("The length of arguments `mark` and `𝐌` must be the same."))
    if M===nothing
        return [trialsWeights(X, wl, m; offset=offset) for m ∈ mark]
    else
        return [trialsWeights(X, wl, m; M=g, offset=offset) for (m, g) ∈ zip(mark, M)]
    end
end



## Extract trials from a tall data matrix `X` in several ways:
# - if `mark` is a markers' vector, RETURN a vector of trials
# - if `mark` is a vector of markers' vectors, RETURN:
#   i. a vector of vectors of trials if `shape` ≠ `:cat`,
#   ii. all trials concatenated in a single vector if `shape` == `:cat`.
#   By default `shape` is equal to `:cat`.
# Trials are defined starting at samples listed in `mark` plus the `offset`
# (0 by default) and are `wl` samples long. `wl` must be provided.
# Each trial is a `wl` x ne matrix, where ne is the number of electrodes,
# i.e., the number of columns of `X`.
# Optional keyword arguments:
# `weights` can be given to the trials. N.B: No normalization is performed.
#   Normalized weights can be obtained via function 'trialsWeights'
#   in this module.
#   `weights` must be a vector of the same shape as `mark`, thus,
#   it must be a vector if `mark` is a vector, a vector of vectors
#   if `mark` is a vector of vectors. There is no check that the corresponding
#   vectors in `mark` and `weights` have exactly the same size.
# `offset` is the offset for determining the trial starting sample
#    with respect to the samples in `mark`. Can be zero, positive or negative.
#    BEWARE! If mark has been created using an offset, you won't need it here!
trials( X::Matrix{R}, mark::Vector{S}, wl::S;
        weights::Union{Vector{R}, Nothing}=nothing,
        offset::S=0
        ) where {R<:Real, S<:Int} =
    if weights===nothing
        return [X[mark[j]+offset:mark[j]+offset+wl-1, :] for j=1:length(mark)]
    else
        return [X[mark[j]+offset:mark[j]+offset+wl-1, :]*weights[j] for j=1:length(mark)]
    end

trials( X::Matrix{R}, mark::Vector{Vector{S}}, wl::S;
        weights::Union{Vector{Vector{R}}, Nothing}=nothing,
        offset::S=0,
        shape::Symbol=:cat
        ) where {R<:Real, S<:Int} =
    if shape==:cat
        if weights===nothing
            return [X[mark[i][j]+offset:mark[i][j]+offset+wl-1, :] for i=1:length(mark) for j=1:length(mark[i])]
        else
            return [X[mark[i][j]+offset:mark[i][j]+offset+wl-1, :]*weights[i][j] for i=1:length(mark) for j=1:length(mark[i])]
        end
    else
        if weights===nothing
            return [trials(X, m, wl; offset=offset) for m ∈ mark]
        else
            return [trials(X, m, wl; weights=w, offset=offset) for (m, w) ∈ zip(mark, weights)]
        end
    end

##
function extractTrials( X::Matrix{R},
                   mark::Vector{Vector{S}},
                   #conditionInd,
                   wl::S;
                   offset::S = 0,
                   what::Union{Vector{R}, S, Nothing} = nothing,
                   weights::Symbol=:none
                   ) where {R<:Real, S<:Int}
    #𝐑_=trials(X, mark, wl; offset=offset, shape=:byclass)
    #𝐑=[vcat(𝐑_[c]...) for c ∈ conditionInd]
    𝐑=trials(X, mark, wl; offset=offset, shape=:byclass)

    # weights
    if weights≠:none
        w=[[norm(t) for t ∈ r] for r ∈ 𝐑]
        for v∈w v[:]=v/mean(v) end
    else
        w=nothing
    end

    if      what === nothing
            return weights==:none ? 𝐑 : (𝐑, w)
    elseif  what isa Int
            𝐒 = [[t[:, what] for t ∈ r] for r ∈ 𝐑]
            return weights==:none ? 𝐒 : (𝐒, w)
    else
            𝐒 = [[t*what for t ∈ r] for r ∈ 𝐑]
            return weights==:none ? 𝐒 : (𝐒, w)
    end
end


## Automatic rejection of artefacted trials in ERP data
# given a tall data matrix `X` and a vector of stimulations `stim`
# with length equal to size(X, 1), the trial duration in samples `wl`
# and the number of classes `nc`.
# `stim` is filled with 0 where there is no stimulation and natural
# numbers 1, 2,...,nc where there is a stimulation for the corresponding class.
# METHOD: a threshold is determined as m+((m-a)*`limit`),
# where `limit` can be provided as an Int or Real number (4 by default).
# m and a are the trimmed mean and minimum of the Field Root Mean Square (FRMS).
# The FRMS is the square root of the Global Field Power (GFP)
# divided by the number of electrodes.
# The GFP is the sum of squares of the potentials across electrodes
# at all samples.
# The trimmed mean is computed as the mean of the 2*wl (sorted) FRMS values
# around the median. Therefore, `X` should hold several trials.
# RETURN the 5-tuple holding the following elements:
# 1) cleanstim: the stim vector with the accepted trials,
# 2) rejecstim: the stim vector with the rejected trial,
# 3) cleanmark: cleanstim in mark format (see below)
# 4) rejecmark: rejecstim in mark format (see below)
# 5) a tuple with three plots (p1, p2, p3):
#   p1 is a line plot of the FRMS sorted in ascending order
#   p2 is a line plot of the maximum FRMS in each trial sorted in
#     ascending order and plotted separatedly for accepted and rejected trials.
#   p3 is a bar plot with the number of accepted and rejected trials per class.
# EXAMPLE (reject trials and show all plots):
# a, b, a_, b_, p = reject(X, stim, wl, nc; limit=4)
# 📈 = plot(p...; layout=(3, 1), size=(500, 750))
# savefig(📈, homedir()*"\\_ArtefactRejection.png")
# NB: stim = cleanstim+rejecstim
# The mark format: a vector of nc Int vectors, each holding the samples
# where there is a stimulation for the corresponding class, that is,
# a non-zero element of stim.
function reject(X::Matrix{R}, stim::Vector{Int}, wl::S, nc::S;
                offset::S = 0,
                limit::Union{R, S} = 4) where {R<:Real, S<:Int}

    ns, ne = size(X)
    gfp = [x⋅x for x ∈ eachrow(X)] # global field power
    frms = @.sqrt(gfp/ne) # field root mean square

    p = sortperm(frms)
    m = mean(frms[p][ns÷2-wl:ns÷2+wl])
    threshold = m+(m-frms[p][1])*limit

    cleanstim = copy(stim)
    rejected = zeros(Int, nc)
    for s=1:ns
          if cleanstim[s]>0 && maximum(frms[s:s+wl-1])>threshold
                rejected[cleanstim[s]] += 1
                cleanstim[s] = 0
          end
    end

    rejecstim = stim-cleanstim

    cleanmark = stim2mark(cleanstim, wl; offset=offset)
    rejecmark = stim2mark(rejecstim, wl; offset=offset)

    # plots
    p1 = plot(frms[p], ylims=(0, threshold*3), legend=false,
        title="sorted Field Root Mean Square (FRMS)",
        xtickfontsize=13, ytickfontsize=13, xguidefontsize=16,
        yguidefontsize=16, titlefontsize=16);

    Xclean = trials(X, cleanmark, wl)
    maxfrmsclean = [ maximum(sqrt.([x⋅x for x ∈ eachrow(X)]/ne)) for X ∈ Xclean  ]
    Xrejec=trials(X, rejecmark, wl)
    maxfrmsrejec = [ maximum(sqrt.([x⋅x for x ∈ eachrow(X)]/ne)) for X ∈ Xrejec  ]
    p2 = plot([sort(maxfrmsrejec), sort(maxfrmsclean)], labels=["rejected" "accepted"],
        title="sorted FRMS per Trial",
        xtickfontsize=13, ytickfontsize=13, xguidefontsize=16,
        yguidefontsize=16, legendfontsize=12, titlefontsize=16)

    original = zeros(Int, nc)
    for s=1:ns if stim[s]>0 original[stim[s]] +=1 end end
    p3 = groupedbar(hcat(rejected, original.-rejected), bar_position = :stack,
        labels=["rejected" "accepted"], title="Status by Class",
        xtickfontsize=13, ytickfontsize=13, xguidefontsize=16,
        yguidefontsize=16, legendfontsize=12, titlefontsize=16);

    return cleanstim, rejecstim, cleanmark, rejecmark, (p1, p2, p3)
end


# Analytic Signal of a vector of vectors of data vectors `𝐓`.
# RETURN the corresponding a vector of vectors of FourierAnalysis.jl T
# Fanalyticsignal object.
# Each data vector in `𝐓` is tall with time along the first dimension.
# All data vectors in `𝐓` must have the same length.
# `sr` is the sampling rate of the data
# `wl` is the trial duration. The number of samples in the vector matrices
# may be equal of bigger then `wl`. In the latter case the trial must be
# centered in the data matrix.
# `bandwidth` and `fmax` are parameters passed to the TFanalyticsignal
# constructor of FourierAnalysis.jl.
# if `smooth` is true, the analytic signal estimations are smoothed both
# along time and along frequency, in this order.
function tfas(  𝐓::Vector{Vector{Vector{R}}},
                    sr::S, wl::S, bandwidth::Union{S, R};
                fmax::Union{S, R, Nothing} = nothing,
                smoothing::Bool = true) where {R<:Real, S<:Int}
      h = smoothing ? FourierAnalysis.hannSmoother : FourierAnalysis.noSmoother
      f, l = FourierAnalysis.TFanalyticsignal, length(𝐓[1][1])
      𝐘=[smooth(h, noSmoother, f(T, sr, l, bandwidth;
                fmax=fmax===nothing ? sr÷2 : fmax, tsmoothing=h)) for T ∈ 𝐓]
      flabels=𝐘[1][1].flabels
      tlabels=[(i-(l-wl)÷2-1)/sr*1000 for i ∈ 1:l]
      return flabels, tlabels, 𝐘
end


function tfas(  𝑻::Vector{Vector{Vector{Vector{R}}}},
                    sr::S, wl::S, bandwidth::Union{S, R};
                fmax::Union{S, R, Nothing} = nothing,
                smoothing::Bool = true) where {R<:Real, S<:Int}
      h = smoothing ? FourierAnalysis.hannSmoother : FourierAnalysis.noSmoother
      f, l = FourierAnalysis.TFanalyticsignal, length(𝑻[1][1][1])
      𝒀=[[smooth(h, noSmoother, f(T, sr, l, bandwidth;
        fmax=fmax===nothing ? sr÷2 : fmax, tsmoothing=h)) for T ∈ 𝐓] for 𝐓 ∈ 𝑻]
      flabels=𝒀[1][1][1].flabels
      tlabels=[(i-(l-wl)÷2-1)/sr*1000 for i ∈ 1:l]
      return flabels, tlabels, 𝒀
end



## Compute all cospectra averaged across the trials in data matrix `X`
# marked in vector of vectors `marks`.
# `sr` is the sampling rate
# `wl` is the window length(trial duration in samples)
# Cospectra are estimated in band-pass [`fmin`, `fmax`]
# `tapering` is the tapering window for FFT computations.
# if `non-linear`, non-linear cospectra are computed (false by default).
# RETURN a vector of Heritian matrices holding the cospectra
function trialsCospectra(X, marks, sr, wl, fmin, fmax;
            tapering  = harris4,
            nonlinear = false)
      #plan=Planner(plan_exhaustive, 8.0, o.wl, eltype(o.X)) # pre-compute a planner
      𝐑=trials(X, marks, wl; shape=:cat)
      𝙎=[crossSpectra(R, sr, wl; tapering=tapering) for R ∈ 𝐑]

      # average cospectra across trials in band-pass region (fmin, fmax)
      f=f2b(fmin, sr, wl):f2b(fmax, sr, wl)
      return ℍVector([ℍ(mean(real(𝙎[i].y[j]) for i=1:length(𝙎))) for j=f])
end

# Blind Source Separation for ERPs.
# The method here implemented is a refinement of the method presented in
# https://hal.archives-ouvertes.fr/hal-01078589
# BSS is solved by approximate joint diagonalization (AJD) of a set of
# (1) Fourier cospectra (induced and background activity) and
# (2) covariance matrices of ERP means.
# `Xerp` is a tall data matrix used to extract ERP means for (2)
# `sensor` is a vector of electrode labels (string)
# `Xcospectra` is a tall data matrix used to extract cospectra for (1)
#       This may be the same as `Xerp`, however in general `Xerp` is the
#       data passed through a narrower band-pass filter (e.g., 1-16 Hz)
# `markERP` is a Vector of vectors of Integers with the markers
#       for extracting the trials used for (2). There are exactly as many
#       Vectors as ERP classes to be included in the AJD set.
# `markCospectra` is a vector of vectors of Integers with the markers
#       for extracting epochs used for (1). This may be equal t `markERP`,
#       but it may hold a larger set of markers. Also note that `markCospectra`
#       does not have to hold the same number of vectors as ``markERP` since
#       here all trials are used to estimate the cospectra.
# `sr` is the sampling rate of the data in `Xerp` and `Xcospectra`
# `wl` is the window length of the data in `Xerp` and `Xcospectra`
#       This is the length of the trials used both as window length for
#       Fourier co-spectra computations and for computing ERPs.
# KEYWORD ARGUMENTS :
# if `erpOverlap` is true the multivariate regression method is used to compute
#       ERP means. If false (default) the arithmetic mean is used
# if `erpWeights` = `:a`, the adaptive weighting is used for computing
#       the ERP means. It defaults to `:none` (equal weighting).
#       For this and the previous argument see `Mean` in this module.
# `erpCovEst` is the covariance matrix estimation method used for the ERP means.
#       Possible choices are (defaulting to :nrtme) :
#       `:scm` = sample covariance matrix
#       `lse` = Ledoit and Wolf linear shrinkage
#       `:tme` = Tyler's M-estimator
#       `:nrtme` normalized regularized M-estimator of Zhang (see Tyler module)
# `cospectraBandPass` is the band-pass region in which cospectra are estimated.
#       The actual number of cospectra estimated depends on the Fourier
#       frequency resolution sr/wl. The default is `(1, 32)` (Hz).
#       For noisy data starting at 2Hz and/or stopping at 28 to 28 Hz
#       can give better results.
# `cospectraTapering` is the tapering window used for the FFT.
#       See the `cospectra` function in FourierAnalysis.jl
# `whiteningeVar` is the explained variance retained in the pre-whitening step.
#       Pre-whitening determines the number of sources to be estimated and
#       is a crucial hyper-parameter for this and similar BSS procedures.
#       If you pass a real number ∈(0, 1] the dimension will be adjusted to the
#       minimum integer guranteeing to explain at least the requested variance.
#       You can enforce e specific dimension passing it as an integer.
#       The default (0.999) is an appropriate choice in general
# `AJDalgorithm` is the AJD algorithm to be employed.
#       See Diagonalizations.jl for the options. The suggested choices are
#       :QNLogLike (default) and :LogLike.
# `AJDmaxIter` is the maximum number of iterations allowed for the AJD algorithm.
# If `verbose` is true (default), information on the convergence reached at
#       each iteration by all iterative algorithms employed is shown in the REPL.
function bss(Xerp::Matrix{R}, sensors::Vector{String}, Xcospectra::Matrix{R},
             markErp::Vector{Vector{Int}}, markCospectra::Vector{Vector{Int}},
             sr::Int, wl::Int;
                erpOverlap :: Bool = false,
                erpWeights :: Symbol = :none,
                erpCovEst :: Symbol = :nrtme,
                cospectraBandPass :: Tuple = (1, 32),
                cospectraTapering = slepians(sr, wl, golden),
                whiteningeVar :: Union{R, Int} = 0.999,
                AJDalgorithm :: Symbol = :QNLogLike,
                AJDmaxIter :: Int = 2000,
                verbose :: Bool = true) where R<:Real

      # Covariance matrix of ERP per condition
      𝐌 = mean(Xerp, wl, markErp, erpOverlap; weights=erpWeights)

      # Mean ERP covariances
      if          erpCovEst == :nrtme
                  𝐂 = [nrtme(M'; maxiter=500, verbose = verbose) for M ∈ 𝐌]
      elseif      erpCovEst == :tme
                  𝐂 = [tme(M'; maxiter=500, verbose = verbose) for M ∈ 𝐌]
      elseif      erpCovEst == :lse
                  𝐂 = _cov(𝐌; covEst = LinearShrinkage(ConstantCorrelation()),
                               dims = 1, meanX = 0)
      elseif      erpCovEst == :scm
                  𝐂 = _cov(𝐌; covEst = SimpleCovariance(), dims = 1, meanX = 0)
      else trhow(ArgumentError("`function `BSS`: keyword argument `erpCovEst` can be :scm, :lse, :tme or :nrtme"))
      end

      # average cospectra across all ERP trials in region [fmin, fmax]
      𝗦 = trialsCospectra(Xcospectra, markCospectra, sr, wl, cospectraBandPass...;
                  tapering=cospectraTapering, nonlinear = false)

      # Pre-whitening: noralize AJD set to unit trace. Base whitening on mean
      # ERP covariances regularized by mean cospectra
      set=vcat(𝗦, 𝐂)
      for S ∈ set S=tr1(S) end
      W=whitening(ℍ(tr1(mean(𝗦))*0.1+tr1(mean(𝐂))*0.9); eVar=whiteningeVar)
      nsources=minimum(size(W.F))
      slabels=[string(i) for i=1:nsources]

      # Whitened Mean ERP covariances
      W𝐂=ℍVector([ℍ(W.F'*C*W.F) for C∈𝐂])
      nERPcov=length(W𝐂)

      # Whitened cospectra
      W𝗦=ℍVector([ℍ(W.F'*S*W.F) for S∈𝗦])
      ncospectra=length(W𝗦)

      # get whitened spectra from cospectra (for plotting)
      wspectra=[W𝗦[i][j, j] for i=1:ncospectra, j=1:nsources]

      # cospectra smoothed non-diagonality weigths
      ndw=hannSmooth([nonD(S) for S ∈ W𝗦])

      # create plot of whitened cospectra and non-diagonality weights
      p1=plot(wspectra, labels=reshape(sensors, 1, length(sensors)),
                title="Whitened Spectra");
      p2=plot(ndw, legend=false,
                title="Non-Diagonality");
      plot1=(p1, p2)

      # uniform weights for ERP covariances
      mw=ones(Float64, nERPcov)
      # alternatively, give as weights the square root of number of trials
      # mw=[√(length(cleanmark[i])) for i=1:length(cleanmark)]

      # concatenate weights and create a StatsBase weights object
      w=weights([ndw/sum(ndw); mw/sum(mw)])

      # AJDset: whitened covariance matrices of average ERPs and cospectra
      Wset=vcat(W𝗦, W𝐂)
      for S ∈ Wset S=tr1(S) end

      # do AJD
      J=ajd(Wset;
            w=w, algorithm=AJDalgorithm, maxiter=AJDmaxIter, verbose=verbose)

      # demixing(B) and mixing(A) matrix
      B=W.F*J.F
      A=J.iF*W.iF

      # explained variance of the mean ERP energy (DISMISSED; use RATIO)
      #expVar=[evar(A, B, C, i) for i=1:nsources, C∈𝐂]
      #normalizeMean!(expVar; dims=1)
      # find key for sorting the average explained variance in desc. order
      #p=sortperm(reshape(sum(expVar, dims=2), :); rev=true)

      # explained variance RATIO of the mean ERP energy / mean Cospectra energy
      expVarERP=[evar(A, B, C, i) for i=1:nsources, C∈𝐂]
      normalizeMean!(expVarERP; dims=1)
      expVarBSS=[evar(A, B, C, i) for i=1:nsources, C∈𝗦]
      normalizeMean!(expVarBSS; dims=1)
      expVar=sum(expVarERP, dims=2)./sum(expVarBSS, dims=2)
      # find key for sorting the explained variance in desc. order
      p=sortperm(reshape(expVar, :); rev=true)

      # sort the columns of B, rows of A and rows of expVar using this key
      B=B[:, p]
      A=A[p, :]
      expVar[:]=expVar[p, :]

      # create plot of expected variance
      plot2 = plot(expVar, xticks = 1:1:nsources,
                    xtickfontsize=12, ytickfontsize=12,
                    legend=:none, title="source SNR");


      # Diagonal matrix with the square root of the mean source variance
      # across ERP means for all sources. Used to normalize A and B so that
      # B gives sources with unit mean variance
      D=Diagonal(inv.(sqrt.(mean([sum(abs2.(m)) for m ∈ eachcol(M*B)] for M ∈ 𝐌))))
      B[:]=B*D
      A[:]=inv(D)*A
      sourceERP=[M*B for M ∈ 𝐌]
      # check: the source variance must be now the identity matrix
      #D1=Diagonal(sqrt.(mean([sum(abs2.(m)) for m ∈ eachcol(M)] for M ∈ sourceERP)))

      # compute the reprojection in the sensor space of all sources
      reproMeans=[[(M*B[:, i])*A[i, :]' for M ∈ 𝐌] for i=1:nsources]

      # find out the sign of sources making them correlating with the
      # mean reprojected source and apply it to B, A, and sourceERP.
      # BEWARE: there is a chance that the repro EEG is bipolar (pos and neg),
      # in which case this solution may fail.
      for i=1:nsources
            meanSourceERPmax=mean(sourceERP)[:, i]
            meanReproMeans=reshape(mean(mean(reproMeans[i]); dims=2), :)
            c = cor(meanSourceERPmax, meanReproMeans)
            D[i, i] = c < 0. ? -1.0 : 1.0
      end
      B[:]=B*D
      A[:]=D*A
      for S∈sourceERP S[:]=S*D end

      return 𝐌, 𝐂, 𝗦, A, B, nsources, slabels, expVar, sourceERP, reproMeans,
             plot1, plot2
end




end # module


#=
push!(LOAD_PATH, homedir()*"\\Documents\\Code\\julia\\Modules")
using EEGpreprocessing, EEGio, EEGtopoPlot, System
X=Matrix(readASCII("C:\\temp\\data")')
XTt=Matrix(readASCII("C:\\temp\\XTt")')
stims=[Vector{Int64}(readASCII("C:\\temp\\stim1")[:]), Vector{Int}(readASCII("C:\\temp\\stim2")[:]) ]
A=mulTX(X, stims, 128)
using LinearAlgebra
norm(A-XTt)
=#
