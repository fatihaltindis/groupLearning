module EEGpreprocessing

using StatsBase, Statistics, LinearAlgebra, DSP, PosDefManifold

# functions:

import DSP:resample

export
    standardizeEEG,
    resample,
    resample2,
    removeChannels,
    removeSamples


# standardize the whole data in `X` using the winsor mean and std dev.
# the winsor statistics are computed in module StatsBase excluding
# the `prop` proportion of data at both sides.
# If the data elements are not Float64, they are converted to Float64.
function standardizeEEG(X::AbstractArray{T}; prop::Real=0.25) where T<:Real
    vec=X[:]
    μ=mean(winsor(vec; prop=prop))
    σ=√trimvar(vec; prop=prop)
    return eltype(X) === Float64 ? ((X.-μ)./σ) : (convert(Array{Float64}, (X.-μ)./σ))
end



# integer or rational rate resampling using Kaiser FIR filters (from DSP.jl).
# rate is an integer or rational number, e.g., 1//4 downsample by 4
# rel_bw and attenuation are parameter of the FIR filter, see line 637
# in function resample_filter in https://github.com/JuliaDSP/DSP.jl/blob/master/src/Filters/design.jl
# `X` is a tall data matrix with samples along rows.
# If stim is passed as OKA, it must be a vector of as many integers as samples
# in `X`, holding 0 (zero) for samples when there was no stimulation and
# a natural number for samples where there was a stimulation.
# For resampling stimulations, blocks of rate or 1/rate samples are considered
# and if a stimulation appears in those blocks, it is rewritten in the
# first position of the resampled block.
# NB: for the moment being only rational number with one at the nominator
# are supported! (Int(inv(rate)) is computed)
# Examples: Y=resample(X, 1//4); Y=resample(X, 4);
# Y, newstim=resample(X, 4; stim=s);
function resample(X::Matrix{T},
                  rate::Union{S, Rational};
                  rel_bw::Float64 = 1.0,
                  attenuation::Int = 60,
                  stim::Union{Vector{S}, Nothing} = nothing) where {T<:Real, S<:Int}


    if rate==1 return stim===nothing ? X : (X, stim) end

    # resample data
    ne = size(X, 2) # of electrodes
    h = DSP.resample_filter(rate, rel_bw, attenuation)
    # first see how long will be the resampled data
    x = DSP.resample(X[:, 1], rate, h)
    t = length(x)
    Y = Matrix{eltype(X)}(undef, t, ne)
    Y[:, 1] = x
    for i=2:ne
        x = DSP.resample(X[:, i], rate, h)
        Y[:, i] = x
    end

    # if downsampling check that the number of samples in Y is a multiple
    # of 1/rate. If not padd zeros or remove samples. Change stim accordingly.
    if rate<1
        i=Int(inv(rate))
        r, c = size(Y)
        global t=(r ÷ i) * i
        if r > t Y = Y[1:t, :] end
        if r < t Y = [Y; zeros(eltype(Y), t-r, c)] end
        if stim≠nothing
            r=length(stim)
            if r > t*i stim=stim[1:t*i] end
            if r < t*i stim=[stim; zeros(eltype(stim), (t*i)-r)] end
        end
    end

    # check that upsampling constructs an exact multiple number of samples
    if rate>1
      diff=size(X, 1)*rate-size(Y, 1)
      if diff>0 Y=vcat(Y, zeros(eltype(Y), diff, size(Y, 2))) end
      if diff<0 Y=Y[1:size(X, 1)*rate, :] end
    end

    # resample stimulation channel
    if stim≠nothing
        l=length(stim)
        if rate<1
            # downsample
            irate=Int(inv(rate))
            s=reshape(stim[1:t*irate], (irate, :))'
            u=[filter(x->x≠0, s[i, :]) for i=1:size(s, 1)]
            for i=1:length(u)
                if length(u[i])>1
                    @error "function `resampling`: the interval between stimulations does not allow the desired downsampling of the stimulation channel" rate
                    return Y, nothing
                end
            end
            newstim=[isempty(v) ? 0 : v[1] for v ∈ u]
        else
            # upsample
            r=vcat(stim, zeros(eltype(stim), l*(rate-1)))
            newstim=reshape(reshape(r, (l, rate))', (l*rate))
        end
        length(newstim)≠size(Y, 1) && @warn "the size of the resampled data and stimulation channel do not match" size(Y, 1) length(newstim)
    end

    return stim===nothing ? Y : (Y, newstim)
end


function resample2(X::Matrix{T},
                  sr::Int,
                  rate::Union{S, Rational};
                  rel_bw::Float64 = 1.0,
                  attenuation::Int = 60,
                  stim::Union{Vector{S}, Nothing} = nothing) where {T<:Real, S<:Int}


    if rate==1 return stim===nothing ? X : (X, stim) end
    newsr=round(Int, sr/rate)
    sr/rate-newsr≠0 && throw(ArgumentError("resample function: sr/rate must be an integer"))

    # resample data
    ne = size(X, 2) # of electrodes
    h = DSP.resample_filter(rate, rel_bw, attenuation)
    # first see how long will be the resampled data
    x = DSP.resample(X[:, 1], rate, h)
    t = length(x)
    Y = Matrix{eltype(X)}(undef, t, ne)
    Y[:, 1] = x
    for i=2:ne
        Y[:, i] = DSP.resample(X[:, i], rate, h)
    end

    # resample stimulation channel
    if stim≠nothing
        newstim=zeros(Int, t)
        for i=1:lenght(stim)
            if stim[i]≠0
                newsample=clamp(round(Int, i/sr*newsr), 1, t)
                newstim[newsample]≠0 && @warn "resample function: several stimulations are resamples in the same position!" newsample
                newstim[newsample]=stim[i]
            end
        end
    end
    return stim===nothing ? Y : (Y, newstim)

end




# Remove one or more channels from data matrix `X`.
# To remove one channel pass an Integer (1-based serial number) as `what`.
# To remove several channels pass a vector of integers as `what`.
# Besides `X` and `what`, you must pass a Vector{String} of sensor labels
# holding as many elements as channels in `X`.
# The function finds out if the channels are along the columns or lines of `X`.
# RETURN the 3-tuple newX, s, ne
# where s is the updated sensor labels vector and
# ne is the updated number of channels in `X`
# EXAMPLES:
# X, sensors, ne = removeChannels(X, 2, sensors)
# X, sensors, ne = removeChannels(X, collect(1:5), sensors)
# X, sensors, ne = removeChannels(X, findfirst(x->x=="Cz", sensors), sensors)
function removeChannels(X::Matrix{T}, what::Union{Int, Vector{S}},
                       sensors::Vector{String}) where {T<:Real, S<:Int}
    di = findfirst(length(sensors).==(size(X)))
    X = remove(X, what; dims=di)
    return X, remove(sensors, what), size(X, di)
end

# Remove one or more samples from data matrix `X`.
# To remove one sample pass an Integer (1-based serial number) as `what`.
# To remove several samples pass a vector of integers as `what`.
# Besides `X` and `what`, you must pass a Vector{String} of stimulations
# holding as many elements as samples in `X`.
# The function finds out if the samples are along the columns or lines of `X`.
# RETURN the 3-tuple newX, s, ns
# where s is the updated stim vector and
# ns is the updated number of samples in `X`
# EXAMPLES:
# X, stim, ne = removeSamples(X, 2, stim)
# X, stim, ne = removeSamples(X, collect(1:128), stim)
# X, stim, ne = removeSamples(X, collect(1:2:length(stim)), stim)
function removeSamples(X::Matrix{T}, what::Union{Int, Vector{S}},
                       stim::Vector{String}) where {T<:Real, S<:Int}
    di = findfirst(length(stim).==(size(X)))
    X = remove(X, what; dims=di)
    return X, remove(stim, what), size(X, di)
end




end # module


# useful code:

# of classes, 0 is no stimulation and is excluded
# z=length(unique(stim))-1

# vector with number of stim. for each class 1, 2, ...
# nTrials=counts(stim, z) # or: [count(x->x==i, stim) for i=1:z]
