module EEGio
using NPZ, YAML, HDF5, EzXML, FileSystem, EEGpreprocessing, DSP, ERPs

# ? ¤ CONTENT ¤ ? #

# STRUCTURES
# EEG | holds data and metadata of an EEG recording

# FUNCTIONS:
# CreateEEG | create an EEG object with minimal arguments
# loadNYdb | return a list of .npz files in a directory
# readNY   | read an EEG recording in NY (npz, yml) format
# readASCII (2 methods) | read one ASCII file or all ASCII files in a directory
# writeASCII            | write one abstractArray dara matrix in ASCII format

# max number of elements in an EEG matrix that can be handled by ICoN
const titleFont     = "\x1b[95m"
const separatorFont = "\x1b[35m"
const defaultFont   = "\x1b[0m"
const greyFont      = "\x1b[90m"

import Statistics: mean

export
    EEG,
    loadNYdb,
    readNY,
    parseXML,
    readgTec,
    readASCII,
    writeASCII

# `EEG` structure holding data and metadata information for an EEG recording
# An instance is created by the `readNY` function, which reads into files
# in NY (npz, yml) format.
# Fundamental fields can be accessed directly, for example, if `o` is an
# instance of the structure, the EEG data is in `o.X`.
# All metadata can be accessed in the dictionaries. For reading them
# use for example syntax `o.acquisition["ground"]`
# formatversion: 0.0.1
struct EEG
    id            :: Dict{Any,Any} # `id` Dictionary of the .yml file
    # it includes keys:   "run", "other", "database", "subject", "session"
    acquisition   :: Dict{Any,Any} # `acquisition` Dictionary of the .yml file
    # it includes keys:   "sensors", "software", "ground", "reference",
    #                      "filter", "sensortype", "samplingrate", "hardware"
    documentation :: Dict{Any,Any} # `acquisition` Dictionary of the .yml file
    # it includes keys:   "doi", "repository", "description"
    formatversion :: String        # `formatversion` field of the .yml file

    # the following fields are what is useful in practice
    db            :: String        # name of the database to which this file belongs
    subject       :: Int           # serial number of the subject in database
    session       :: Int           # serial number of the session of this subject
    run           :: Int           # serial number of the run of this session
    sensors       :: Vector{String}# electrode leads on the scalp in standard 10-10 notation
    sr            :: Int           # sampling rate
    ne            :: Int           # number of electrodes (excluding reference and ground)
    ns            :: Int           # number of samples
    wl            :: Int           # window length: typically, the duration of the trials
    offset        :: Int           # each trial start at `stim` sample + offset
    nc            :: Int           # number of classes
    clabels       :: Vector{String} # class labels given as strings
    stim          :: Vector{Int}    # stimulations for each sample (0, 1, 2...). 0 means no stimulation
    mark          :: Vector{Vector{Int}}  # markers (in sample) for class 1, 2...
    y             :: Vector{Int}          # the vectors in `mark` concatenated
    X             :: Matrix{T} where T<:Real # whole recording EEG data (ns x ne)
    trials        :: Union{Vector{Matrix{T}}, Nothing} where T<:Real # all trials in order ot `stims` (optional)
end

# Create an EEG New York object given ad minima the EEG data `X`,
# sampling rate `sr` and the sensor list `sensors`.
# Other useful fields can be given as OKA (Optional Keyword Arguments)
# This constructors does not fill the dictionaries that are read
# from a New York file, it only fills the useful fields.
EEG(X::Matrix{T}, sr::Int, sensors::Vector{String};
    db::String="",
    subject::Int=0,
    session::Int=1,
    run::Int=1,
    wl::Int=sr,
    offset::Int=0,
    nc::Int=1,
    clabels::Vector{String}=[""],
    stim::Vector{Int}=["0"],
    mark::Vector{Vector{Int}}=[[""]],
    y::Vector{Int}=[0]) where T<:Real =
    EEG(Dict(), Dict(), Dict(), "0.0.1", db, subject,
        session, run, sensors, sr, size(X, 2), size(X, 1), wl, offset,
        nc, clabels, stim, mark, y, X, nothing)


# Return a list of the complete path of all .npz files
# in directory `DBdir` for which a corresponding .yml file exist.
# If a string is provided as `isin`, only the files whose name
# contains the string will be included. See FileSytem.jl.getFilesInDir
function loadNYdb(dbDir=AbstractString, isin::String="")
  # create a list of all .npz files found in dbDir (complete path)
  npzFiles=getFilesInDir(dbDir; ext=(".npz", ), isin=isin)

  # check if for each .npz file there is a corresponding .yml file
  missingYML=[i for i ∈ eachindex(npzFiles) if !isfile(splitext(npzFiles[i])[1]*".yml")]
  if !isempty(missingYML)
    @warn "the following .yml files have not been found:\n"
    for i ∈ missingYML println(splitext(npzFiles[i])[1]*".yml") end
    deleteat!(npzFiles, missingYML)
    println("\n $(length(npzFiles)) files have been retained.")
  end
  return npzFiles
end

# Read EEG data in NY (npz, yml) format and create an `EEE` structure.
# The complete path of the file given by `filename`.
#  Either the .npz or the .yml file can be passed.
# If a 2-tuple is passed as `bandpass`, data is filtered in the bandpass.
# If a fractional or integer number is given as `resample`, the data is
# resampled, e.g., `resample=1//2` will downsample by half and `resample=3`
# will upsample by 3.
# If `upperLimit` is different fro zero, this is used as argument in function
# ERPS.jl.reject to perform artefact rejection.
# If `upperLimit` is different from zero, this is used as argument in function
# ERPs.jl.reject to perform artefact rejection.
# If `getTrials` is true, the `trials` field of the `EEG` structure is filled.
# If `msg` is not empty, print `msg` on exit.
# NB If the field `offset` of the NY file is different from zero,
# the stimulations in `stim` and markers in `mark` will be shifted one
# with respect to the other: `stim` will be as read, while markers will
# apply the offset. See `ERPs.jl.mark2stim` to derive stimulations that
# are synchronized with the markers.
# As a rule of thumb, only markers should be use for data analysis.
function readNY(filename  :: AbstractString;
                bandpass  :: Tuple=(),
                resample  :: Union{Rational, Int}=1,
                upperLimit:: Union{Real, Int} = 0,
                getTrials :: Bool=true,
                msg       :: String="")

  data = npzread(splitext(filename)[1]*".npz") # read data file
  info = YAML.load(open(splitext(filename)[1]*".yml")) # read info file

  sr      = info["acquisition"]["samplingrate"]
  stim    = data["stim"]                  # stimulations
  (ns, ne)= size(data["data"])            # of sample, # of electrodes)
  os      = info["stim"]["offset"]        # offset for trial starting sample
  wl      = info["stim"]["windowlength"]  # trial duration
  nc      = info["stim"]["nclasses"]      # of classes

  # band-pass the data if requested
  if isempty(bandpass)
    X=data["data"]
  else
    BPfilter = digitalfilter(Bandpass(first(bandpass)/(sr/2), last(bandpass)/(sr/2)), Butterworth(4))
    X        = filtfilt(BPfilter, data["data"])
  end

  # resample data if requested
  if resample≠1
    #X, stim   = resample(X, resample; stim=stim)
    X, stim   = resample2(X, sr, resample; stim=stim)
    (ns, ne)  = size(X)
    wl        = round(Int, wl*resample)
    os        = round(Int, os*resample)
    sr        = round(Int, sr*resample)
  end

  stim=Vector{Int64}(stim)

  if upperLimit≠0
      # artefact rejection; change stim and compute mark
      stim, rejecstim, mark, rejecmark, rejected, rejectionPlot =
            reject(X, stim, wl; offset=os, upperLimit=upperLimit)
  else
      # only mark, i.e., samples where the trials start for each class 1, 2,...
      mark=stim2mark(stim, wl; offset=os)
      #mark=[[i+os for i in eachindex(stim) if stim[i]==j && i+os+wl<=ns] for j=1:nc]
  end

    #   println(maximum.(mark))
    #   println(wl)
    # Replaced one
    #   trials = getTrials ?  [X[mark[i][j]+os:mark[i][j]+os+wl-1,:] for i=1:nc for j=1:length(mark[i])] :
    #                         nothing
  trials = getTrials ?  [X[mark[i][j]:mark[i][j]+wl-1,:] for i=1:nc for j=1:length(mark[i])] :
                        nothing

  if !isempty(msg) println(msg) end

  # this creates the `EEG` structure
  EEG(
     info["id"],
     info["acquisition"],
     info["documentation"],
     info["formatversion"],

     info["id"]["database"],
     info["id"]["subject"],
     info["id"]["session"],
     info["id"]["run"],
     info["acquisition"]["sensors"],
     sr,
     ne,
     ns,
     wl,
     os, # trials offset
     nc,
     collect(keys(info["stim"]["labels"])), # clabels
     stim,
     mark,
     [i for i=1:nc for j=1:length(mark[i])], # y: all labels
     X, # whole EEG recording
     trials # all trials, by class
  )

end

## #######################  g.Tec HDF5 files ############################## ##

# Parse one-string XML files.
# If `verbose` is true (default) the file is shown in the REPL
function parseXML(xmlStringVector::String; verbose::Bool=true)
      doc = EzXML.parsexml(xmlStringVector)
      verbose && print(doc)
      return doc
end

# Read an EEG data from a HDF5 file saved by g.Tec g.Recorder software.
# `fileName` is the complete path of the .hdf5 file to be read.
# The data by default is of the Float32 type. Use `dataType` to convert it
# in another format.
# If `writeMetaDataFiles` is true (default) all metadata files will be
# saved as xml or text files, in the same director where `filename` is
# with the same name to which a suffix indicating the type of metadata
# will be appended.
# If `writeMetaDataFiles` is true AND `verbose` is true (default)
# the metadata will be shown in the REPL.
# If `skipFirstSamples` is greater than 0 (default), this number of samples
# will not be read at the beginning of the file. With the g.USBamp the first
# few seconds should always be removed as the amplifier stabilizes after a
# a few seconds.
# If a `chRange` range is provided (e.g., 1:10), only this range of channels
# will be read.
# The function returns the data as a Matrix{`dataType`, 2}, with as many
# lines as samples and as many columns as channels.
function readgTec(fileName::AbstractString;
                    dataType::Type=Float32,
                    writeMetaDataFiles::Bool=true,
                    verbose::Bool=true,
                    skipFirstSamples::Int=0,
                    chRange::Union{UnitRange, Symbol}=:All)

    fid = h5open(fileName, "r")
        if writeMetaDataFiles
              write(splitext(fileName)[1]*"_acqXML.xml", parseXML(read(fid["RawData/AcquisitionTaskDescription"])[1], verbose=verbose))
              write(splitext(fileName)[1]*"_chProp.xml", parseXML(read(fid["RawData/DAQDeviceCapabilities"])[1], verbose=verbose))
              write(splitext(fileName)[1]*"_chUnits.xml", parseXML(read(fid["RawData/DAQDeviceDescription"])[1], verbose=verbose))
              write(splitext(fileName)[1]*"_sessDescr.xml", parseXML(read(fid["RawData/SessionDescription"])[1], verbose=verbose))
              write(splitext(fileName)[1]*"_subjDescr.xml", parseXML(read(fid["RawData/SubjectDescription"])[1], verbose=verbose))
              write(splitext(fileName)[1]*"_triggers.xml", parseXML(read(fid["AsynchronData/AsynchronSignalTypes"])[1], verbose=verbose))
              verbose && println("features: ", read(fid["SavedFeatues/NumberOfFeatures"]))
              writeVector(read(fid["SavedFeatues/NumberOfFeatures"]), splitext(fileName)[1]*"_features.txt"; overwrite=true)
              verbose && println("version: ", read(fid["Version/Version"])[1])
              writeVector(read(fid["Version/Version"]), splitext(fileName)[1]*"_version.txt"; overwrite=true)
        end
        # convert data to Float64 and transpose
        data=Matrix(Array{dataType, 2}(read(fid["RawData/Samples"]))')
        println("\nHere we go...")
        println("# Channels: ", chRange==:All ? size(data, 2) : length(chRange))
        println("# Samples: ", size(data, 1)-skipFirstSamples)
    close(fid)

    return chRange==:All ? data[1+skipFirstSamples:end, :] : data[1+skipFirstSamples:end, chRange]
end


## Methods for EEG Structure

# see ERSs.jl
mean(o::EEG;
        overlapping :: Bool = false,
        weights     :: Union{Vector{R}, Vector{Vector{R}}, Symbol} = :none,
        mark        :: Union{Vector{S}, Vector{Vector{S}}, Nothing} = nothing,
        offset      :: S = 0) where {R<:Real, S<:Int} =
    mean(o.X, o.wl, mark===nothing ? o.mark : mark;
            overlapping = overlapping, offset = offset, weights = weights)


# overwrite the Base.show function to nicely print information
# about the sturcure in the REPL
# ++++++++++++++++++++  Show override  +++++++++++++++++++ # (REPL output)
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, o::EEG)
    r, c=size(o.X)
    type=eltype(o.X)
    l=length(o.stim)
    println(io, titleFont, "∿ EEG Data type; $r x $c ")
    println(io, separatorFont, "∼∽∿∽∽∽∿∼∿∽∿∽∿∿∿∼∼∽∿∼∽∽∿∼∽∽∼∿∼∿∿∽∿∽∼∽", greyFont)
    println(io, "NY format info:")
    println(io, "Dict: id, acquisition, documentation")
    println(io, "formatversion   : $(o.formatversion)")
    println(io, separatorFont, "∼∽∿∽∽∽∿∼∿∽∿∽∿∿∿∼∼∽∿∼∽∽∿∼∽∽∼∿∼∿∿∽∿∽∼∽", defaultFont)
    println(io, "db (database)   : $(o.db)")
    println(io, "subject         : $(o.subject)")
    println(io, "session         : $(o.session)")
    println(io, "run             : $(o.run)")
    println(io, "sensors         : $(length(o.sensors))-Vector{String}")
    println(io, "sr(samp. rate)  : $(o.sr)")
    println(io, "ne(# electrodes): $(o.ne)")
    println(io, "ns(# samples)   : $(o.ns)")
    println(io, "wl(win. length) : $(o.wl)")
    println(io, "offset          : $(o.offset)")
    println(io, "nc(# classes)   : $(o.nc)")
    println(io, "clabels(c=class): $(length(o.clabels))-Vector{String}")
    println(io, "stim(ulations)  : $(length(o.stim))-Vector{Int}")
    println(io, "mark(ers) : $([length(o.mark[i]) for i=1:length(o.mark)])-Vectors{Int}")
    println(io, "y (all c labels): $(length(o.y))-Vector{Int}")
    println(io, "X (EEG data)    : $(r)x$(c)-Matrix{$(type)}")
    o.trials==nothing ? println("                : nothing") :
    println(io, "trials          : $(length(o.trials))-Matrix{$(type)}")
    r≠l && @warn "number of class labels in y does not match the data size in X" l r
end


##

# read EEG data from a .txt file in LORETA format and put it in a matrix
# of dimension txn, where n=#electrodes and t=#samples.
# If optional keyword argument `msg` is not empty, print `msg` on exit.
function readASCII(fileName::AbstractString; msg::String="")
    if !isfile(fileName)
        @error "function `readASCII`: file not found" fileName
        return nothing
    end

    S=readlines(fileName) # read the lines of the file as a vector of strings
    filter!(!isempty, S)
    t=length(S) # number of samples
    n=length(split(S[1])) # get the number of electrodes
    X=Matrix{Float64}(undef, t, n) # declare the X Matrix
    for j=1:t
        x=split(S[j]) # this get the n potentials from a string
        for i=1:n
            X[j, i]=parse(Float64, replace(x[i], "," => "."))
        end
    end
    if !isempty(msg) println(msg) end
    return X
end

# Read several EEG data from .txt files in LORETA format given in `filenames`
# (a Vector of strings) and put them in a vector of matrices object.
# `skip` is an optional vector of serial numbers of files in `filenames` to skip.
# print: "read file "*[filenumber]*": "*[filename] after each file has been read.
function readASCII(fileNames::Vector{String}, skip::Vector{Int}=[0])
        X = [readASCII(fileNames[f]; msg="read file $f: "*basename(fileNames[f])) for f in eachindex(fileNames) if f ∉ skip]
        skip≠[0] && println("skypped files: ", skip)
        return X
end


# Write an EEG data matrix into a text ASCII file in LORETA tabular format
# (# of samples x # of electrodes).
# The data is written as `filename` which must be a complete path from root.
# If `filename` already exists, if `overwrite` is true the file will be
# overwritten, otherwise a warning in printed and nothing is done.``
# `SamplesRange` is a UnitRange delimiting the samples (rows of `X`) to be written.
# If optional keyword argument `msg` is not empty, print `msg` on exit
function writeASCII(X::Matrix{T}, fileName::String;
              samplesRange::UnitRange=1:size(X, 1),
              overwrite::Bool=false,
              digits=9,
              msg::String="") where T <: Real

    if isfile(fileName) && !overwrite
        @error "writeASCII function: `filename` already exists. Use argument `overwrite` if you want to overwrite it."
    else
        io = open(fileName, "w")
        write(io, replace(chop(string(round.(X[samplesRange, :]; digits=digits)); head=1, tail=1), ";" =>"\r\n" ))
        close(io)
        if !isempty(msg) println(msg) end
    end
end


function writeASCII(X::Vector{T}, fileName::String;
    overwrite::Bool=false,
    msg::String="") where T <: String

    if isfile(fileName) && !overwrite
        @error "writeASCII function: `filename` already exists. Use argument `overwrite` if you want to overwrite it."
    else
        io = open(fileName, "w")
        for s∈X 
            write(io, s, "\r\n" )
        end
        close(io)
        if !isempty(msg) println(msg) end
    end
end


end # module

# Example
# dir="C:\\Users\\congedom\\Documents\\My Data\\EEG data\\NTE 84 Norms"

# Gat all file names with complete path
# S=getFilesInDir(dir) # in FileSystem.jl

# Gat all file names with complete path with extension ".txt"
# S=getFilesInDir(@__DIR__; ext=(".txt", ))

# read one file of NTE database and put it in a Matrix object
# X=readASCII(S[1])

# read all files of NTE database and put them in a vector of matrix
# 𝐗=readASCII(S)
