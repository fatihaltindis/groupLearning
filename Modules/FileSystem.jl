#v0.0.1
module FileSystem

# functions:

# getFilesInDir | get all files in a directory 
# getFoldersInDir | get all folders in a directory
# writeVector | write a vector of numbers as an ASCII file

export
    getFilesInDir,
    getFoldersInDir,
    writeVector

    # Get the complete path of all files in `dir` as a vector of strings
    # `ext` is an optional tuple of file extensions given strings.
    # If it is provided, only files with those extensions will be included
    # in the returned vector. The extensions must be entered in lowercase.
    # If a string is provided as `isin`, only the files whose name
    # contains the string will be included
    ## Examples
    # S=getFilesInDir(@__DIR__)
    # S=getFilesInDir(@__DIR__; ext=(".txt", ))
    # S=getFilesInDir(@__DIR__; ext=(".txt", ".jl"), isin="Analysis")
    getFilesInDir(dir::String; ext::Tuple=(), isin::String="") =
    if !isdir(dir) @error "Function `getFilesInDir`: input directory is incorrect!"
    else
        S=[]
        for (root, dirs, files) in walkdir(dir)
            if root==dir
                for file in files
                    if ext==() || ( lowercase(string(splitext(file)[2])) ∈ ext )
                        if occursin(isin, file) # if isin=="" this is always true
                            push!(S, joinpath(root, file)) # complete path and file name
                        end
                    end
                end
            end
        end
        isempty(S) && @warn "Function `getFilesInDir`: input directory does not contain any files"
        return Vector{String}(S)
    end


    # Get the complete path of all folders in `dir` as a vector of strings.
    # If a string is provided as `isin`, only the folders whose name
    # contains the string will be included
    ## Examples
    # S=getFoldersInDir(@__DIR__)
    # S=getFoldersInDir(@__DIR__; isin="Analysis")
    getFoldersInDir(dir::String; isin::String="") =
    if !isdir(dir) @error "Function `getFoldersInDir`: input directory is incorrect!"
    else
        S=[]
        for (root, dirs, files) in walkdir(dir)
            if root==dir
                for dir in dirs
                    if occursin(isin, dir) # if isin=="" this is always true
                        push!(S, joinpath(root, dir)) # complete path and file name
                    end
                end
            end
        end
        isempty(S) && @warn "Function `getFoldersInDir`: input directory does not contain any folders"
        return Vector{String}(S)
    end

    # Write a vector of any elements as an ASCII file of strings.
    # NB: This is meant for numbers, thus all spaces are removed!
    function writeVector(v::Vector{T}, fileName::String;
                  samplesRange::UnitRange=1:size(v, 1),
                  overwrite::Bool=false,
                  msg::String="") where T

        if isfile(fileName) && !overwrite
            @error "writeVector function: `filename` already exists. Use argument `overwrite` if you want to overwrite it."
        else
            io = open(fileName, "w")

            #write(io, replace(chop(string(v[samplesRange]); head=1, tail=1), "," =>"\r\n" ))
            write(io, replace(replace(strip(string(v[samplesRange]), ['[', ']']), " " =>""), "," =>"\r\n" ))
            close(io)
            if !isempty(msg) println(msg) end
        end
    end


end # module
