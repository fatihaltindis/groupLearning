using Pkg
Pkg.activate("groupL")

# Add packages
using LinearAlgebra, CovarianceEstimation,
PosDefManifold, PosDefManifoldML, Plots,
Diagonalizations, NPZ, YAML, HDF5, EzXML,
BlockDiagonals, DelimitedFiles, TimerOutputs, 
Base.Threads

# Add custom packages 
push!(LOAD_PATH, joinpath(@__DIR__, "Modules"))
using EEGio, System, FileSystem, 
EEGpreprocessing, EEGprocessing, ERPs

include(".\\src\\pipelineTools.jl");

# Main folder of the database
filepath = joinpath(@__DIR__, "exampleData");
# Folder name of the selected database
dbName = "bi2015a";

# Parameters
PCA_dim = nothing;
random_state = 1;
n_of_subject = 10;
selection_rule = :none;
threshold = 0.66;
turn = :local;
random_state = 1;
bandpass = (1,16);
artefact_rej = true;
n_splits = 5;
split_ratio = 50;
whitening = :svd;
initialize_U = :smart;
paradigm = :ERP;
verbose = true;

obj_list = initiateObjects(dbName, filepath;
                           n_of_subject=n_of_subject, PCA_dim=PCA_dim, verbose=true,
                           split_ratio=split_ratio, n_splits=n_splits,
                           whitening=whitening, initialize_U=initialize_U);

pipeline = [createTSVectors, trainSW, prepareGL, runGL, trainGL];
runPipe!(pipeline, obj_list)

pipeline2 = [runLeaveOut, trainFA];
runPipe!(pipeline2, obj_list)

plotAcc(obj_list)



pipe2 = [prepareGL, runGL, trainGL];
runPipe!(pipe2, obj_list)

pipe3 = [createTSVectors, trainSW, prepareGL, runLeaveOut];
runPipe!(pipe3, obj_list)

pipe4 =[trainFA];
runPipe!(pipe4, obj_list)


