

cd("G:\\Mon Drive\\PhD\\Codes\\groupLearning")

using 	LinearAlgebra, CovarianceEstimation,
PosDefManifold, PosDefManifoldML,
Diagonalizations, NPZ, YAML, HDF5, EzXML,
BlockDiagonals, DelimitedFiles, TimerOutputs, Base.Threads

# Include required custom packages 
push!(LOAD_PATH, joinpath(@__DIR__, "Modules"))
using 	EEGio, System, FileSystem, EEGpreprocessing, EEGprocessing, ERPs

include(".\\src\\pipelineTools.jl");

filepath = "G:\\Mon Drive\\PhD\\Codes\\julia_codes\\group-learning\\Group-Learning-BCI\\ERP Analysis\\DataBases\\P300";
dbName = "bi2014a";

PCA_dim = 8;
random_state = 1;
n_of_subject = 5;
selection_rule = :none;
threshold = 0.66;
turn = :local;
random_state = 1;
bandpass = (1,16);
artefact_rej = true;
n_splits = 4;
split_ratio = 40;
paradigm = :ERP;

obj_list = initiateObjects(dbName, filepath;
                           n_of_subject=n_of_subject, PCA_dim=PCA_dim, verbose=true);

pipeline = [createTSVectors, trainSW, prepareGL, runGL, trainGL];

runPipe!(pipeline, obj_list)

plotAcc(obj_list)

