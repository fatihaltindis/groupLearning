# Unit pipelineTools.jl, part of groupLearning Package for julia
#
# MIT License 
# version: 10 Sept 2022
# Copyright (c) - 2023
# Fatih Altindis and Marco Congedo
# Abdullah Gul University, Kayseri
# GIPSA-lab, CNRS, University Grenoble Alpes

include("pipelineObjects.jl")
include("databaseUtilityTools.jl")
include("dataTransformation.jl")
include("covarianceEstimation.jl")
include("erpTools.jl")
include("tangentSpaceVectors.jl")
include("utils.jl")
include("galia.jl")
include("learning.jl")
include("dbInfo.jl")

function createTSVectors(db_obj     :: Database, 
                         param      :: Parameters, 
                         ts_obj     :: TSVectorData,
                         gl_obj     :: GLData,
                         res_obj    :: ResultData)
    
    # Initialization of required variables
    all_clabels = []; 	        # Class labels of all subjects
    train_vecs, test_vecs = Vector{Matrix{Float64}}[], Vector{Matrix{Float64}}[];
    train_labels, test_labels = Vector{}[], Vector{}[];

    # Get direction of EEG data files of selected subjects
    selected_files, db_obj.subject_list = selectSubjects(db_obj.dbName;
        n_of_subject = db_obj.n_of_subject, selection_rule = db_obj.selection_rule,
        random_state = db_obj.random_state, turn = db_obj.turn,
        threshold = db_obj.threshold, filepath = db_obj.filepath);

    for (n,temp_file) ∈ enumerate(selected_files)
        param.verbose && println("processing file $n of $(length(selected_files))...")
        db_obj.artefact_rej ? upp=1 : upp=0;
        o=readNY(temp_file;
                 bandpass=db_obj.bandpass, upperLimit=upp);
        TAindex = findfirst(isequal("Target"), o.clabels);
        # Split the data
        train_splits, _ = getSplits(o.mark; 
                n_splits = db_obj.n_splits,
                split_ratio = db_obj.split_ratio,
                return_type = :index,
                random_state = db_obj.random_state);
        train_splits_no, test_splits_no, y_train, y_test = getSplits(o.mark; 
                n_splits = db_obj.n_splits,
                split_ratio = db_obj.split_ratio,
                return_type = :trial_no,
                random_state = db_obj.random_state);
        # ERP Prototype estimation
        db_obj.paradigm == :ERP ? Y = createPrototype(o.X, o.wl, train_splits, TAindex;
                overlapping = param.overlapping,
                weights = param.ERP_weight,
                PCA_dim = param.PCA_dim,
                verbose = param.verbose) : nothing;
        # Covariance Estimation
        train_tsvec, test_tsvec = [], [];
        if db_obj.paradigm == :ERP
            train_covs, test_covs = estimateCov(o.trials, train_splits_no, 
                                    test_splits_no, Y;
                                    estimator = param.cov_estimator,
                                    det_normalization = param.det_normalization);
            train_tsvec, test_tsvec = tangentVectors(train_covs, test_covs, 1:o.ne,
                                      db_obj.paradigm; 
                                      transpose=false);
        elseif db_obj.paradigm == :MI
            train_covs, test_covs = estimateCov(o.trials, train_splits_no, test_splits_no;
                                    estimator = param.cov_estimator,
                                    det_normalization = param.det_normalization);
            train_tsvec, test_tsvec = tangentVectors(train_covs, test_covs, db_obj.paradigm; 
                                        transpose=false);                        
        end
        # append data
        push!(train_vecs, train_tsvec);
        push!(test_vecs, test_tsvec);
        push!(train_labels, y_train);
        push!(test_labels, y_test);
        append!(all_clabels,[o.clabels]);
    end
    train_vecs, test_vecs = swapDims(train_vecs, test_vecs);

    ts_obj.train_vecs = train_vecs;
    ts_obj.test_vecs = test_vecs;
    ts_obj.train_labels = train_labels;
    ts_obj.test_labels = test_labels;
    ts_obj.all_clabels = all_clabels;
    return nothing
end
