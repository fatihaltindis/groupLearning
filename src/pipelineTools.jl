# Unit pipelineTools.jl, part of groupLearning Package for julia
#
# MIT License 
# version: 10 Sept 2022
# Copyright (c) - 2023
# Fatih Altindis⁺ꜝ and Marco Congedo ꜝ
# ⁺ Abdullah Gul University, Kayseri
# ꜝ GIPSA-lab, CNRS, University Grenoble Alpes

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

function initiateObjects(dbName, filepath;
                         n_of_subject            :: Integer = 0,
                         selection_rule          :: Symbol = :none,
                         threshold               :: Float64 = 0.7,
                         turn                    :: Symbol = :local,
                         random_state            :: Union{Integer, Nothing} = nothing,
                         bandpass                :: Tuple = (1,16),
                         artefact_rej            :: Bool = true,
                         n_splits                :: Integer = 5,
                         split_ratio             :: Union{Integer, Nothing} = nothing,
                         paradigm                :: Symbol = :ERP,
                         ERP_weight              :: Symbol = :a,
                         overlapping             :: Bool = true,
                         PCA_dim                 :: Union{Nothing, Integer} = nothing,
                         cov_estimator           :: Symbol = :lw,
                         det_normalization       :: Bool = false,
                         dt                      :: Bool = false,
                         ch_union                :: Union{Nothing,Vector{String}} = nothing,
                         vec_mean                :: Bool = true,
                         vec_norm                :: Bool = true,
                         whitening               :: Symbol = :smart,
                         white_dim               :: Integer = 16,
                         sub_dim                 :: Vector{<:Integer} = [16],
                         smart_subspace          :: Integer = 16,
                         n_of_boot               :: Integer = 100,
                         bootsize                :: Integer = 25,
                         normalize_U             :: Symbol = :white,
                         initialize_U            :: Symbol = :none,
                         sort_U                  :: Bool = true,
                         classifier              :: Symbol = :LinearSVC,
                         leaveout                :: Bool = false,
                         repetition              :: Union{Nothing,Int} = nothing,
                         timer                   :: Union{Nothing,TimerOutput} = nothing,
                         save_opt                :: Symbol = :v1,
                         verbose                 :: Bool = false)
    
    db_obj = Database(dbName, n_of_subject, selection_rule, threshold,
                      turn, random_state, filepath, bandpass,
                      artefact_rej, n_splits, split_ratio, paradigm, []);

    param_obj = Parameters(ERP_weight, overlapping, PCA_dim, cov_estimator,
                           det_normalization, dt, ch_union, vec_mean, vec_norm,
                           whitening, white_dim, sub_dim, smart_subspace,
                           n_of_boot, bootsize, normalize_U, initialize_U,
                           sort_U, classifier, random_state, leaveout,
                           repetition, timer, save_opt, verbose);
    
    ts_obj = TSVectorData();
    gl_obj = GLData();
    res_obj = ResultData();

    return db_obj, param_obj, ts_obj, gl_obj, res_obj
end

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

function prepareGL(db_obj     :: Database, 
                   param      :: Parameters, 
                   ts_obj     :: TSVectorData,
                   gl_obj     :: GLData,
                   res_obj    :: ResultData)
    M = size(ts_obj.train_vecs[1], 1);
    n_splits = size(ts_obj.train_vecs, 1);

    for sp in 1:n_splits
        𝐖 = Vector{Matrix{Float64}}(undef,M);
        for m in 1:M
            param.verbose && @info("bootstrapping >>> split: $sp/$n_splits and domain: $m/$M")
            𝐖[m] = createBootstrap(ts_obj.train_vecs[sp][m], ts_obj.train_labels[m];
                                    bootsize = param.bootsize, 
                                    n_of_boot = param.n_of_boot,
                                    w = :b,
                                    random_state = param.random_state);
            # make each bootstrapp zero mean if enabled
            param.vec_mean ? 𝐖[m] .-= mean(𝐖[m], dims=2) : nothing;
            # normalize bootstraps if enabled
            param.vec_norm ? 𝐖[m] ./= mean(norm.(eachcol(𝐖[m]))) : nothing;
        end

        𝐓, 𝐒, 𝐰𝐡 = whitenData(𝐖; type = param.whitening, white_dim = param.white_dim,
                               smart_subspace = param.smart_subspace,
                               verbose = param.verbose);

        push!(gl_obj.T, 𝐓)
        push!(gl_obj.S, 𝐒)
        push!(gl_obj.wh, 𝐰𝐡)
    end
    # normalize feature vectors if enabled
    param.vec_mean ? normVecs!(ts_obj.train_vecs, ts_obj.test_vecs) : nothing;
    return nothing
end

function runGL(db_obj     :: Database, 
               param      :: Parameters, 
               ts_obj     :: TSVectorData,
               gl_obj     :: GLData,
               res_obj    :: ResultData)
    
    n_splits = size(ts_obj.train_vecs, 1);
    𝐁 = Vector{Matrix{Float64}}(undef,n_splits);

    for sp in 1:n_splits
        # Initialize U matrices if needed
        U_init = joalInitializer(gl_obj.T[sp];
                                 type = param.initialize_U);
        # Run joint alignment algorithm 
        𝐔, iter, conv, _ = joal(deepcopy(gl_obj.T[sp]), init = U_init,
                                threaded = false, verbose = param.verbose,
                                maxiter = 2500, tol = 1e-8);
        # Normalize 𝐔 matrices
        𝐔_ = normU(𝐔; type = param.normalize_U, 𝐓 = deepcopy(gl_obj.T[sp]));
        # Sort 𝐔_ matrices
        param.sort_U ? sortU!(𝐔_, gl_obj.T[sp]) : nothing;
        # Compute 𝐁 matrices
        𝐁 = estimateB(𝐔_, gl_obj.wh[sp];
                      𝐒 = gl_obj.S[sp],
                      type = param.whitening,
                      white_dim = param.white_dim,
                      reverse_selection = false);
        
        push!(gl_obj.B, 𝐁);
        push!(gl_obj.U, 𝐔_);
        push!(res_obj.iter, iter);
        push!(res_obj.conv, conv);
    end
    return nothing
end

function runLeaveOut(db_obj     :: Database, 
                     param      :: Parameters, 
                     ts_obj     :: TSVectorData,
                     gl_obj     :: GLData,
                     res_obj    :: ResultData)
    
    M = size(gl_obj.T[1], 1);
    n_splits = size(gl_obj.T, 1);
    
    for sp in 1:n_splits
        # Copy the all subjects bootstrapp and whitening matrices
        all_T = deepcopy(gl_obj.T[sp]);
        all_S = deepcopy(gl_obj.S[sp]);
        all_wh = deepcopy(gl_obj.wh[sp]);
        U_fast = Matrix{Float64}[];
        B_fast = Matrix{Float64}[];
        B_rest = [];
        for m in 1:M
            # Assign leaveout subject
            loo_T = deepcopy(all_T[m]);
            loo_S = deepcopy(all_S[m]);
            loo_wh = deepcopy(all_wh[m]);

            # Exclude leaveout subject from rest
            deleteat!(all_T, m);
            isempty(all_S) ? nothing : deleteat!(all_S, m);
            deleteat!(all_wh, m);

            # Run GALIA on the rest of the group
            U_init = joalInitializer(all_T; type = param.initialize_U);

            𝐔, iter, conv, _ = joal(deepcopy(all_T), init = U_init,
                                    threaded = false, verbose = param.verbose,
                                    maxiter = 2500, tol = 1e-8);
            
            # Fast alignment of leaveout subject
            loo_U = fastAlignment(deepcopy(all_T), deepcopy(𝐔), loo_T; threaded = false);

            # Recover originial bootstrapp and whitening matrices
            insert!(all_T, m, loo_T);
            isempty(all_S) ? nothing : insert!(all_S, m, loo_S);
            insert!(all_wh, m, loo_wh);

            # normalize leaveout U matrix
            insert!(𝐔, m, loo_U);
            𝐔_ = normU(deepcopy(𝐔); type = param.normalize_U, 𝐓 = deepcopy(all_T));

            # Sort 𝐔 matrices 
            param.sort_U ? sortU!(𝐔_, all_T) : nothing;

            # Push leaveout U matrix into this splits U_fast vector
            push!(U_fast, 𝐔_[m]);

            # Estimate B matrix of the leaveout subject
            loo_B = estimateB(𝐔_, all_wh; type = param.whitening, white_dim = param.white_dim,
                              reverse_selection = false, 𝐒 = all_S);
            
            # Push leaveout B matrix into this splits B_fast vector
            push!(B_fast, loo_B[m]);
            deleteat!(loo_B, m);
            println(size(loo_B[1]))
            push!(B_rest, loo_B);
        end
        # Once all subjects of the given split are one by one completed push them
        # into B_fast and U_fast vectors of the gl_obj
        push!(gl_obj.U_fast, U_fast);
        push!(gl_obj.B_fast, B_fast);
        push!(gl_obj.B_rest, B_rest);
    end    
end

function trainGL(db_obj     :: Database, 
                 param      :: Parameters, 
                 ts_obj     :: TSVectorData,
                 gl_obj     :: GLData,
                 res_obj    :: ResultData)
    
    n_splits = size(ts_obj.train_vecs,1);
    temp_acc = [];
    for subdim in param.sub_dim
        if isnothing(subdim) 
            temp_acc, temp_std = glTraining(ts_obj.train_vecs, ts_obj.test_vecs, ts_obj.train_labels,
                                            ts_obj.test_labels, gl_obj.B;
                                            classifier = param.classifier,
                                            verbose = param.verbose);
            res_obj.gl_res = vec(temp_acc);
        else
            temp_acc, temp_std = glTraining(ts_obj.train_vecs, ts_obj.test_vecs, ts_obj.train_labels,
                                            ts_obj.test_labels, gl_obj.B;
                                            sub_dim = subdim,
                                            classifier = param.classifier,
                                            verbose = param.verbose);
            res_obj.gl_res = vec(temp_acc);
        end
        if param.save_opt == :v1
            # isnothing(param.repetition) ? saveResults() : 
            #                               saveResults();
        else
            # isnothing(param.repetition) ? saveResults() : 
            #                               saveResults();
        end
    end
    return nothing
end

function trainSW(db_obj     :: Database, 
                 param      :: Parameters, 
                 ts_obj     :: TSVectorData,
                 gl_obj     :: GLData,
                 res_obj    :: ResultData)
    
    res_obj.sw_res = swTraining(ts_obj.train_vecs, ts_obj.test_vecs,
                                ts_obj.train_labels, ts_obj.test_labels,
                                classifier = param.classifier)[1];
    return nothing
end

function trainFA(db_obj     :: Database, 
                 param      :: Parameters, 
                 ts_obj     :: TSVectorData,
                 gl_obj     :: GLData,
                 res_obj    :: ResultData)

    M = size(gl_obj.T[1], 1);
    n_splits = size(gl_obj.T, 1);
    accuracy = Matrix{Float64}(undef, n_splits, M);
    for sp in 1:n_splits
        for m in 1:M
            tr_vecs = alignFeatures(ts_obj.train_vecs[sp][1:end .!=m], gl_obj.B_rest[sp][m]; 
                                    sub_dim = param.sub_dim[1]);
            te_vecs = alignFeatures([ts_obj.test_vecs[sp][m]], [gl_obj.B_fast[sp][m]];
                                    sub_dim = param.sub_dim[1]);

            accuracy[sp,m] = faTraining(tr_vecs, te_vecs, ts_obj.train_labels[1:end .!=m], 
                                        ts_obj.test_labels, m; classifier = param.classifier,
                                        verbose = param.verbose);
        end
    end

    res_obj.fa_res = vec(mean(accuracy, dims=1))
    return nothing
end

function saveResults(db_obj     :: Database, 
                     param      :: Parameters, 
                     ts_obj     :: TSVectorData,
                     gl_obj     :: GLData,
                     res_obj    :: ResultData)


    println("Save function will be added.")
end

function runPipe!(pipeline, obj_list)
    map(x -> x(obj_list...), pipeline)
    return nothing
end

function plotAcc(obj_list)
    p = Plots.plot();
    ksort = [];
    available_res = Dict("SW" => obj_list[5].sw_res,
                         "GL" => obj_list[5].gl_res,
                         "FA" => obj_list[5].fa_res);
    
    isempty(available_res["SW"]) ? nothing : ksort = sortperm(vec(available_res["SW"]), rev=true);

    for r in available_res
        isempty(ksort) ? ksort = sortperm(vec(r[2]), rev=true) : nothing;
        isempty(r[2]) ? nothing : plot!(p, vec(r[2])[ksort], label=r[1]);
    end
    Plots.ylims!(p, (0.4, 1))
    return p
end
