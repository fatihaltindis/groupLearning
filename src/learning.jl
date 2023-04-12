# Unit learning.jl, part of groupLearning Package for julia
#
# MIT License 
# version: 10 Sept 2022
# Copyright (c) - 2023
# Fatih Altindis and Marco Congedo
# Abdullah Gul University, Kayseri
# GIPSA-lab, CNRS, University Grenoble Alpes
using ScikitLearn
@sk_import svm: LinearSVC

function swTraining(train_splits           :: Vector{Vector{Matrix{Float64}}}, 
                    test_splits            :: Vector{Vector{Matrix{Float64}}},
                    train_labels           :: Vector{Vector{}},
                    test_labels            :: Vector{Vector{}};
                    classifier             :: Symbol = :LinearSVC)

    accuracy = Matrix{Float64}(undef, length(train_splits), length(train_splits[1]))
    if classifier == :ENLR     
        for sp in eachindex(train_splits)
            for sub in eachindex(train_splits[sp])
                lasso_tol = 1e-5;
                co = 1;
                while true
                    try
                        @info("Training subject $sub splits $sp")
                        model = fit(ENLR(), Matrix(train_splits[sp][sub]'), train_labels[sub];
                            verbose = false, tol = lasso_tol);
                        accuracy[sp,sub] = predictAcc(test_labels[sub],
                            predict(model, Matrix(test_splits[sp][sub]'), :l; verbose = false))
                        break
                    catch e
                        if e isa BoundsError
                            @warn("LASSO tolerance is decreased for subject $sub !!!")
                            lasso_tol *= 10;
                            continue
                        elseif e isa MethodError
                            if co == 5
                                break
                            end
                            co += 1;
                            continue
                        elseif e isa ArgumentError
                            break
                        else
                            rethrow(e)
                        end
                    end
                end
            end
        end
    elseif classifier == :LinearSVC
        clf = LinearSVC(tol=1e-6, class_weight = "balanced", max_iter=5000);
        for sp in eachindex(train_splits)
            for sub in eachindex(train_splits[sp])
                ScikitLearn.fit!(clf, Matrix(train_splits[sp][sub]'), train_labels[sub]);
                    accuracy[sp,sub] = PosDefManifoldML.predictAcc(test_labels[sub],
                    clf.predict(Matrix(test_splits[sp][sub]')));
            end
        end
    end
    return vec(mean(accuracy, dims=1)), vec(std(accuracy, dims=1))
end

# This function is for group learning of all aligned domains
function glTraining(train_splits           :: Vector{Vector{Matrix{Float64}}}, 
                    test_splits            :: Vector{Vector{Matrix{Float64}}},
                    train_labels           :: Vector{Vector{}},
                    test_labels            :: Vector{Vector{}},
                    𝐁                      :: Vector{Vector{Matrix{Float64}}};
                    sub_dim                :: Union{Int,Nothing} = nothing,
                    classifier             :: Symbol = :LinearSVC,
                    verbose                :: Bool = true)

    accuracy = Matrix{Float64}(undef, length(train_splits), length(train_splits[1]));
    if classifier == :ENLR
        for sp in eachindex(train_splits)
            alg_train, alg_test = alignFeatures(train_splits[sp], test_splits[sp], 𝐁[sp]; sub_dim);
            model = fit(ENLR(), Matrix(hcat(alg_train...)'), vcat(train_labels...);
                verbose = false);
            for sub in eachindex(alg_test)
                accuracy[sp,sub] = predictAcc(test_labels[sub],
                    predict(model, Matrix(alg_test[sub]'), :l; verbose = verbose))
            end
        end

    elseif classifier == :LinearSVC
        for sp in eachindex(train_splits)
            alg_train, alg_test = alignFeatures(train_splits[sp], test_splits[sp], 𝐁[sp]; sub_dim);
            clf = LinearSVC(tol=1e-6, class_weight = "balanced", max_iter=5000, verbose = verbose);
            ScikitLearn.fit!(clf, Matrix(hcat(alg_train...)'), vcat(train_labels...));
            for sub in eachindex(alg_test)
                accuracy[sp,sub] = PosDefManifoldML.predictAcc(test_labels[sub],
                    clf.predict(Matrix(alg_test[sub]')))
            end
        end
    else
        throw(ErrorException("LinearSVC or ENLR classifier must be chosen!!!"))
    end
    return mean(accuracy, dims=1), std(accuracy, dims=1)
end

# This function is for fast alignment of a new subject (Leave-One-Out scenario)
function faTraining(train_splits           :: Vector{Matrix{Float64}}, 
                    test_splits            :: Vector{Matrix{Float64}},
                    train_labels           :: Vector{Vector{}},
                    test_labels            :: Vector{Vector{}},
                    test_sub_idx           :: Int;
                    classifier             :: Symbol = :LinearSVC,
                    verbose                :: Bool = true)

    accuracy = Matrix{Float64}(undef, 1, 1)
    if classifier == :ENLR
        model = fit(ENLR(), Matrix(hcat(train_splits...)'), vcat(train_labels...);
            verbose = false);
        accuracy = predictAcc(test_labels[1],
            predict(model, Matrix(test_splits[1]'), :l; verbose = verbose))

    elseif classifier == :LinearSVC
        train_s = deepcopy(train_labels);
        deleteat!(train_s, test_sub_idx);
        clf = LinearSVC(tol=1e-6, class_weight = "balanced", max_iter=5000, verbose = verbose);
        ScikitLearn.fit!(clf, Matrix(hcat(train_splits...)'), vcat(train_s...));
        accuracy = PosDefManifoldML.predictAcc(test_labels[1], 
                   clf.predict(Matrix(test_splits[1]')))
    else
        throw(ErrorException("LinearSVC or ENLR classifier must be chosen!!!"))
    end

    return accuracy
end
