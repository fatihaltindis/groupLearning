# Unit utils.jl, part of groupLearning Package for julia
#
# MIT License 
# Copyright (c) - 2023
# Fatih Altindis and Marco Congedo
# Abdullah Gul University, Kayseri
# GIPSA-lab, CNRS, University Grenoble Alpes



function createBootstrap(features             :: Matrix{Float64},
                         labels               :: Vector{};
                         bootsize             :: Int = 10,
                         n_of_boot            :: Int = 10,
                         w                    :: Symbol = :b,
                         random_state         :: Union{Nothing,Integer} = nothing)

    ## w corresponds to weights of bootstraps, if it is :b, weights are uniform
    ## if it is :r, weights are random
    size(features,2) != length(labels) ? throw(ErrorException("Number of feature vectors and 
        labels must be equal!")) : nothing
    
    isnothing(random_state) ? seeds = Int.(rand(MersenneTwister(1), 1:1e12, n_of_boot)) :
        seeds = Int.(rand(MersenneTwister(random_state), 1:1e12, n_of_boot))

    l_class = unique(labels)
    Wall = []
    for c in eachindex(l_class)
        w == :r ? seed_weight = rand(bootsize) : seed_weight = ones(bootsize) ./ sum(ones(bootsize));
        seed_weight ./= sum(seed_weight);
        Wboot = []
        temp = features[:, labels.==l_class[c]]
        map(x-> push!(Wboot, vec(sum(temp[:,rand(MersenneTwister(seeds[x]), 1:size(temp,2), bootsize)] .* seed_weight' ,dims=2))),1:n_of_boot)
        push!(Wall, hcat(Wboot...))
    end
    return hcat(Wall...)
end

function whitenData(bootstraps             :: Vector{Matrix{Float64}};
                    type                   :: Symbol = :smart,
                    white_dim              :: Integer = 16,
                    smart_subspace         :: Integer = 16,
                    verbose                :: Bool = true)
    
    # Initialize required variables
    M = length(bootstraps)                  # Number of domains
    𝐓 = Vector{Matrix{Float64}}(undef, M)   # Empty matrix for pre-whitened data
    𝐒 = Vector{}(undef, M)                  # SVD objects of each domain
    𝐰𝐡 = Vector{Matrix{Float64}}(undef, M)  # Whitening matrices
    
    ## WHITENING BEGINS ##
    verbose && @info("$(type) is used at pre-whitening!!!")
    verbose && @info("Pre-whitening started...")
    if type == :svd
        for m = 1:M
            𝐒[m] = svd(bootstraps[m], alg=LinearAlgebra.QRIteration())
            𝐰𝐡[m] = (𝐒[m].U[:, 1:white_dim]).*(inv.(𝐒[m].S[1:white_dim]))'
            𝐓[m] = 𝐰𝐡[m]' * bootstraps[m]
            verbose && @info("Completed SVD >>> % $(100*(m/M))")
        end
    elseif type == :pca
        for m = 1:M
            𝐒[m] = svd(bootstraps[m], alg=LinearAlgebra.QRIteration())
            𝐓[m] = 𝐒[m].U[1:white_dim,1:white_dim] * diagm(
                𝐒[m].S[1:white_dim]) * 𝐒[m].Vt[1:white_dim,:]
        end
    elseif type == :smart
        temp_T = Vector{Matrix{Any}}(undef,M)
        𝐂 = Vector{Vector{Matrix{Any}}}(undef,M)
        for m = 1:M
            𝐒[m] = svd(bootstraps[m], alg=LinearAlgebra.QRIteration())
            temp_T[m] = (𝐒[m].U[:,1:white_dim] * diagm(
                𝐒[m].S[1:white_dim]))' * bootstraps[m]
            𝐂[m]=[Matrix{}(undef, size(temp_T[1], 1), size(temp_T[1], 1)) for i=1:m]
            verbose && @info("Completed SVD >>> % $(100*(m/M))")    
        end
        verbose && @info("SVD completed...")

        for m = 1:M, i = m+1:M
            𝐂[i][m] = temp_T[i] * temp_T[m]'
        end
        verbose && @info("Cross-covarainces estimated...")

        H = Matrix{}(undef, size(temp_T[1], 1), size(temp_T[1], 1))
        for m = 1:M
            fill!(H,0)
            for j=1:M
                if m>j 
                    H += 𝐂[m][j]
                end 
                if m<j 
                    H += 𝐂[j][m]'
                end 
            end 
            F = svd(H,alg=LinearAlgebra.QRIteration())
            𝐰𝐡[m] = F.U[:,1:smart_subspace]
        end
        verbose && @info("Whitening matrices estimated...")
        for m = 1:M
            𝐓[m] = 𝐰𝐡[m]' * temp_T[m]
        end
        verbose && @info("Whitening completed...")
    else
        𝐓 = bootstraps
    end
    return 𝐓, 𝐒, 𝐰𝐡
end

function normU(𝐔        :: Vector{Matrix{Float64}};
               type    :: Symbol = :unit,
               𝐓       :: Union{Nothing,Vector{Matrix{Float64}}} = nothing)

    𝐔_ = deepcopy(𝐔)
    M = length(𝐔_)

    # Unit normalization
    if type == :unit
        for m = 1:M normalizeCol!(𝐔_[m], 1:size(𝐔_[m],2)) end
    # Whitening preserving normalization
    elseif type == :white
        isnothing(𝐓) ? throw(ErrorException("𝐓 matrix cannot be empty!!!")) : nothing
        for m = 1:M
            temp_U = deepcopy(𝐔_[m])
            bk = [sqrt(quadraticForm(temp_U[:,k], 𝐓[m]*𝐓[m]')) for k = 1:size(temp_U,1)]
            temp_U ./= bk'
            𝐔_[m] = deepcopy(temp_U)
        end
    # No normalization
    else type == :none
        nothing
    end
    return 𝐔_
end

function sortU!(𝐔       :: Vector{Matrix{Float64}},
                𝐓       :: Vector{Matrix{Float64}})
    
    M = size(𝐔,1);
    # Sort 𝐔 matrices 
    𝐔 = Diagonalizations._flipAndPermute!(𝐔, deepcopy(𝐓), M, 1, :d; dims=2);
    return nothing
end

function estimateB(𝐔                       :: Vector{Matrix{Float64}},
                   𝐰𝐡                      :: Vector{Matrix{Float64}};
                   type                    :: Symbol = :smart,
                   white_dim               :: Integer = 16,
                   reverse_selection       :: Bool = false,
                   𝐒                       :: Vector{Any} = [])

    length(𝐔) == length(𝐰𝐡) ? M = length(𝐔) : throw(
        ErrorException("Number of alignment matrices is not 
        equal to number of whitening matrices!!!"))
    (type == :smart) && isempty(𝐒) ? throw(ErrorException("𝐒 can not be empty for 
        smart whitening!!!")) : nothing;
    𝐁 = []
    if type == :svd
        reverse_selection ? 𝐁 = [𝐰𝐡[m] * 𝐔[m] for m=1:M] : 𝐁 = [𝐰𝐡[m] * 𝐔[m] for m=1:M];
    elseif type == :smart
        reverse_selection ? 𝐁 = [(𝐒[m].U[:,1:white_dim] * diagm(𝐒[m].S[1:white_dim])) *
                𝐰𝐡[m] * 𝐔[m] for m=1:M] : 𝐁 = [(𝐒[m].U[:,1:white_dim] * diagm(𝐒[m].S[1:white_dim])) *
                𝐰𝐡[m] * 𝐔[m] for m=1:M];
    elseif type == :pca
        𝐁 = [𝐒[m].U[:,1:white_dim] * diagm(𝐒[m].S[1:white_dim]) * 𝐔[m] for m=1:M]
    elseif type == :wh_test
        𝐁 = [𝐰𝐡[m] for m=1:M]
    else
        𝐁 = [𝐔[m] for m=1:M]
    end
    return 𝐁
end

# Operates on single train and test split
# Note that each split have vector of matrices, where each matrix belongs to
# a different domain. 
function alignFeatures(train_split          :: Vector{Matrix{Float64}},
                       test_split           :: Vector{Matrix{Float64}},
                       𝐁                    :: Vector{Matrix{Float64}};
                       sub_dim              :: Union{Int,Nothing} = nothing)

    isnothing(sub_dim) ? sub_dim = size(𝐁[1],2) : nothing;

    aligned_train = Vector{Matrix{Float64}}(undef,length(𝐁))
    aligned_test = Vector{Matrix{Float64}}(undef,length(𝐁))
    
    aligned_train = [𝐁[d][:,1:sub_dim]' * Z 
        for (d,Z) ∈ enumerate(train_split)];
    aligned_test = [𝐁[d][:,1:sub_dim]' * Z 
        for (d,Z) ∈ enumerate(test_split)];
    
    return aligned_train, aligned_test
end

# Basically does the same thing, but altered to call when leave-one-out scenario is ran.
function alignFeatures(train_split          :: Vector{Matrix{Float64}},
                       test_matrix          :: Matrix{Float64},
                       𝐁                    :: Vector{Matrix{Float64}},
                       test_sub_idx         :: Int;
                       sub_dim              :: Union{Int,Nothing} = nothing,
                       exclude_from_train   :: Bool = false)

    isnothing(sub_dim) ? sub_dim = size(𝐁[1],2) : nothing;

    aligned_train = [𝐁[d][:,1:sub_dim]' * Z for (d,Z) ∈ enumerate(train_split)];
    aligned_test = [𝐁[test_sub_idx][:,1:sub_dim]' * test_matrix];

    exclude_from_train ? deleteat!(aligned_train,test_sub_idx) : nothing;

    return aligned_train, aligned_test
end

function measureNonDiagonality(𝐔 :: AbstractArray,
                               𝐓 :: AbstractArray)  
    M = size(𝐔,1);
    qn = Matrix{Float64}(0I,M,M);
    for i = 1:M
        for j = 1:i-1
            qn[i,j] = nonDiagonality(𝐔[i]' * 𝐓[i] * 𝐓[j]' * 𝐔[j]);
            qn[j,i] = qn[i,j];
        end
    end
    return qn
end