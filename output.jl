#  GaussianProcessRegressor
GaussianProcessRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.gaussian_process")).GaussianProcessRegressor
mutable struct GaussianProcessRegressor <: MLJBase.Deterministic
    kernel::Any
    alpha::Union{Float64, Any}
    optimizer::Union{String, Any}
    n_restarts_optimizer::Int
    normalize_y::Bool
    copy_X_train::Bool
    random_state::Int
end
function GaussianProcessRegressor(; kernel=nothing, alpha=1.0e-10, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0, normalize_y=false, copy_X_train=true, random_state=nothing)
    model = GaussianProcessRegressor(kernel, alpha, optimizer, n_restarts_optimizer, normalize_y, copy_X_train, random_state)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::GaussianProcessRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = GaussianProcessRegressor_(kernel = model.kernel, alpha = model.alpha, optimizer = model.optimizer, n_restarts_optimizer = model.n_restarts_optimizer, normalize_y = model.normalize_y, copy_X_train = model.copy_X_train, random_state = model.random_state)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::GaussianProcessRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:GaussianProcessRegressor}) = begin
            string("MLJModels.ScikitLearn_.", GaussianProcessRegressor)
        end
    MLJBase.package_name(::Type{<:GaussianProcessRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:GaussianProcessRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:GaussianProcessRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:GaussianProcessRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:GaussianProcessRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:GaussianProcessRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:GaussianProcessRegressor}) = begin
            true
        end
end
#  RandomForestRegressor
RandomForestRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).RandomForestRegressor
mutable struct RandomForestRegressor <: MLJBase.Deterministic
    n_estimators::Int
    criterion::String
    max_depth::Union{Int, Any}
    min_samples_split::Int
    min_samples_leaf::Int
    min_weight_fraction_leaf::Float64
    max_features::Int
    max_leaf_nodes::Union{Int, Any}
    min_impurity_decrease::Float64
    min_impurity_split::Float64
    bootstrap::Bool
    oob_score::Bool
    n_jobs::Union{Int, Any}
    random_state::Int
    verbose::Int
    warm_start::Bool
end
function RandomForestRegressor(; n_estimators="warn", criterion="mse", max_depth=nothing, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=nothing, min_impurity_decrease=0.0, min_impurity_split=nothing, bootstrap=true, oob_score=false, n_jobs=nothing, random_state=nothing, verbose=0, warm_start=false)
    model = RandomForestRegressor(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, min_impurity_split, bootstrap, oob_score, n_jobs, random_state, verbose, warm_start)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::RandomForestRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = RandomForestRegressor_(n_estimators = model.n_estimators, criterion = model.criterion, max_depth = model.max_depth, min_samples_split = model.min_samples_split, min_samples_leaf = model.min_samples_leaf, min_weight_fraction_leaf = model.min_weight_fraction_leaf, max_features = model.max_features, max_leaf_nodes = model.max_leaf_nodes, min_impurity_decrease = model.min_impurity_decrease, min_impurity_split = model.min_impurity_split, bootstrap = model.bootstrap, oob_score = model.oob_score, n_jobs = model.n_jobs, random_state = model.random_state, verbose = model.verbose, warm_start = model.warm_start)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::RandomForestRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:RandomForestRegressor}) = begin
            string("MLJModels.ScikitLearn_.", RandomForestRegressor)
        end
    MLJBase.package_name(::Type{<:RandomForestRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:RandomForestRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:RandomForestRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:RandomForestRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:RandomForestRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:RandomForestRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:RandomForestRegressor}) = begin
            true
        end
end
#  LinearSVR
LinearSVR_ = ((ScikitLearn.Skcore).pyimport("sklearn.svm")).LinearSVR
mutable struct LinearSVR <: MLJBase.Deterministic
    epsilon::Float64
    tol::Float64
    C::Float64
    loss::String
    fit_intercept::Bool
    intercept_scaling::Float64
    dual::Bool
    verbose::Int
    random_state::Int
    max_iter::Int
end
function LinearSVR(; epsilon=0.0, tol=0.0001, C=1.0, loss="epsilon_insensitive", fit_intercept=true, intercept_scaling=1.0, dual=true, verbose=0, random_state=nothing, max_iter=1000)
    model = LinearSVR(epsilon, tol, C, loss, fit_intercept, intercept_scaling, dual, verbose, random_state, max_iter)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::LinearSVR, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = LinearSVR_(epsilon = model.epsilon, tol = model.tol, C = model.C, loss = model.loss, fit_intercept = model.fit_intercept, intercept_scaling = model.intercept_scaling, dual = model.dual, verbose = model.verbose, random_state = model.random_state, max_iter = model.max_iter)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::LinearSVR, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:LinearSVR}) = begin
            string("MLJModels.ScikitLearn_.", LinearSVR)
        end
    MLJBase.package_name(::Type{<:LinearSVR}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:LinearSVR}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:LinearSVR}) = begin
            false
        end
    MLJBase.package_url(::Type{<:LinearSVR}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:LinearSVR}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:LinearSVR}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:LinearSVR}) = begin
            true
        end
end
#  AdaBoostRegressor
AdaBoostRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).AdaBoostRegressor
mutable struct AdaBoostRegressor <: MLJBase.Deterministic
    base_estimator::Any
    n_estimators::Int
    learning_rate::Float64
    loss::Any
    random_state::Int
end
function AdaBoostRegressor(; base_estimator=nothing, n_estimators=50, learning_rate=1.0, loss="linear", random_state=nothing)
    model = AdaBoostRegressor(base_estimator, n_estimators, learning_rate, loss, random_state)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::AdaBoostRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = AdaBoostRegressor_(base_estimator = model.base_estimator, n_estimators = model.n_estimators, learning_rate = model.learning_rate, loss = model.loss, random_state = model.random_state)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::AdaBoostRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:AdaBoostRegressor}) = begin
            string("MLJModels.ScikitLearn_.", AdaBoostRegressor)
        end
    MLJBase.package_name(::Type{<:AdaBoostRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:AdaBoostRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:AdaBoostRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:AdaBoostRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:AdaBoostRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:AdaBoostRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:AdaBoostRegressor}) = begin
            true
        end
end
#  MultiTaskElasticNetCV
MultiTaskElasticNetCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).MultiTaskElasticNetCV
mutable struct MultiTaskElasticNetCV <: MLJBase.Deterministic
    l1_ratio::Union{Float64, Any}
    eps::Float64
    n_alphas::Int
    alphas::Any
    fit_intercept::Bool
    normalize::Bool
    max_iter::Int
    tol::Float64
    cv::Int
    copy_X::Bool
    verbose::Union{Bool, Int}
    n_jobs::Union{Int, Any}
    random_state::Int
    selection::String
end
function MultiTaskElasticNetCV(; l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=nothing, fit_intercept=true, normalize=false, max_iter=1000, tol=0.0001, cv="warn", copy_X=true, verbose=0, n_jobs=nothing, random_state=nothing, selection="cyclic")
    model = MultiTaskElasticNetCV(l1_ratio, eps, n_alphas, alphas, fit_intercept, normalize, max_iter, tol, cv, copy_X, verbose, n_jobs, random_state, selection)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::MultiTaskElasticNetCV, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = MultiTaskElasticNetCV_(l1_ratio = model.l1_ratio, eps = model.eps, n_alphas = model.n_alphas, alphas = model.alphas, fit_intercept = model.fit_intercept, normalize = model.normalize, max_iter = model.max_iter, tol = model.tol, cv = model.cv, copy_X = model.copy_X, verbose = model.verbose, n_jobs = model.n_jobs, random_state = model.random_state, selection = model.selection)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::MultiTaskElasticNetCV, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:MultiTaskElasticNetCV}) = begin
            string("MLJModels.ScikitLearn_.", MultiTaskElasticNetCV)
        end
    MLJBase.package_name(::Type{<:MultiTaskElasticNetCV}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:MultiTaskElasticNetCV}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:MultiTaskElasticNetCV}) = begin
            false
        end
    MLJBase.package_url(::Type{<:MultiTaskElasticNetCV}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:MultiTaskElasticNetCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:MultiTaskElasticNetCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:MultiTaskElasticNetCV}) = begin
            true
        end
end
#  RANSACRegressor
RANSACRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).RANSACRegressor
mutable struct RANSACRegressor <: MLJBase.Deterministic
    base_estimator::Any
    min_samples::Union{Any, Any}
    residual_threshold::Float64
    is_data_valid::Any
    is_model_valid::Any
    max_trials::Int
    max_skips::Int
    stop_n_inliers::Int
    stop_score::Float64
    stop_probability::Any
    loss::String
    random_state::Int
end
function RANSACRegressor(; base_estimator=nothing, min_samples=nothing, residual_threshold=nothing, is_data_valid=nothing, is_model_valid=nothing, max_trials=100, max_skips=nothing, stop_n_inliers=nothing, stop_score=nothing, stop_probability=0.99, loss="absolute_loss", random_state=nothing)
    model = RANSACRegressor(base_estimator, min_samples, residual_threshold, is_data_valid, is_model_valid, max_trials, max_skips, stop_n_inliers, stop_score, stop_probability, loss, random_state)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::RANSACRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = RANSACRegressor_(base_estimator = model.base_estimator, min_samples = model.min_samples, residual_threshold = model.residual_threshold, is_data_valid = model.is_data_valid, is_model_valid = model.is_model_valid, max_trials = model.max_trials, max_skips = model.max_skips, stop_n_inliers = model.stop_n_inliers, stop_score = model.stop_score, stop_probability = model.stop_probability, loss = model.loss, random_state = model.random_state)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::RANSACRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:RANSACRegressor}) = begin
            string("MLJModels.ScikitLearn_.", RANSACRegressor)
        end
    MLJBase.package_name(::Type{<:RANSACRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:RANSACRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:RANSACRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:RANSACRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:RANSACRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:RANSACRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:RANSACRegressor}) = begin
            true
        end
end
#  LassoLars
LassoLars_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).LassoLars
mutable struct LassoLars <: MLJBase.Deterministic
    alpha::Float64
    fit_intercept::Bool
    verbose::Union{Bool, Int}
    normalize::Bool
    precompute::Any
    max_iter::Int
    eps::Float64
    copy_X::Bool
    fit_path::Bool
    positive::Any
end
function LassoLars(; alpha=1.0, fit_intercept=true, verbose=false, normalize=true, precompute="auto", max_iter=500, eps=2.220446049250313e-16, copy_X=true, fit_path=true, positive=false)
    model = LassoLars(alpha, fit_intercept, verbose, normalize, precompute, max_iter, eps, copy_X, fit_path, positive)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::LassoLars, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = LassoLars_(alpha = model.alpha, fit_intercept = model.fit_intercept, verbose = model.verbose, normalize = model.normalize, precompute = model.precompute, max_iter = model.max_iter, eps = model.eps, copy_X = model.copy_X, fit_path = model.fit_path, positive = model.positive)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::LassoLars, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:LassoLars}) = begin
            string("MLJModels.ScikitLearn_.", LassoLars)
        end
    MLJBase.package_name(::Type{<:LassoLars}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:LassoLars}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:LassoLars}) = begin
            false
        end
    MLJBase.package_url(::Type{<:LassoLars}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:LassoLars}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:LassoLars}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:LassoLars}) = begin
            true
        end
end
#  ElasticNetCV
ElasticNetCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).ElasticNetCV
mutable struct ElasticNetCV <: MLJBase.Deterministic
    l1_ratio::Union{Float64, Any}
    eps::Float64
    n_alphas::Int
    alphas::Any
    fit_intercept::Bool
    normalize::Bool
    precompute::Any
    max_iter::Int
    tol::Float64
    cv::Int
    copy_X::Bool
    verbose::Union{Bool, Int}
    n_jobs::Union{Int, Any}
    positive::Bool
    random_state::Int
    selection::String
end
function ElasticNetCV(; l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=nothing, fit_intercept=true, normalize=false, precompute="auto", max_iter=1000, tol=0.0001, cv="warn", copy_X=true, verbose=0, n_jobs=nothing, positive=false, random_state=nothing, selection="cyclic")
    model = ElasticNetCV(l1_ratio, eps, n_alphas, alphas, fit_intercept, normalize, precompute, max_iter, tol, cv, copy_X, verbose, n_jobs, positive, random_state, selection)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::ElasticNetCV, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = ElasticNetCV_(l1_ratio = model.l1_ratio, eps = model.eps, n_alphas = model.n_alphas, alphas = model.alphas, fit_intercept = model.fit_intercept, normalize = model.normalize, precompute = model.precompute, max_iter = model.max_iter, tol = model.tol, cv = model.cv, copy_X = model.copy_X, verbose = model.verbose, n_jobs = model.n_jobs, positive = model.positive, random_state = model.random_state, selection = model.selection)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::ElasticNetCV, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:ElasticNetCV}) = begin
            string("MLJModels.ScikitLearn_.", ElasticNetCV)
        end
    MLJBase.package_name(::Type{<:ElasticNetCV}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:ElasticNetCV}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:ElasticNetCV}) = begin
            false
        end
    MLJBase.package_url(::Type{<:ElasticNetCV}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:ElasticNetCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:ElasticNetCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:ElasticNetCV}) = begin
            true
        end
end
#  MultiOutputRegressor
MultiOutputRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.multioutput")).MultiOutputRegressor
mutable struct MultiOutputRegressor <: MLJBase.Deterministic
    n_jobs::Union{Int, Any}
end
function MultiOutputRegressor(; n_jobs=nothing)
    model = MultiOutputRegressor(n_jobs)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::MultiOutputRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = MultiOutputRegressor_(n_jobs = model.n_jobs)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::MultiOutputRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:MultiOutputRegressor}) = begin
            string("MLJModels.ScikitLearn_.", MultiOutputRegressor)
        end
    MLJBase.package_name(::Type{<:MultiOutputRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:MultiOutputRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:MultiOutputRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:MultiOutputRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:MultiOutputRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:MultiOutputRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:MultiOutputRegressor}) = begin
            true
        end
end
#  RidgeClassifierCV
RidgeClassifierCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).RidgeClassifierCV
mutable struct RidgeClassifierCV <: MLJBase.Deterministic
    alphas::Any
    fit_intercept::Bool
    normalize::Bool
    scoring::String
    cv::Int
    class_weight::Union{Any, Any}
    store_cv_values::Bool
end
function RidgeClassifierCV(; alphas=nothing, fit_intercept=true, normalize=false, scoring=nothing, cv=nothing, class_weight=nothing, store_cv_values=false)
    model = RidgeClassifierCV(alphas, fit_intercept, normalize, scoring, cv, class_weight, store_cv_values)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::RidgeClassifierCV, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = RidgeClassifierCV_(alphas = model.alphas, fit_intercept = model.fit_intercept, normalize = model.normalize, scoring = model.scoring, cv = model.cv, class_weight = model.class_weight, store_cv_values = model.store_cv_values)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::RidgeClassifierCV, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:RidgeClassifierCV}) = begin
            string("MLJModels.ScikitLearn_.", RidgeClassifierCV)
        end
    MLJBase.package_name(::Type{<:RidgeClassifierCV}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:RidgeClassifierCV}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:RidgeClassifierCV}) = begin
            false
        end
    MLJBase.package_url(::Type{<:RidgeClassifierCV}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:RidgeClassifierCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:RidgeClassifierCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:RidgeClassifierCV}) = begin
            true
        end
end
#  SVR
SVR_ = ((ScikitLearn.Skcore).pyimport("sklearn.svm")).SVR
mutable struct SVR <: MLJBase.Deterministic
    kernel::String
    degree::Int
    gamma::Float64
    coef0::Float64
    tol::Float64
    C::Float64
    epsilon::Float64
    shrinking::Bool
    cache_size::Float64
    verbose::Bool
    max_iter::Int
end
function SVR(; kernel="rbf", degree=3, gamma="auto_deprecated", coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=true, cache_size=200, verbose=false, max_iter=-1)
    model = SVR(kernel, degree, gamma, coef0, tol, C, epsilon, shrinking, cache_size, verbose, max_iter)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::SVR, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = SVR_(kernel = model.kernel, degree = model.degree, gamma = model.gamma, coef0 = model.coef0, tol = model.tol, C = model.C, epsilon = model.epsilon, shrinking = model.shrinking, cache_size = model.cache_size, verbose = model.verbose, max_iter = model.max_iter)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::SVR, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:SVR}) = begin
            string("MLJModels.ScikitLearn_.", SVR)
        end
    MLJBase.package_name(::Type{<:SVR}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:SVR}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:SVR}) = begin
            false
        end
    MLJBase.package_url(::Type{<:SVR}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:SVR}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:SVR}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:SVR}) = begin
            true
        end
end
#  LassoCV
LassoCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).LassoCV
mutable struct LassoCV <: MLJBase.Deterministic
    eps::Float64
    n_alphas::Int
    alphas::Any
    fit_intercept::Bool
    normalize::Bool
    precompute::Any
    max_iter::Int
    tol::Float64
    copy_X::Bool
    cv::Int
    verbose::Union{Bool, Int}
    n_jobs::Union{Int, Any}
    positive::Bool
    random_state::Int
    selection::String
end
function LassoCV(; eps=0.001, n_alphas=100, alphas=nothing, fit_intercept=true, normalize=false, precompute="auto", max_iter=1000, tol=0.0001, copy_X=true, cv="warn", verbose=false, n_jobs=nothing, positive=false, random_state=nothing, selection="cyclic")
    model = LassoCV(eps, n_alphas, alphas, fit_intercept, normalize, precompute, max_iter, tol, copy_X, cv, verbose, n_jobs, positive, random_state, selection)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::LassoCV, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = LassoCV_(eps = model.eps, n_alphas = model.n_alphas, alphas = model.alphas, fit_intercept = model.fit_intercept, normalize = model.normalize, precompute = model.precompute, max_iter = model.max_iter, tol = model.tol, copy_X = model.copy_X, cv = model.cv, verbose = model.verbose, n_jobs = model.n_jobs, positive = model.positive, random_state = model.random_state, selection = model.selection)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::LassoCV, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:LassoCV}) = begin
            string("MLJModels.ScikitLearn_.", LassoCV)
        end
    MLJBase.package_name(::Type{<:LassoCV}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:LassoCV}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:LassoCV}) = begin
            false
        end
    MLJBase.package_url(::Type{<:LassoCV}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:LassoCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:LassoCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:LassoCV}) = begin
            true
        end
end
#  BayesianRidge
BayesianRidge_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).BayesianRidge
mutable struct BayesianRidge <: MLJBase.Deterministic
    n_iter::Int
    tol::Float64
    alpha_1::Float64
    alpha_2::Float64
    lambda_1::Float64
    lambda_2::Float64
    compute_score::Bool
    fit_intercept::Bool
    normalize::Bool
    copy_X::Bool
    verbose::Bool
end
function BayesianRidge(; n_iter=300, tol=0.001, alpha_1=1.0e-6, alpha_2=1.0e-6, lambda_1=1.0e-6, lambda_2=1.0e-6, compute_score=false, fit_intercept=true, normalize=false, copy_X=true, verbose=false)
    model = BayesianRidge(n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, compute_score, fit_intercept, normalize, copy_X, verbose)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::BayesianRidge, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = BayesianRidge_(n_iter = model.n_iter, tol = model.tol, alpha_1 = model.alpha_1, alpha_2 = model.alpha_2, lambda_1 = model.lambda_1, lambda_2 = model.lambda_2, compute_score = model.compute_score, fit_intercept = model.fit_intercept, normalize = model.normalize, copy_X = model.copy_X, verbose = model.verbose)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::BayesianRidge, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:BayesianRidge}) = begin
            string("MLJModels.ScikitLearn_.", BayesianRidge)
        end
    MLJBase.package_name(::Type{<:BayesianRidge}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:BayesianRidge}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:BayesianRidge}) = begin
            false
        end
    MLJBase.package_url(::Type{<:BayesianRidge}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:BayesianRidge}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:BayesianRidge}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:BayesianRidge}) = begin
            true
        end
end
#  ElasticNet
ElasticNet_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).ElasticNet
mutable struct ElasticNet <: MLJBase.Deterministic
    alpha::Float64
    l1_ratio::Float64
    fit_intercept::Bool
    normalize::Bool
    precompute::Any
    max_iter::Int
    copy_X::Bool
    tol::Float64
    warm_start::Bool
    positive::Bool
    random_state::Int
    selection::String
end
function ElasticNet(; alpha=1.0, l1_ratio=0.5, fit_intercept=true, normalize=false, precompute=false, max_iter=1000, copy_X=true, tol=0.0001, warm_start=false, positive=false, random_state=nothing, selection="cyclic")
    model = ElasticNet(alpha, l1_ratio, fit_intercept, normalize, precompute, max_iter, copy_X, tol, warm_start, positive, random_state, selection)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::ElasticNet, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = ElasticNet_(alpha = model.alpha, l1_ratio = model.l1_ratio, fit_intercept = model.fit_intercept, normalize = model.normalize, precompute = model.precompute, max_iter = model.max_iter, copy_X = model.copy_X, tol = model.tol, warm_start = model.warm_start, positive = model.positive, random_state = model.random_state, selection = model.selection)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::ElasticNet, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:ElasticNet}) = begin
            string("MLJModels.ScikitLearn_.", ElasticNet)
        end
    MLJBase.package_name(::Type{<:ElasticNet}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:ElasticNet}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:ElasticNet}) = begin
            false
        end
    MLJBase.package_url(::Type{<:ElasticNet}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:ElasticNet}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:ElasticNet}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:ElasticNet}) = begin
            true
        end
end
#  OrthogonalMatchingPursuitCV
OrthogonalMatchingPursuitCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).OrthogonalMatchingPursuitCV
mutable struct OrthogonalMatchingPursuitCV <: MLJBase.Deterministic
    copy::Bool
    fit_intercept::Bool
    normalize::Bool
    max_iter::Int
    cv::Int
    n_jobs::Union{Int, Any}
    verbose::Union{Bool, Int}
end
function OrthogonalMatchingPursuitCV(; copy=true, fit_intercept=true, normalize=true, max_iter=nothing, cv="warn", n_jobs=nothing, verbose=false)
    model = OrthogonalMatchingPursuitCV(copy, fit_intercept, normalize, max_iter, cv, n_jobs, verbose)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::OrthogonalMatchingPursuitCV, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = OrthogonalMatchingPursuitCV_(copy = model.copy, fit_intercept = model.fit_intercept, normalize = model.normalize, max_iter = model.max_iter, cv = model.cv, n_jobs = model.n_jobs, verbose = model.verbose)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::OrthogonalMatchingPursuitCV, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:OrthogonalMatchingPursuitCV}) = begin
            string("MLJModels.ScikitLearn_.", OrthogonalMatchingPursuitCV)
        end
    MLJBase.package_name(::Type{<:OrthogonalMatchingPursuitCV}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:OrthogonalMatchingPursuitCV}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:OrthogonalMatchingPursuitCV}) = begin
            false
        end
    MLJBase.package_url(::Type{<:OrthogonalMatchingPursuitCV}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:OrthogonalMatchingPursuitCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:OrthogonalMatchingPursuitCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:OrthogonalMatchingPursuitCV}) = begin
            true
        end
end
#  Ridge
Ridge_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).Ridge
mutable struct Ridge <: MLJBase.Deterministic
    alpha::Any
    fit_intercept::Bool
    normalize::Bool
    copy_X::Bool
    max_iter::Int
    tol::Float64
    solver::Any
    random_state::Int
end
function Ridge(; alpha=1.0, fit_intercept=true, normalize=false, copy_X=true, max_iter=nothing, tol=0.001, solver="auto", random_state=nothing)
    model = Ridge(alpha, fit_intercept, normalize, copy_X, max_iter, tol, solver, random_state)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::Ridge, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = Ridge_(alpha = model.alpha, fit_intercept = model.fit_intercept, normalize = model.normalize, copy_X = model.copy_X, max_iter = model.max_iter, tol = model.tol, solver = model.solver, random_state = model.random_state)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::Ridge, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:Ridge}) = begin
            string("MLJModels.ScikitLearn_.", Ridge)
        end
    MLJBase.package_name(::Type{<:Ridge}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:Ridge}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:Ridge}) = begin
            false
        end
    MLJBase.package_url(::Type{<:Ridge}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:Ridge}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:Ridge}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:Ridge}) = begin
            true
        end
end
#  GraphicalLassoCV
GraphicalLassoCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.covariance")).GraphicalLassoCV
mutable struct GraphicalLassoCV <: MLJBase.Deterministic
    alphas::Int
    n_refinements::Any
    cv::Int
    tol::Any
    enet_tol::Any
    max_iter::Int
    mode::Any
    n_jobs::Union{Int, Any}
    verbose::Bool
    assume_centered::Bool
end
function GraphicalLassoCV(; alphas=4, n_refinements=4, cv="warn", tol=0.0001, enet_tol=0.0001, max_iter=100, mode="cd", n_jobs=nothing, verbose=false, assume_centered=false)
    model = GraphicalLassoCV(alphas, n_refinements, cv, tol, enet_tol, max_iter, mode, n_jobs, verbose, assume_centered)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::GraphicalLassoCV, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = GraphicalLassoCV_(alphas = model.alphas, n_refinements = model.n_refinements, cv = model.cv, tol = model.tol, enet_tol = model.enet_tol, max_iter = model.max_iter, mode = model.mode, n_jobs = model.n_jobs, verbose = model.verbose, assume_centered = model.assume_centered)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::GraphicalLassoCV, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:GraphicalLassoCV}) = begin
            string("MLJModels.ScikitLearn_.", GraphicalLassoCV)
        end
    MLJBase.package_name(::Type{<:GraphicalLassoCV}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:GraphicalLassoCV}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:GraphicalLassoCV}) = begin
            false
        end
    MLJBase.package_url(::Type{<:GraphicalLassoCV}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:GraphicalLassoCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:GraphicalLassoCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:GraphicalLassoCV}) = begin
            true
        end
end
#  MultiTaskElasticNet
MultiTaskElasticNet_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).MultiTaskElasticNet
mutable struct MultiTaskElasticNet <: MLJBase.Deterministic
    alpha::Float64
    l1_ratio::Float64
    fit_intercept::Bool
    normalize::Bool
    copy_X::Bool
    max_iter::Int
    tol::Float64
    warm_start::Bool
    random_state::Int
    selection::String
end
function MultiTaskElasticNet(; alpha=1.0, l1_ratio=0.5, fit_intercept=true, normalize=false, copy_X=true, max_iter=1000, tol=0.0001, warm_start=false, random_state=nothing, selection="cyclic")
    model = MultiTaskElasticNet(alpha, l1_ratio, fit_intercept, normalize, copy_X, max_iter, tol, warm_start, random_state, selection)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::MultiTaskElasticNet, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = MultiTaskElasticNet_(alpha = model.alpha, l1_ratio = model.l1_ratio, fit_intercept = model.fit_intercept, normalize = model.normalize, copy_X = model.copy_X, max_iter = model.max_iter, tol = model.tol, warm_start = model.warm_start, random_state = model.random_state, selection = model.selection)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::MultiTaskElasticNet, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:MultiTaskElasticNet}) = begin
            string("MLJModels.ScikitLearn_.", MultiTaskElasticNet)
        end
    MLJBase.package_name(::Type{<:MultiTaskElasticNet}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:MultiTaskElasticNet}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:MultiTaskElasticNet}) = begin
            false
        end
    MLJBase.package_url(::Type{<:MultiTaskElasticNet}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:MultiTaskElasticNet}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:MultiTaskElasticNet}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:MultiTaskElasticNet}) = begin
            true
        end
end
#  LassoLarsIC
LassoLarsIC_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).LassoLarsIC
mutable struct LassoLarsIC <: MLJBase.Deterministic
    criterion::Any
    fit_intercept::Bool
    verbose::Union{Bool, Int}
    normalize::Bool
    precompute::Any
    max_iter::Int
    eps::Float64
    copy_X::Bool
    positive::Any
end
function LassoLarsIC(; criterion="aic", fit_intercept=true, verbose=false, normalize=true, precompute="auto", max_iter=500, eps=2.220446049250313e-16, copy_X=true, positive=false)
    model = LassoLarsIC(criterion, fit_intercept, verbose, normalize, precompute, max_iter, eps, copy_X, positive)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::LassoLarsIC, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = LassoLarsIC_(criterion = model.criterion, fit_intercept = model.fit_intercept, verbose = model.verbose, normalize = model.normalize, precompute = model.precompute, max_iter = model.max_iter, eps = model.eps, copy_X = model.copy_X, positive = model.positive)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::LassoLarsIC, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:LassoLarsIC}) = begin
            string("MLJModels.ScikitLearn_.", LassoLarsIC)
        end
    MLJBase.package_name(::Type{<:LassoLarsIC}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:LassoLarsIC}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:LassoLarsIC}) = begin
            false
        end
    MLJBase.package_url(::Type{<:LassoLarsIC}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:LassoLarsIC}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:LassoLarsIC}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:LassoLarsIC}) = begin
            true
        end
end
#  TheilSenRegressor
TheilSenRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).TheilSenRegressor
mutable struct TheilSenRegressor <: MLJBase.Deterministic
    fit_intercept::Bool
    copy_X::Bool
    max_subpopulation::Int
    n_subsamples::Int
    max_iter::Int
    tol::Float64
    random_state::Int
    n_jobs::Union{Int, Any}
    verbose::Bool
end
function TheilSenRegressor(; fit_intercept=true, copy_X=true, max_subpopulation=10000.0, n_subsamples=nothing, max_iter=300, tol=0.001, random_state=nothing, n_jobs=nothing, verbose=false)
    model = TheilSenRegressor(fit_intercept, copy_X, max_subpopulation, n_subsamples, max_iter, tol, random_state, n_jobs, verbose)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::TheilSenRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = TheilSenRegressor_(fit_intercept = model.fit_intercept, copy_X = model.copy_X, max_subpopulation = model.max_subpopulation, n_subsamples = model.n_subsamples, max_iter = model.max_iter, tol = model.tol, random_state = model.random_state, n_jobs = model.n_jobs, verbose = model.verbose)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::TheilSenRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:TheilSenRegressor}) = begin
            string("MLJModels.ScikitLearn_.", TheilSenRegressor)
        end
    MLJBase.package_name(::Type{<:TheilSenRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:TheilSenRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:TheilSenRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:TheilSenRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:TheilSenRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:TheilSenRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:TheilSenRegressor}) = begin
            true
        end
end
#  BaggingRegressor
BaggingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).BaggingRegressor
mutable struct BaggingRegressor <: MLJBase.Deterministic
    base_estimator::Union{Any, Any}
    n_estimators::Int
    max_samples::Union{Int, Float64}
    max_features::Union{Int, Float64}
    bootstrap::Bool
    bootstrap_features::Bool
    oob_score::Bool
    warm_start::Bool
    n_jobs::Union{Int, Any}
    random_state::Int
    verbose::Int
end
function BaggingRegressor(; base_estimator=nothing, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=true, bootstrap_features=false, oob_score=false, warm_start=false, n_jobs=nothing, random_state=nothing, verbose=0)
    model = BaggingRegressor(base_estimator, n_estimators, max_samples, max_features, bootstrap, bootstrap_features, oob_score, warm_start, n_jobs, random_state, verbose)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::BaggingRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = BaggingRegressor_(base_estimator = model.base_estimator, n_estimators = model.n_estimators, max_samples = model.max_samples, max_features = model.max_features, bootstrap = model.bootstrap, bootstrap_features = model.bootstrap_features, oob_score = model.oob_score, warm_start = model.warm_start, n_jobs = model.n_jobs, random_state = model.random_state, verbose = model.verbose)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::BaggingRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:BaggingRegressor}) = begin
            string("MLJModels.ScikitLearn_.", BaggingRegressor)
        end
    MLJBase.package_name(::Type{<:BaggingRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:BaggingRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:BaggingRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:BaggingRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:BaggingRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:BaggingRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:BaggingRegressor}) = begin
            true
        end
end
#  DummyRegressor
DummyRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.dummy")).DummyRegressor
mutable struct DummyRegressor <: MLJBase.Deterministic
    strategy::String
    constant::Union{Int, Float64, Any}
    quantile::Any
end
function DummyRegressor(; strategy="mean", constant=nothing, quantile=nothing)
    model = DummyRegressor(strategy, constant, quantile)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::DummyRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = DummyRegressor_(strategy = model.strategy, constant = model.constant, quantile = model.quantile)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::DummyRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:DummyRegressor}) = begin
            string("MLJModels.ScikitLearn_.", DummyRegressor)
        end
    MLJBase.package_name(::Type{<:DummyRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:DummyRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:DummyRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:DummyRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:DummyRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:DummyRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:DummyRegressor}) = begin
            true
        end
end
#  ARDRegression
ARDRegression_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).ARDRegression
mutable struct ARDRegression <: MLJBase.Deterministic
    n_iter::Int
    tol::Float64
    alpha_1::Float64
    alpha_2::Float64
    lambda_1::Float64
    lambda_2::Float64
    compute_score::Bool
    threshold_lambda::Float64
    fit_intercept::Bool
    normalize::Bool
    copy_X::Bool
    verbose::Bool
end
function ARDRegression(; n_iter=300, tol=0.001, alpha_1=1.0e-6, alpha_2=1.0e-6, lambda_1=1.0e-6, lambda_2=1.0e-6, compute_score=false, threshold_lambda=10000.0, fit_intercept=true, normalize=false, copy_X=true, verbose=false)
    model = ARDRegression(n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, compute_score, threshold_lambda, fit_intercept, normalize, copy_X, verbose)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::ARDRegression, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = ARDRegression_(n_iter = model.n_iter, tol = model.tol, alpha_1 = model.alpha_1, alpha_2 = model.alpha_2, lambda_1 = model.lambda_1, lambda_2 = model.lambda_2, compute_score = model.compute_score, threshold_lambda = model.threshold_lambda, fit_intercept = model.fit_intercept, normalize = model.normalize, copy_X = model.copy_X, verbose = model.verbose)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::ARDRegression, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:ARDRegression}) = begin
            string("MLJModels.ScikitLearn_.", ARDRegression)
        end
    MLJBase.package_name(::Type{<:ARDRegression}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:ARDRegression}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:ARDRegression}) = begin
            false
        end
    MLJBase.package_url(::Type{<:ARDRegression}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:ARDRegression}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:ARDRegression}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:ARDRegression}) = begin
            true
        end
end
#  RegressorChain
RegressorChain_ = ((ScikitLearn.Skcore).pyimport("sklearn.multioutput")).RegressorChain
mutable struct RegressorChain <: MLJBase.Deterministic
    order::Any
    cv::Int
    random_state::Int
end
function RegressorChain(; order=nothing, cv=nothing, random_state=nothing)
    model = RegressorChain(order, cv, random_state)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::RegressorChain, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = RegressorChain_(order = model.order, cv = model.cv, random_state = model.random_state)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::RegressorChain, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:RegressorChain}) = begin
            string("MLJModels.ScikitLearn_.", RegressorChain)
        end
    MLJBase.package_name(::Type{<:RegressorChain}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:RegressorChain}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:RegressorChain}) = begin
            false
        end
    MLJBase.package_url(::Type{<:RegressorChain}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:RegressorChain}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:RegressorChain}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:RegressorChain}) = begin
            true
        end
end
#  TransformedTargetRegressor
TransformedTargetRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.compose")).TransformedTargetRegressor
mutable struct TransformedTargetRegressor <: MLJBase.Deterministic
    regressor::Any
    transformer::Any
    func::Function
    inverse_func::Function
    check_inverse::Bool
end
function TransformedTargetRegressor(; regressor=nothing, transformer=nothing, func=nothing, inverse_func=nothing, check_inverse=true)
    model = TransformedTargetRegressor(regressor, transformer, func, inverse_func, check_inverse)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::TransformedTargetRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = TransformedTargetRegressor_(regressor = model.regressor, transformer = model.transformer, func = model.func, inverse_func = model.inverse_func, check_inverse = model.check_inverse)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::TransformedTargetRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:TransformedTargetRegressor}) = begin
            string("MLJModels.ScikitLearn_.", TransformedTargetRegressor)
        end
    MLJBase.package_name(::Type{<:TransformedTargetRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:TransformedTargetRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:TransformedTargetRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:TransformedTargetRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:TransformedTargetRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:TransformedTargetRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:TransformedTargetRegressor}) = begin
            true
        end
end
#  HistGradientBoostingRegressor
HistGradientBoostingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).HistGradientBoostingRegressor
mutable struct HistGradientBoostingRegressor <: MLJBase.Deterministic
    loss::Any
    learning_rate::Float64
    max_iter::Int
    max_leaf_nodes::Union{Int, Any}
    max_depth::Union{Int, Any}
    min_samples_leaf::Int
    l2_regularization::Float64
    max_bins::Int
    scoring::Union{String, Any, Any}
    validation_fraction::Union{Int, Float64, Any}
    n_iter_no_change::Union{Int, Any}
    tol::Union{Float64, Any}
    random_state::Int
end
function HistGradientBoostingRegressor(; loss="least_squares", learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=nothing, min_samples_leaf=20, l2_regularization=0.0, max_bins=256, scoring=nothing, validation_fraction=0.1, n_iter_no_change=nothing, tol=1.0e-7, random_state=nothing)
    model = HistGradientBoostingRegressor(loss, learning_rate, max_iter, max_leaf_nodes, max_depth, min_samples_leaf, l2_regularization, max_bins, scoring, validation_fraction, n_iter_no_change, tol, random_state)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::HistGradientBoostingRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = HistGradientBoostingRegressor_(loss = model.loss, learning_rate = model.learning_rate, max_iter = model.max_iter, max_leaf_nodes = model.max_leaf_nodes, max_depth = model.max_depth, min_samples_leaf = model.min_samples_leaf, l2_regularization = model.l2_regularization, max_bins = model.max_bins, scoring = model.scoring, validation_fraction = model.validation_fraction, n_iter_no_change = model.n_iter_no_change, tol = model.tol, random_state = model.random_state)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::HistGradientBoostingRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:HistGradientBoostingRegressor}) = begin
            string("MLJModels.ScikitLearn_.", HistGradientBoostingRegressor)
        end
    MLJBase.package_name(::Type{<:HistGradientBoostingRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:HistGradientBoostingRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:HistGradientBoostingRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:HistGradientBoostingRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:HistGradientBoostingRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:HistGradientBoostingRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:HistGradientBoostingRegressor}) = begin
            true
        end
end
#  GradientBoostingRegressor
GradientBoostingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).GradientBoostingRegressor
mutable struct GradientBoostingRegressor <: MLJBase.Deterministic
    loss::Any
    learning_rate::Float64
    n_estimators::Any
    subsample::Float64
    criterion::String
    min_samples_split::Int
    min_samples_leaf::Int
    min_weight_fraction_leaf::Float64
    max_depth::Int
    min_impurity_decrease::Float64
    min_impurity_split::Float64
    init::Union{Any, Any}
    random_state::Int
    max_features::Int
    alpha::Any
    verbose::Int
    max_leaf_nodes::Union{Int, Any}
    warm_start::Bool
    presort::Union{Bool, Any}
    validation_fraction::Float64
    n_iter_no_change::Int
    tol::Float64
end
function GradientBoostingRegressor(; loss="ls", learning_rate=0.1, n_estimators=100, subsample=1.0, criterion="friedman_mse", min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=nothing, init=nothing, random_state=nothing, max_features=nothing, alpha=0.9, verbose=0, max_leaf_nodes=nothing, warm_start=false, presort="auto", validation_fraction=0.1, n_iter_no_change=nothing, tol=0.0001)
    model = GradientBoostingRegressor(loss, learning_rate, n_estimators, subsample, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_depth, min_impurity_decrease, min_impurity_split, init, random_state, max_features, alpha, verbose, max_leaf_nodes, warm_start, presort, validation_fraction, n_iter_no_change, tol)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::GradientBoostingRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = GradientBoostingRegressor_(loss = model.loss, learning_rate = model.learning_rate, n_estimators = model.n_estimators, subsample = model.subsample, criterion = model.criterion, min_samples_split = model.min_samples_split, min_samples_leaf = model.min_samples_leaf, min_weight_fraction_leaf = model.min_weight_fraction_leaf, max_depth = model.max_depth, min_impurity_decrease = model.min_impurity_decrease, min_impurity_split = model.min_impurity_split, init = model.init, random_state = model.random_state, max_features = model.max_features, alpha = model.alpha, verbose = model.verbose, max_leaf_nodes = model.max_leaf_nodes, warm_start = model.warm_start, presort = model.presort, validation_fraction = model.validation_fraction, n_iter_no_change = model.n_iter_no_change, tol = model.tol)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::GradientBoostingRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:GradientBoostingRegressor}) = begin
            string("MLJModels.ScikitLearn_.", GradientBoostingRegressor)
        end
    MLJBase.package_name(::Type{<:GradientBoostingRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:GradientBoostingRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:GradientBoostingRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:GradientBoostingRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:GradientBoostingRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:GradientBoostingRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:GradientBoostingRegressor}) = begin
            true
        end
end
#  Lars
Lars_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).Lars
mutable struct Lars <: MLJBase.Deterministic
    fit_intercept::Bool
    verbose::Union{Bool, Int}
    normalize::Bool
    precompute::Any
    n_nonzero_coefs::Int
    eps::Float64
    copy_X::Bool
    fit_path::Bool
    positive::Any
end
function Lars(; fit_intercept=true, verbose=false, normalize=true, precompute="auto", n_nonzero_coefs=500, eps=2.220446049250313e-16, copy_X=true, fit_path=true, positive=false)
    model = Lars(fit_intercept, verbose, normalize, precompute, n_nonzero_coefs, eps, copy_X, fit_path, positive)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::Lars, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = Lars_(fit_intercept = model.fit_intercept, verbose = model.verbose, normalize = model.normalize, precompute = model.precompute, n_nonzero_coefs = model.n_nonzero_coefs, eps = model.eps, copy_X = model.copy_X, fit_path = model.fit_path, positive = model.positive)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::Lars, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:Lars}) = begin
            string("MLJModels.ScikitLearn_.", Lars)
        end
    MLJBase.package_name(::Type{<:Lars}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:Lars}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:Lars}) = begin
            false
        end
    MLJBase.package_url(::Type{<:Lars}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:Lars}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:Lars}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:Lars}) = begin
            true
        end
end
#  VotingRegressor
VotingRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.ensemble")).VotingRegressor
mutable struct VotingRegressor <: MLJBase.Deterministic
    weights::Any
    n_jobs::Union{Int, Any}
end
function VotingRegressor(; weights=nothing, n_jobs=nothing)
    model = VotingRegressor(weights, n_jobs)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::VotingRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = VotingRegressor_(weights = model.weights, n_jobs = model.n_jobs)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::VotingRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:VotingRegressor}) = begin
            string("MLJModels.ScikitLearn_.", VotingRegressor)
        end
    MLJBase.package_name(::Type{<:VotingRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:VotingRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:VotingRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:VotingRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:VotingRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:VotingRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:VotingRegressor}) = begin
            true
        end
end
#  MultiTaskLassoCV
MultiTaskLassoCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).MultiTaskLassoCV
mutable struct MultiTaskLassoCV <: MLJBase.Deterministic
    eps::Float64
    n_alphas::Int
    alphas::Any
    fit_intercept::Bool
    normalize::Bool
    max_iter::Int
    tol::Float64
    copy_X::Bool
    cv::Int
    verbose::Union{Bool, Int}
    n_jobs::Union{Int, Any}
    random_state::Int
    selection::String
end
function MultiTaskLassoCV(; eps=0.001, n_alphas=100, alphas=nothing, fit_intercept=true, normalize=false, max_iter=1000, tol=0.0001, copy_X=true, cv="warn", verbose=false, n_jobs=nothing, random_state=nothing, selection="cyclic")
    model = MultiTaskLassoCV(eps, n_alphas, alphas, fit_intercept, normalize, max_iter, tol, copy_X, cv, verbose, n_jobs, random_state, selection)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::MultiTaskLassoCV, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = MultiTaskLassoCV_(eps = model.eps, n_alphas = model.n_alphas, alphas = model.alphas, fit_intercept = model.fit_intercept, normalize = model.normalize, max_iter = model.max_iter, tol = model.tol, copy_X = model.copy_X, cv = model.cv, verbose = model.verbose, n_jobs = model.n_jobs, random_state = model.random_state, selection = model.selection)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::MultiTaskLassoCV, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:MultiTaskLassoCV}) = begin
            string("MLJModels.ScikitLearn_.", MultiTaskLassoCV)
        end
    MLJBase.package_name(::Type{<:MultiTaskLassoCV}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:MultiTaskLassoCV}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:MultiTaskLassoCV}) = begin
            false
        end
    MLJBase.package_url(::Type{<:MultiTaskLassoCV}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:MultiTaskLassoCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:MultiTaskLassoCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:MultiTaskLassoCV}) = begin
            true
        end
end
#  RidgeClassifier
RidgeClassifier_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).RidgeClassifier
mutable struct RidgeClassifier <: MLJBase.Deterministic
    alpha::Float64
    fit_intercept::Bool
    normalize::Bool
    copy_X::Bool
    max_iter::Int
    tol::Float64
    class_weight::Union{Any, Any}
    solver::Any
    random_state::Int
end
function RidgeClassifier(; alpha=1.0, fit_intercept=true, normalize=false, copy_X=true, max_iter=nothing, tol=0.001, class_weight=nothing, solver="auto", random_state=nothing)
    model = RidgeClassifier(alpha, fit_intercept, normalize, copy_X, max_iter, tol, class_weight, solver, random_state)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::RidgeClassifier, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = RidgeClassifier_(alpha = model.alpha, fit_intercept = model.fit_intercept, normalize = model.normalize, copy_X = model.copy_X, max_iter = model.max_iter, tol = model.tol, class_weight = model.class_weight, solver = model.solver, random_state = model.random_state)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::RidgeClassifier, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:RidgeClassifier}) = begin
            string("MLJModels.ScikitLearn_.", RidgeClassifier)
        end
    MLJBase.package_name(::Type{<:RidgeClassifier}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:RidgeClassifier}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:RidgeClassifier}) = begin
            false
        end
    MLJBase.package_url(::Type{<:RidgeClassifier}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:RidgeClassifier}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:RidgeClassifier}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:RidgeClassifier}) = begin
            true
        end
end
#  DecisionTreeRegressor
DecisionTreeRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.tree")).DecisionTreeRegressor
mutable struct DecisionTreeRegressor <: MLJBase.Deterministic
    criterion::String
    splitter::String
    max_depth::Union{Int, Any}
    min_samples_split::Int
    min_samples_leaf::Int
    min_weight_fraction_leaf::Float64
    max_features::Int
    random_state::Int
    max_leaf_nodes::Union{Int, Any}
    min_impurity_decrease::Float64
    min_impurity_split::Float64
    presort::Bool
end
function DecisionTreeRegressor(; criterion="mse", splitter="best", max_depth=nothing, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=nothing, random_state=nothing, max_leaf_nodes=nothing, min_impurity_decrease=0.0, min_impurity_split=nothing, presort=false)
    model = DecisionTreeRegressor(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, presort)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::DecisionTreeRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = DecisionTreeRegressor_(criterion = model.criterion, splitter = model.splitter, max_depth = model.max_depth, min_samples_split = model.min_samples_split, min_samples_leaf = model.min_samples_leaf, min_weight_fraction_leaf = model.min_weight_fraction_leaf, max_features = model.max_features, random_state = model.random_state, max_leaf_nodes = model.max_leaf_nodes, min_impurity_decrease = model.min_impurity_decrease, min_impurity_split = model.min_impurity_split, presort = model.presort)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::DecisionTreeRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:DecisionTreeRegressor}) = begin
            string("MLJModels.ScikitLearn_.", DecisionTreeRegressor)
        end
    MLJBase.package_name(::Type{<:DecisionTreeRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:DecisionTreeRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:DecisionTreeRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:DecisionTreeRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:DecisionTreeRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:DecisionTreeRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:DecisionTreeRegressor}) = begin
            true
        end
end
#  KernelRidge
KernelRidge_ = ((ScikitLearn.Skcore).pyimport("sklearn.kernel_ridge")).KernelRidge
mutable struct KernelRidge <: MLJBase.Deterministic
    alpha::Any
    kernel::Union{String, Any}
    gamma::Float64
    degree::Float64
    coef0::Float64
    kernel_params::Any
end
function KernelRidge(; alpha=1, kernel="linear", gamma=nothing, degree=3, coef0=1, kernel_params=nothing)
    model = KernelRidge(alpha, kernel, gamma, degree, coef0, kernel_params)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::KernelRidge, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = KernelRidge_(alpha = model.alpha, kernel = model.kernel, gamma = model.gamma, degree = model.degree, coef0 = model.coef0, kernel_params = model.kernel_params)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::KernelRidge, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:KernelRidge}) = begin
            string("MLJModels.ScikitLearn_.", KernelRidge)
        end
    MLJBase.package_name(::Type{<:KernelRidge}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:KernelRidge}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:KernelRidge}) = begin
            false
        end
    MLJBase.package_url(::Type{<:KernelRidge}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:KernelRidge}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:KernelRidge}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:KernelRidge}) = begin
            true
        end
end
#  HuberRegressor
HuberRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).HuberRegressor
mutable struct HuberRegressor <: MLJBase.Deterministic
    epsilon::Float64
    max_iter::Int
    alpha::Float64
    warm_start::Bool
    fit_intercept::Bool
    tol::Float64
end
function HuberRegressor(; epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=false, fit_intercept=true, tol=1.0e-5)
    model = HuberRegressor(epsilon, max_iter, alpha, warm_start, fit_intercept, tol)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::HuberRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = HuberRegressor_(epsilon = model.epsilon, max_iter = model.max_iter, alpha = model.alpha, warm_start = model.warm_start, fit_intercept = model.fit_intercept, tol = model.tol)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::HuberRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:HuberRegressor}) = begin
            string("MLJModels.ScikitLearn_.", HuberRegressor)
        end
    MLJBase.package_name(::Type{<:HuberRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:HuberRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:HuberRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:HuberRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:HuberRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:HuberRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:HuberRegressor}) = begin
            true
        end
end
#  IsotonicRegression
IsotonicRegression_ = ((ScikitLearn.Skcore).pyimport("sklearn.isotonic")).IsotonicRegression
mutable struct IsotonicRegression <: MLJBase.Deterministic
    y_min::Any
    y_max::Any
    increasing::Union{Bool, String}
    out_of_bounds::String
end
function IsotonicRegression(; y_min=nothing, y_max=nothing, increasing=true, out_of_bounds="nan")
    model = IsotonicRegression(y_min, y_max, increasing, out_of_bounds)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::IsotonicRegression, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = IsotonicRegression_(y_min = model.y_min, y_max = model.y_max, increasing = model.increasing, out_of_bounds = model.out_of_bounds)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::IsotonicRegression, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:IsotonicRegression}) = begin
            string("MLJModels.ScikitLearn_.", IsotonicRegression)
        end
    MLJBase.package_name(::Type{<:IsotonicRegression}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:IsotonicRegression}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:IsotonicRegression}) = begin
            false
        end
    MLJBase.package_url(::Type{<:IsotonicRegression}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:IsotonicRegression}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:IsotonicRegression}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:IsotonicRegression}) = begin
            true
        end
end
#  LarsCV
LarsCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).LarsCV
mutable struct LarsCV <: MLJBase.Deterministic
    fit_intercept::Bool
    verbose::Union{Bool, Int}
    max_iter::Int
    normalize::Bool
    precompute::Any
    cv::Int
    max_n_alphas::Int
    n_jobs::Union{Int, Any}
    eps::Float64
    copy_X::Bool
    positive::Any
end
function LarsCV(; fit_intercept=true, verbose=false, max_iter=500, normalize=true, precompute="auto", cv="warn", max_n_alphas=1000, n_jobs=nothing, eps=2.220446049250313e-16, copy_X=true, positive=false)
    model = LarsCV(fit_intercept, verbose, max_iter, normalize, precompute, cv, max_n_alphas, n_jobs, eps, copy_X, positive)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::LarsCV, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = LarsCV_(fit_intercept = model.fit_intercept, verbose = model.verbose, max_iter = model.max_iter, normalize = model.normalize, precompute = model.precompute, cv = model.cv, max_n_alphas = model.max_n_alphas, n_jobs = model.n_jobs, eps = model.eps, copy_X = model.copy_X, positive = model.positive)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::LarsCV, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:LarsCV}) = begin
            string("MLJModels.ScikitLearn_.", LarsCV)
        end
    MLJBase.package_name(::Type{<:LarsCV}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:LarsCV}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:LarsCV}) = begin
            false
        end
    MLJBase.package_url(::Type{<:LarsCV}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:LarsCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:LarsCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:LarsCV}) = begin
            true
        end
end
#  LassoLarsCV
LassoLarsCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).LassoLarsCV
mutable struct LassoLarsCV <: MLJBase.Deterministic
    fit_intercept::Bool
    verbose::Union{Bool, Int}
    max_iter::Int
    normalize::Bool
    precompute::Any
    cv::Int
    max_n_alphas::Int
    n_jobs::Union{Int, Any}
    eps::Float64
    copy_X::Bool
    positive::Any
end
function LassoLarsCV(; fit_intercept=true, verbose=false, max_iter=500, normalize=true, precompute="auto", cv="warn", max_n_alphas=1000, n_jobs=nothing, eps=2.220446049250313e-16, copy_X=true, positive=false)
    model = LassoLarsCV(fit_intercept, verbose, max_iter, normalize, precompute, cv, max_n_alphas, n_jobs, eps, copy_X, positive)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::LassoLarsCV, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = LassoLarsCV_(fit_intercept = model.fit_intercept, verbose = model.verbose, max_iter = model.max_iter, normalize = model.normalize, precompute = model.precompute, cv = model.cv, max_n_alphas = model.max_n_alphas, n_jobs = model.n_jobs, eps = model.eps, copy_X = model.copy_X, positive = model.positive)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::LassoLarsCV, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:LassoLarsCV}) = begin
            string("MLJModels.ScikitLearn_.", LassoLarsCV)
        end
    MLJBase.package_name(::Type{<:LassoLarsCV}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:LassoLarsCV}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:LassoLarsCV}) = begin
            false
        end
    MLJBase.package_url(::Type{<:LassoLarsCV}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:LassoLarsCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:LassoLarsCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:LassoLarsCV}) = begin
            true
        end
end
#  LinearRegression
LinearRegression_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).LinearRegression
mutable struct LinearRegression <: MLJBase.Deterministic
    fit_intercept::Bool
    normalize::Bool
    copy_X::Bool
    n_jobs::Union{Int, Any}
end
function LinearRegression(; fit_intercept=true, normalize=false, copy_X=true, n_jobs=nothing)
    model = LinearRegression(fit_intercept, normalize, copy_X, n_jobs)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::LinearRegression, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = LinearRegression_(fit_intercept = model.fit_intercept, normalize = model.normalize, copy_X = model.copy_X, n_jobs = model.n_jobs)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::LinearRegression, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:LinearRegression}) = begin
            string("MLJModels.ScikitLearn_.", LinearRegression)
        end
    MLJBase.package_name(::Type{<:LinearRegression}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:LinearRegression}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:LinearRegression}) = begin
            false
        end
    MLJBase.package_url(::Type{<:LinearRegression}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:LinearRegression}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:LinearRegression}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:LinearRegression}) = begin
            true
        end
end
#  OrthogonalMatchingPursuit
OrthogonalMatchingPursuit_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).OrthogonalMatchingPursuit
mutable struct OrthogonalMatchingPursuit <: MLJBase.Deterministic
    n_nonzero_coefs::Int
    tol::Float64
    fit_intercept::Bool
    normalize::Bool
    precompute::Any
end
function OrthogonalMatchingPursuit(; n_nonzero_coefs=nothing, tol=nothing, fit_intercept=true, normalize=true, precompute="auto")
    model = OrthogonalMatchingPursuit(n_nonzero_coefs, tol, fit_intercept, normalize, precompute)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::OrthogonalMatchingPursuit, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = OrthogonalMatchingPursuit_(n_nonzero_coefs = model.n_nonzero_coefs, tol = model.tol, fit_intercept = model.fit_intercept, normalize = model.normalize, precompute = model.precompute)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::OrthogonalMatchingPursuit, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:OrthogonalMatchingPursuit}) = begin
            string("MLJModels.ScikitLearn_.", OrthogonalMatchingPursuit)
        end
    MLJBase.package_name(::Type{<:OrthogonalMatchingPursuit}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:OrthogonalMatchingPursuit}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:OrthogonalMatchingPursuit}) = begin
            false
        end
    MLJBase.package_url(::Type{<:OrthogonalMatchingPursuit}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:OrthogonalMatchingPursuit}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:OrthogonalMatchingPursuit}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:OrthogonalMatchingPursuit}) = begin
            true
        end
end
#  NuSVR
NuSVR_ = ((ScikitLearn.Skcore).pyimport("sklearn.svm")).NuSVR
mutable struct NuSVR <: MLJBase.Deterministic
    nu::Float64
    C::Float64
    kernel::String
    degree::Int
    gamma::Float64
    coef0::Float64
    shrinking::Bool
    tol::Float64
    cache_size::Float64
    verbose::Bool
    max_iter::Int
end
function NuSVR(; nu=0.5, C=1.0, kernel="rbf", degree=3, gamma="auto_deprecated", coef0=0.0, shrinking=true, tol=0.001, cache_size=200, verbose=false, max_iter=-1)
    model = NuSVR(nu, C, kernel, degree, gamma, coef0, shrinking, tol, cache_size, verbose, max_iter)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::NuSVR, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = NuSVR_(nu = model.nu, C = model.C, kernel = model.kernel, degree = model.degree, gamma = model.gamma, coef0 = model.coef0, shrinking = model.shrinking, tol = model.tol, cache_size = model.cache_size, verbose = model.verbose, max_iter = model.max_iter)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::NuSVR, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:NuSVR}) = begin
            string("MLJModels.ScikitLearn_.", NuSVR)
        end
    MLJBase.package_name(::Type{<:NuSVR}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:NuSVR}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:NuSVR}) = begin
            false
        end
    MLJBase.package_url(::Type{<:NuSVR}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:NuSVR}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:NuSVR}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:NuSVR}) = begin
            true
        end
end
#  MLPRegressor
MLPRegressor_ = ((ScikitLearn.Skcore).pyimport("sklearn.neural_network")).MLPRegressor
mutable struct MLPRegressor <: MLJBase.Deterministic
    hidden_layer_sizes::Any
    activation::Any
    solver::Any
    alpha::Float64
    batch_size::Int
    learning_rate::Any
    learning_rate_init::Any
    power_t::Any
    max_iter::Int
    shuffle::Bool
    random_state::Int
    tol::Float64
    verbose::Bool
    warm_start::Bool
    momentum::Float64
    nesterovs_momentum::Bool
    early_stopping::Bool
    validation_fraction::Float64
    beta_1::Float64
    beta_2::Float64
    epsilon::Float64
    n_iter_no_change::Int
end
function MLPRegressor(; hidden_layer_sizes=nothing, activation="relu", solver="adam", alpha=0.0001, batch_size="auto", learning_rate="constant", learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=true, random_state=nothing, tol=0.0001, verbose=false, warm_start=false, momentum=0.9, nesterovs_momentum=true, early_stopping=false, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1.0e-8, n_iter_no_change=10)
    model = MLPRegressor(hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init, power_t, max_iter, shuffle, random_state, tol, verbose, warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::MLPRegressor, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = MLPRegressor_(hidden_layer_sizes = model.hidden_layer_sizes, activation = model.activation, solver = model.solver, alpha = model.alpha, batch_size = model.batch_size, learning_rate = model.learning_rate, learning_rate_init = model.learning_rate_init, power_t = model.power_t, max_iter = model.max_iter, shuffle = model.shuffle, random_state = model.random_state, tol = model.tol, verbose = model.verbose, warm_start = model.warm_start, momentum = model.momentum, nesterovs_momentum = model.nesterovs_momentum, early_stopping = model.early_stopping, validation_fraction = model.validation_fraction, beta_1 = model.beta_1, beta_2 = model.beta_2, epsilon = model.epsilon, n_iter_no_change = model.n_iter_no_change)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::MLPRegressor, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:MLPRegressor}) = begin
            string("MLJModels.ScikitLearn_.", MLPRegressor)
        end
    MLJBase.package_name(::Type{<:MLPRegressor}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:MLPRegressor}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:MLPRegressor}) = begin
            false
        end
    MLJBase.package_url(::Type{<:MLPRegressor}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:MLPRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:MLPRegressor}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:MLPRegressor}) = begin
            true
        end
end
#  RidgeCV
RidgeCV_ = ((ScikitLearn.Skcore).pyimport("sklearn.linear_model")).RidgeCV
mutable struct RidgeCV <: MLJBase.Deterministic
    alphas::Any
    fit_intercept::Bool
    normalize::Bool
    scoring::String
    cv::Int
    gcv_mode::Any
    store_cv_values::Bool
end
function RidgeCV(; alphas=nothing, fit_intercept=true, normalize=false, scoring=nothing, cv=nothing, gcv_mode=nothing, store_cv_values=false)
    model = RidgeCV(alphas, fit_intercept, normalize, scoring, cv, gcv_mode, store_cv_values)
    message = MLJBase.clean!(model)
    isempty(message) || @warn(message)
    return model
end
function MLJBase.fit(model::RidgeCV, verbosity::Int, X, y)
    Xmatrix = MLJBase.matrix(X)
    cache = RidgeCV_(alphas = model.alphas, fit_intercept = model.fit_intercept, normalize = model.normalize, scoring = model.scoring, cv = model.cv, gcv_mode = model.gcv_mode, store_cv_values = model.store_cv_values)
    result = ScikitLearn.fit!(cache, Xmatrix, y)
    fitresult = result
    report = NamedTuple{}()
    return (fitresult, nothing, report)
end
function MLJBase.predict(model::RidgeCV, fitresult, Xnew)
    xnew = MLJBase.matrix(Xnew)
    prediction = ScikitLearn.predict(fitresult, xnew)
    return prediction
end
begin
    MLJBase.load_path(::Type{<:RidgeCV}) = begin
            string("MLJModels.ScikitLearn_.", RidgeCV)
        end
    MLJBase.package_name(::Type{<:RidgeCV}) = begin
            "ScikitLearn"
        end
    MLJBase.package_uuid(::Type{<:RidgeCV}) = begin
            "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
        end
    MLJBase.is_pure_julia(::Type{<:RidgeCV}) = begin
            false
        end
    MLJBase.package_url(::Type{<:RidgeCV}) = begin
            "https://github.com/cstjean/ScikitLearn.jl"
        end
    MLJBase.input_scitype_union(::Type{<:RidgeCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.target_scitype_union(::Type{<:RidgeCV}) = begin
            MLJBase.Continuous
        end
    MLJBase.input_is_multivariate(::Type{<:RidgeCV}) = begin
            true
        end
end
