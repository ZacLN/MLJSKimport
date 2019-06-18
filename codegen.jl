using CSV, CSTParser
function types_map(t::AbstractString)
    if occursin(" or ", t)
        return Expr(:curly, :Union, types_map.(split(t, " or "))...)
    end
    if t in ("bool", "boolean")
        return :Bool
    elseif t in ("integer", "int")
        return :Int
    elseif t == "positive integer"
        return :Int
    elseif t == "float"
        return :Float64
    elseif startswith(t, "{'") || t == "string" || t == "str"
        return :String
    elseif t == "function"
        return :Function
    else 
        return :Any
    end 
end

function tryparse(d)
    try
        return Meta.parse(d)        
    catch e
        return d
    end
end
function default_map(d)
    pd = tryparse(d)
    
    if ismissing(d)
        return :nothing
    elseif pd isa Number
        return pd
    elseif d == "None"
        return :nothing
    elseif d == "True"
        return :true
    elseif d == "False"
        return :false
    elseif startswith(d, "’")
        return strip(d, '’')
    else
        return :nothing
    end
end

function get_support(t)
    
end

function get_data(data_address = "https://raw.githubusercontent.com/alan-turing-institute/MLJ.jl/master/material/sklearn_reggressors.csv")
    f = download(data_address)
    table = CSV.read(f)
end

function code_gen(table = get_data())
    models = Dict()
    for model in unique(collect(table[:class]))
        s = split(model, ".")
        root = join(s[1:end-1], ".")
        if !startswith(root, "sklearn")
            root = root[first(findfirst("sklearn", root)):end]
        end
        mname = Symbol(last(s))

        import_statement = :($(Symbol(string(mname, "_"))) = (ScikitLearn.Skcore.pyimport)($root).$(mname))

        struct_block = Expr(:block)
        constructor_kw = Expr(:parameters)
        constructor_call = Expr(:call, mname)

        fit_call = :(MLJBase.fit(model::$mname, verbosity::Int, X, y))
        fit_block = Expr(:block, :(Xmatrix = MLJBase.matrix(X)))
        sk_call = Expr(:call, Symbol(string(mname, "_")))


        mod1 = filter(r->r.class == model, table)
        for i = 0:maximum(mod1.Column1)
            field = filter(r-> r.Column1 == i, mod1)
            fn =Symbol(field[3][1])
            ft = types_map(field[4][1])
            f_default = default_map(field[5][1])
            push!(struct_block.args, Expr(:(::), fn, ft))
            push!(constructor_kw.args, Expr(:kw, fn, f_default))
            push!(constructor_call.args, fn)
            push!(sk_call.args, Expr(:(=), fn, :(model.$fn)))
        end
        struct_ex = Expr(:struct, true, :($(mname) <: MLJBase.Deterministic), struct_block)
        constructor_ex = :(function $mname($constructor_kw)
                model = $constructor_call
                message = MLJBase.clean!(model)
                isempty(message) || @warn message
                return model
            end)
        

        push!(fit_block.args, :(cache = $sk_call))
        push!(fit_block.args, :(result = ScikitLearn.fit!(cache, Xmatrix, y)))
        push!(fit_block.args, :(fitresult = result))
        ## get attributes (needs input data)
        res_attr = :(NamedTuple{}())
        push!(fit_block.args, :(report = $res_attr))
        push!(fit_block.args, :(return fitresult, nothing, report))
        fit_ex = Expr(:function, fit_call, fit_block)
        predict_ex = :(function MLJBase.predict(model::$mname
                , fitresult
                , Xnew)
                xnew = MLJBase.matrix(Xnew)
                prediction = ScikitLearn.predict(fitresult,xnew)
                return prediction
            end)
        traits = quote
            MLJBase.load_path(::Type{<:$mname}) = string("MLJModels.ScikitLearn_.",$mname)
            MLJBase.package_name(::Type{<:$mname}) = "ScikitLearn"
            MLJBase.package_uuid(::Type{<:$mname}) = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
            MLJBase.is_pure_julia(::Type{<:$mname}) = false
            MLJBase.package_url(::Type{<:$mname}) = "https://github.com/cstjean/ScikitLearn.jl"
            MLJBase.input_scitype_union(::Type{<:$mname}) = MLJBase.Continuous
            MLJBase.target_scitype_union(::Type{<:$mname}) = MLJBase.Continuous
            MLJBase.input_is_multivariate(::Type{<:$mname}) = true
        end
        CSTParser.remlineinfo!(import_statement)
        CSTParser.remlineinfo!(struct_ex)
        CSTParser.remlineinfo!(constructor_ex)
        CSTParser.remlineinfo!(fit_ex)
        CSTParser.remlineinfo!(predict_ex)
        CSTParser.remlineinfo!(traits)
        models[mname] = (import_statement, struct_ex, constructor_ex, fit_ex, predict_ex, traits)
    end
    models
end

function writecode(f = "output.jl", models = code_gen())
    io = open(f, "w")
    for (m,exs) in models
        println(io, "#  $m")
        for e in exs
            println(io, e)
        end
        println()
    end
    close(io)

end

