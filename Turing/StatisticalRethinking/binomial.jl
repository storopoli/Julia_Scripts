using Turing, CSV, HTTP, DataFrames, LazyArrays

df = CSV.File(
    HTTP.get("https://github.com/StatisticalRethinkingJulia/TuringModels.jl/blob/master/data/UCBadmit.csv?raw=TRUE").body; delim=';') |>
    DataFrame

# Data Prep
dept_map = Dict(key => idx for (idx, key) in enumerate(unique(df.dept)))
df.male = [g == "male" ? 1 : 0 for g in df.gender]
df.dept_id = [dept_map[de] for de in df.dept]
df

# Model
@model binomial_model(applications, dept_id, male, admit) = begin
    sigma_dept ~ truncated(Cauchy(0, 2), 0, Inf)
    a ~ Normal(0, 10)
    a_dept ~ filldist(Normal(a, sigma_dept), 6)

    logit_p = a_dept[dept_id]
    # admit .~ BinomialLogit.(applications, logit_p)
    admit ~ arraydist(LazyArray(@~ BinomialLogit.(applications, logit_p)))
end

model = binomial_model(df.applications, df.dept_id, df.male, df.admit)

chn = sample(model, NUTS(1_000, 0.65), MCMCThreads(), 2_000, 4)
chn = sample(model, NUTS(1_000, 0.65), 2_000)

