using Parquet, DataFrames, CSV, Chain
using BenchmarkTools, Statistics

# 11.5s (28.93 M allocations: 2.237 GiB)
# RAM 11.8GB -> 12.7GB (File has 209mb)
@time df = DataFrame(CSV.File("/Volumes/SSD_DUDA/MEC_data/enade_jl/data/microdados_enade_2019.txt", delim=';', decimal=',', missingstrings=["", "NA"]))

# 7.6s (23.26 M allocations: 2.591 GiB)
# RAM 12.7GB -> 13.5GB (File has 209mb)
@time df2 = DataFrame(read_parquet("/Volumes/SSD_DUDA/MEC_data/enade/data/parquet/2019/data.parquet"))

# 105ms (983 allocations: 395.48 MiB)
@btime @chain df begin
    dropmissing([:NT_GER, :NT_FG, :NT_CE])
    groupby([:TP_SEXO, :CO_TURNO_GRADUACAO], skipmissing=true)
    combine(nrow,
            [:NT_GER, :NT_FG, :NT_CE] .=> mean)
end

function transform_grades(x::Union{Float64,String,Missing})
    if x isa String
        parse(Float64, replace(x, "," => "."))
    else
        x
    end
end

df2.NT_GER = transform_grades.(df2.NT_GER)
df2.NT_FG = transform_grades.(df2.NT_FG)
df2.NT_CE = transform_grades.(df2.NT_CE)

#  134ms (902 allocations: 336.33 MiB)
@btime @chain df2 begin
    dropmissing([:NT_GER, :NT_FG, :NT_CE])
    groupby([:TP_SEXO, :CO_TURNO_GRADUACAO], skipmissing=true)
    combine(nrow,
            [:NT_GER, :NT_FG, :NT_CE] .=> mean)
end
