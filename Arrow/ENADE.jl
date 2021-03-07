using CSV, Arrow, Chain
using BenchmarkTools, Query, Tables, DataFrames, Statistics
using Statistics
Threads.nthreads()

dir = "/Volumes/SSD_DUDA/MEC_data/enade_jl/"

# 31s
@time df = CSV.File(dir * "processed_data/ENADE.csv") |> DataFrame

# 1.3s
@time tbl = Arrow.Table(dir * "processed_data/ENADE.arrow")


## DataFrame Stuff ##

# 14.5s (750 alloc: 2.31GiB)
@btime @chain df begin
    dropmissing([:NT_GER, :NT_FG, :NT_CE])
    groupby([:TP_SEXO, :CO_TURNO_GRADUACAO], skipmissing=true)
    combine(nrow,
            [:NT_GER, :NT_FG, :NT_CE] .=> mean)
end

# 64s (240862625 allocations: 13.86 GiB)
@btime df |>
    @dropna(:NT_GER, :NT_FG, :NT_CE, :TP_SEXO, :CO_TURNO_GRADUACAO) |>
    @groupby({_.TP_SEXO, _.CO_TURNO_GRADUACAO}) |>
    @map({
        sexo = key(_)[1],
        turno = key(_)[2],
        nrows = length(_),
        NT_GER_mean = mean(_.NT_GER),
        NT_FG_mean = mean(_.NT_FG),
        NT_CE_mean = mean(_.NT_CE)
    }) |>
    DataFrame

## Arrow Stuff ##

# 47s (200743310 allocations: 8.60 GiB)
@btime @chain DataFrame(tbl) begin
    dropmissing([:NT_GER, :NT_FG, :NT_CE])
    groupby([:TP_SEXO, :CO_TURNO_GRADUACAO], skipmissing=true)
    combine(nrow,
            [:NT_GER, :NT_FG, :NT_CE] .=> mean)
end

# 205s (960678695 allocations: 29.13 GiB)
@btime Tables.datavaluerows(tbl) |>
    @dropna(:NT_GER, :NT_FG, :NT_CE, :TP_SEXO, :CO_TURNO_GRADUACAO) |>
    @groupby({_.TP_SEXO, _.CO_TURNO_GRADUACAO}) |>
    @map({
        sexo = key(_)[1],
        turno = key(_)[2],
        nrows = length(_),
        NT_GER_mean = mean(_.NT_GER),
        NT_FG_mean = mean(_.NT_FG),
        NT_CE_mean = mean(_.NT_CE)
    }) |>
    DataFrame
