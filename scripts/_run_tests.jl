# Scratch test runner (untracked). `runtests.jl` with no args already runs every segment,
# so this only exists to filter the output down to the lines worth reading.
#
#   julia --project=. scripts/_run_tests.jl              # all segments
#   julia --project=. scripts/_run_tests.jl core models  # named segments only
const SEGS = isempty(ARGS) ? String[] : ARGS
cmd = `julia --project=. $(joinpath(@__DIR__, "..", "test", "runtests.jl"))`
run(pipeline(cmd; stdout = stdout, stderr = stderr))
