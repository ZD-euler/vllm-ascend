# Historical reference — not current acceptance

The 2026-11-07 public Q4K trace pair is a reference for experiment scope and
observed results. Its allowlisted numeric fields and source-file SHA256 are in
`references/historical-trace-reference.json`. No historical PASS is consumed by
the new launcher as a gate.

Both arms used H524288/Q4096, 78 layers, TP4/DP4/EP16, local BS1/global BS4,
chunk6144, one warmup and one profile burst. All sixteen ranks executed; only
rank 0 profiled. Comparison covered eight requests and 160 logprob values with
observed maximum absolute difference 0.0. The original comparison tolerance is
1e-5; future reports separately state whether the result is exactly equal.

| Single profiled diagnostic | native | M3v15 |
| --- | ---: | ---: |
| Maximum rank model-forward time, ms | 5287.396 | 4659.673 |
| All first tokens returned, ms | 6493.637 | 5776.849 |
| Maximum framework allocated peak, GiB | 51.331 | 51.639 |

These are one profiled sample per arm, with profiler overhead. Parallel rank
intervals are not summed. Dummy counters cover a wider bench-begin-to-collect
interval. The table is not an unprofiled performance estimate, statistical
benchmark or guaranteed speedup ratio.

The M3v15 code label is commit `db85ccd37d8fe90b9835903355f3bc92c1220e85`;
baseline is `fbb4d867d22095047473e3c7ca9db053f5f4c7b0`. The original deployment
attestation covered nine runtime production files. It did not prove that the
entire remote Git tree was frozen or identical at that historical instant.
The current package's pinned archives and later content inventories are
separate evidence and do not retroactively expand that attestation.

Run this package's fresh CPU gates, both smoke arms, trace capture and numerical
comparison to establish a new result. See `VALIDATION.md` for what was actually
executed during packaging and what remains untested.
