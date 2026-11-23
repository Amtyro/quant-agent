# A1 pilot: iceberg genuine leg x spoofing — design notes

## Question
Gu et al. (ICAIF 2024, Sec. 8.1) show that the spoofer's own visible sell order enters HBL beliefs like a
market maker's ask and works against its spoof bid. Real enforcement cases show spoofers hiding the
genuine side with icebergs (Debie et al. 2023; Verhulst & Pennings 2025). A1 asks: if the genuine leg is
an iceberg, does spoofing become more profitable / more harmful, and through which channel?

## Environment (Gu et al. 2024, Sec. 4)
T=10,000; 12 ZI + 12 HBL (L=4), lam_bg=0.002, q_max=10, shading [250,500], sigma_PV^2=5e6;
fundamental mean 1e5, kappa=0.05, shock variance 1e4; ladder MM with K=8, lam_MM=0.035 and
(xi, omega) from Gu's Table 1 (A1 = most liquid, B2 = middle, C3 = least liquid); spoofer arrives at
lam_SP=0.02 from t=1000, withdraws its orders, posts the genuine sell at r~_t + offset, then (if spoofing)
a 200-unit bid one tick behind the best bid of others.

## Arms (all share seeds: fundamental, arrivals, private values, trader random draws, within-step order)
| arm | seller present | spoof bid | genuine leg |
|---|---|---|---|
| base | no | – | – |
| nsV1 / spV1 | yes | no / yes | 1 unit, visible (Gu's spoofer) |
| nsVG / spVG | yes | no / yes | G units, fully visible |
| nsIG / spIG | yes | no / yes | G units, iceberg with peak 1, replenished at the back of the queue |

Interaction: Gamma = (spIG - nsIG) - (spVG - nsVG), computed seed by seed.

## Implementation choices that differ from upstream PyMarketSim (and why)
* Order book rebuilt: price-time priority (upstream breaks same-price ties by order id), executions at
  each resting order's price (upstream clears all matches of a step at one price), multi-unit partial
  fills that do not mutate historical records (upstream shrinks the historical Order object), and
  self-trade prevention.
* Continuous processing: each order is matched on arrival (upstream batches a time step).
* HBL learns from a public ledger: (submitted price, displayed qty, time, fills, cancel time). Hidden
  iceberg quantity and owner identity are never used. Withdrawal times come from public cancels instead of
  upstream's "next order by the same agent id" inference; on unit-order histories the two coincide and the
  belief equals upstream's exactly (test_belief_matches_upstream_on_unit_orders).
* Pricing optimisation is upstream HBLAgent.determine_optimal_price unchanged.
* For multi-unit orders, the unfilled remainder counts as rejected (upstream: an order that traded at
  all is never rejected — identical for unit orders).

## Checks (experiments/test_a1sim.py)
FIFO and level pricing; partial fills + self-trade prevention; cash/position conservation; iceberg
visible size <= peak; no parent sells more than G; VIS == ICE exactly when no HBL is present; arms are
paired (an invisible, untradeable seller leaves the market unchanged); belief equals upstream.

## Pilot 1 (seeds 100000-100399) -> pre-specified confirmation (seeds 200000-200399)
Pilot 1 was exploratory (A1/B2/C3, offsets 1 and 50, G=10, peak 1). Written down before the
confirmation run, using fresh seeds, the three claims to confirm are:
1. Substitution: in the least liquid market C3, Gamma(spoofer surplus) = D_I - D_V < 0 at offsets 1 and 50
   (pilot: -44.5 and -34.7, CIs excluding 0).
2. Hiding helps the seller without spoofing: nsI - nsV spoofer surplus > 0 in C3 (pilot: +36.8, +45.9).
3. Hiding hurts background traders: spI - spV total background surplus < 0 (pilot: CI < 0 in 5 of 6 cells).
Offsets 25 and 100 are added to check that the pattern is not tied to the two pilot offsets;
counterparty composition is recorded to test the channel (iceberg -> more of the seller's units bought).

## Rerun with fill-time valuation (written 2026-11-22, before the rerun)
Diagnosis (`A1_DIAGNOSIS.md`) showed that 90-99% of the seed-level variance of Gamma(spoofer surplus) comes from
position differences times the terminal fundamental r_T. The rerun adds, for every trader, a surplus that values each
filled unit at r~_t (the estimate of r_T when it trades): sum of side * q * (r~_t - p), plus private value for background
traders. Because fills at t depend only on information up to t, and r~_t = E[r_T | info at t], this measure has the
same expectation as the original. The original fields are kept unchanged.

Runs: the pilot design on seeds 100000-100399 (all 13 arms) and the confirmation design on seeds 200000-200399
(core arms, offsets 1/25/50/100). The simulations are the same as before; only the outputs are added.

Claims, fixed before the rerun:
1. Determinism: every original output equals the earlier run seed by seed.
2. Same expectation: the fill-time and original surplus contrasts have the same mean (paired difference CI includes 0).
3. Precision: in every non-degenerate cell, the per-seed SD of Gamma(spoofer surplus) is at least 3x smaller with the
   fill-time measure.
4. H1 re-test (primary): in C3 at offsets 1 and 50, Gamma(spoofer surplus, fill-time) has a CI that includes 0 and
   lies inside +/-15, on both seed sets; no cell shows the pilot's negative interaction.
5. Crowding out: hiding without spoofing (nsI - nsV) lowers fill-time total background surplus in B2 and C3 (CI < 0),
   and raises the hidden seller's fill-time surplus by about offset x (extra units sold).
