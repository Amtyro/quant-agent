# A1: why the iceberg x spoofing interaction (H1) was not supported

Numbers come from `results_a1/diagnosis.md` (made by `a1_diagnose.py` from the saved pilot and confirmation runs).

## Short answer
The implementation is not the cause. The main reason is the **design**: with a genuine sell fixed at r~ + offset, the
true interaction on spoofer surplus is structurally tiny. A **noisy outcome measure** then made a chance pilot result
look significant. The HBL model itself also predicts that the two levers act almost additively.

## 1. Technical (implementation): ruled out as the main cause
* The 6 checks pass: price-time priority, level pricing, conservation, iceberg invariants, belief equals upstream on unit
  orders, and arms paired. Visible and iceberg arms are identical when no HBL is present.
* The results match the predictions where they should: arms are exactly equal where the ask is outside HBL's
  belief-relevant range (A1, offsets 50 and 100), and Gu et al.'s liquidity gradient is reproduced.
* Remaining risk: for multi-unit orders, an unfilled remainder counts as "rejected", which is a departure from upstream.
  This sets how large the display-size effects are, but it cannot create or remove an interaction between the two
  levers. There is also only one iceberg variant (synthetic, peak 1, refilled at the back of the queue).

## 2. Data / statistics: why the pilot was a false positive
* **Terminal-value noise.** Surplus = cash + position x r_T. All arms of a seed share r_T, so
  Gamma(surplus) = Gamma(cash) + Gamma(position) x r_T. The term Gamma(position) x (r_T - mean r_T) explains
  **R^2 = 0.90-0.99** of the seed-level variance of Gamma, even though its mean is about 0. Removing it cuts the
  per-seed SD from about 220-410 to about 8-90, a 5-10x reduction.
* **Pilot C3 after removing this noise:**

  | Offset | Raw Gamma | Gamma with noise removed |
  |---|---|---|
  | 1 | -44.5* | +1.1 [-3.6, +5.5] |
  | 50 | -34.7* | -5.2 [-11.6, +2.0] |

  The "substitution" signal was terminal-value noise.
* **Multiplicity.** The pilot ran 60 interaction tests and 3 had CIs excluding 0. That is exactly the 3.0 expected
  by chance.
* **Power.** With a raw SD of about 300 and 400 seeds, the SE of Gamma is about 15-20. The minimum detectable effect
  at 80% power is therefore about 45-55, larger than any plausible true effect (see 3).

## 3. Design: why the true effect has to be small
* The genuine leg's price is pinned at r~_t + offset, and r~_t = E[r_T | r_t]. Each unit sold therefore earns the
  offset in expectation, so spoofing or hiding can raise spoofer surplus only by changing **how many** units fill.
* The effect of spoofing on units sold is almost the same with a visible or an iceberg genuine leg. In C3 at
  offset 1, spoofing adds +0.60 units with a visible leg and +0.58 with an iceberg (D_V vs D_I sold). Gamma(units sold)
  is 0 +/- 0.1 in every cell, so offset x Gamma(sold) is at most about +/-3 surplus.
* Other features of the setup that shrink the effect:
  * The market is thin: the spoofer sells 1-9 units per run.
  * The parent order is small (G = 10) and is cancelled and re-posted at every arrival, so there is no pressure to
    get a large order executed. In real enforcement cases, the hidden genuine order is a large order being worked
    through the book.
  * Spoof size, offset and peak are fixed rather than chosen by the spoofer.
  * The mix of background traders is fixed.

## 4. Mechanism (HBL beliefs): additivity is the model's own prediction
* The spoof bid works on **HBL buyers**. Its unfilled 200 units count as "rejected" evidence at prices at or below the
  best bid, so buyers quote higher and the midprice rises. This is Gu et al.'s channel, and we reproduce it.
* The displayed size of the genuine ask works on **HBL sellers**. A large unfilled ask counts as rejected evidence for
  sells at or above its price, so HBL sellers undercut it and take the fills. Hiding the size removes that undercutting,
  which is the crowding-out result (about +0.3 units for the hidden seller).
* The two levers act on different price regions of the belief curve, and on different sides of the market. An
  iceberg with peak 1 also still shows the price. Gu et al.'s point that the spoofer's own ask works against its spoof
  bid is about where that ask sits, not how big it is: their spoofer already shows a single unit. So hiding size leaves
  Gu's counter-effect in place.

## What this means for the next step

| | Lower-variance surplus rerun | Strategic EGTA |
|---|---|---|
| Question | Same fixed-strategy question, answered with a sharper estimator | Does the option to use icebergs change the strategies chosen in equilibrium and the welfare that results? |
| Change | Value each fill at r~ at the time it trades (and for background traders, mark positions to r~). This stays unbiased because r_T - r~_t has mean 0 given everything known at t. | Search over the spoofer's (offset, G, peak, spoof size) and the HBL/ZI mix; solve for equilibria and regret. |
| Expected outcome | A **precise null**: Gamma within about +/-10 instead of +/-40, which supports reporting the levers as additive. | Could bring H1 back in a strategic form, e.g. icebergs make a larger or more aggressive genuine leg worth using. |
| Cost | Add per-fill r~ to the output and rerun the 19,200 confirmation runs on the same seeds (about 1 hour). | Many strategy profiles times many seeds each: days of compute. |
| Necessity | Prerequisite: EGTA payoffs are simulation estimates, and with the current noise each profile needs 25-100x more samples. | Needed for a paper in the Wellman lineage, where fixed-strategy results are only a partial-equilibrium step. |

Recommended order: first the lower-variance rerun (cheap, and it makes the null informative), then EGTA using the
same measure.
