"""A1 pilot: does an iceberg genuine leg change the profitability and harm of spoofing?

Environment follows Gu et al. (ICAIF 2024, "The Effect of Liquidity on the Spoofability of Financial
Markets"): 12 ZI + 12 HBL background traders, one ladder market maker (MM) that sets liquidity, and one
strategic seller ("spoofer") that, at every arrival, withdraws its orders, posts a genuine sell order at
r~_t + offset and (optionally) a large spoof buy order one tick behind the best bid.

What is new here:
* the genuine leg has size G and is either fully displayed (VIS) or an iceberg (ICE) whose visible
  child order holds at most ``peak`` units and is replenished, at the back of the queue, when filled;
* the order book is rebuilt from scratch with price-time priority, multi-unit orders, partial fills at
  the resting order's price, and self-trade prevention (the upstream four-heap breaks ties by order id,
  clears a whole step at one price, and mutates historical orders on partial fills);
* HBL traders learn from an immutable public ledger (submitted price/quantity/time, fills, cancel
  times). Hidden iceberg quantity never enters the ledger. The pricing optimisation itself is the
  upstream ``HBLAgent.determine_optimal_price`` (same anchor points and spline interpolation); only the
  belief evaluation is re-implemented (vectorised, and partial-fill aware). On unit-order histories it
  reproduces the upstream belief exactly (see test_a1sim.py).

Every source of randomness has its own seeded stream, so arms that share a seed differ only through
the spoofer's behaviour.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from marketsim.agent.hbl_agent import HBLAgent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.private_values.private_values import PrivateValues

SPOOFER = "SP"
MM = "MM"


# ----------------------------------------------------------------------------- config
@dataclass
class Config:
    n_zi: int = 12
    n_hbl: int = 12
    T: int = 10_000
    lam_bg: float = 0.002
    mean: float = 1e5
    kappa: float = 0.05
    shock_var: float = 1e4
    pv_var: float = 5e6
    q_max: int = 10
    shade: Tuple[float, float] = (250.0, 500.0)
    hbl_L: int = 4
    # market maker (Gu et al.: K=8, lam=0.035; xi = rung spacing, omega = spread)
    mm: bool = True
    lam_mm: float = 0.035
    mm_xi: float = 10.0
    mm_omega: float = 16.0
    mm_K: int = 8
    # strategic seller
    spoofer: bool = False      # is the strategic seller present at all
    spoof: bool = False        # does it also post the spoof buy order
    lam_sp: float = 0.02
    sp_start: int = 1000
    spoof_qty: int = 200
    sell_offset: float = 1.0   # genuine sell at r~_t + offset
    sell_qty: int = 1          # G: units offered by the genuine leg at each arrival
    iceberg: bool = False      # display mode of the genuine leg
    peak: int = 1              # visible size of an iceberg child
    tick: float = 1.0
    mid_grid: int = 50         # sample the midprice every mid_grid steps
    diag: bool = True          # record HBL decision diagnostics

    @property
    def n_bg(self) -> int:
        return self.n_zi + self.n_hbl


ENVS = {  # Gu et al. Table 1: (xi, omega)
    "A1": (10, 16), "A2": (50, 16), "A3": (100, 16),
    "B1": (10, 64), "B2": (50, 64), "B3": (100, 64),
    "C1": (10, 256), "C2": (50, 256), "C3": (100, 256),
}


# ----------------------------------------------------------------------------- ledger
class Rec:
    """Public record of one submitted order. ``qty`` is the displayed (submitted) size and never
    changes; ``filled``/``fills`` grow as it trades; ``end`` is its cancel time. ``owner`` and
    ``parent`` are private: they are never read by the HBL belief computation."""

    __slots__ = ("oid", "owner", "side", "price", "qty", "time", "filled", "fills", "end", "seq", "parent")

    def __init__(self, oid, owner, side, price, qty, time, parent=None):
        self.oid = oid
        self.owner = owner
        self.side = side
        self.price = float(price)
        self.qty = int(qty)
        self.time = int(time)
        self.filled = 0
        self.fills: List[Tuple[int, int, float]] = []  # (time, qty, price)
        self.end: Optional[int] = None
        self.seq = -1
        self.parent = parent

    @property
    def rem(self) -> int:
        return self.qty - self.filled


class Book:
    """Price-time priority limit order book with self-trade prevention."""

    def __init__(self):
        self.bids: List[Rec] = []
        self.asks: List[Rec] = []
        self._seq = 0

    def best(self, side: int, exclude=None) -> float:
        best = -math.inf if side == BUY else math.inf
        for r in (self.bids if side == BUY else self.asks):
            if exclude is not None and r.owner == exclude:
                continue
            if (side == BUY and r.price > best) or (side == SELL and r.price < best):
                best = r.price
        return best

    def _best_rec(self, side: int, owner):
        best = None
        for r in (self.bids if side == BUY else self.asks):
            if r.owner == owner:
                continue  # self-trade prevention: never match against one's own resting orders
            if best is None:
                best = r
            elif side == BUY and (r.price > best.price or (r.price == best.price and r.seq < best.seq)):
                best = r
            elif side == SELL and (r.price < best.price or (r.price == best.price and r.seq < best.seq)):
                best = r
        return best

    def match(self, rec: Rec, t: int):
        """Match an incoming order; returns the list of (resting, incoming, qty, price) executions."""
        out = []
        opp_side = SELL if rec.side == BUY else BUY
        opp = self.asks if rec.side == BUY else self.bids
        while rec.rem > 0:
            r = self._best_rec(opp_side, rec.owner)
            if r is None:
                break
            if (rec.side == BUY and r.price > rec.price) or (rec.side == SELL and r.price < rec.price):
                break
            q = min(rec.rem, r.rem)
            price = r.price  # trade at the resting order's price, level by level
            for x in (r, rec):
                x.filled += q
                x.fills.append((t, q, price))
            out.append((r, rec, q, price))
            if r.rem == 0:
                opp.remove(r)
        if rec.rem > 0:
            self._seq += 1
            rec.seq = self._seq
            (self.bids if rec.side == BUY else self.asks).append(rec)
        return out

    def cancel_owner(self, owner, t: int):
        for lst in (self.bids, self.asks):
            keep = []
            for r in lst:
                if r.owner == owner:
                    r.end = t
                else:
                    keep.append(r)
            lst[:] = keep


# ----------------------------------------------------------------------------- simulator
class _BookView:
    """Minimal stand-in for the upstream FourHeap used inside HBLAgent.determine_optimal_price."""

    class _Side:
        def __init__(self, book: Book, side: int):
            self.book, self.side = book, side

        def peek(self):
            return self.book.best(self.side)

        def peek_order(self):
            lst = self.book.bids if self.side == BUY else self.book.asks
            return lst[0] if lst else None

    def __init__(self, book: Book):
        self.buy_unmatched = self._Side(book, BUY)
        self.sell_unmatched = self._Side(book, SELL)


class _MarketView:
    def __init__(self, sim: "Sim"):
        self.sim = sim
        self.order_book = _BookView(sim.book)

    def get_time(self):
        return self.sim.t


class LedgerHBL(HBLAgent):
    """Upstream HBL pricing (determine_optimal_price) on top of the public ledger."""

    kind = "HBL"

    def __init__(self, agent_id, sim: "Sim", rng: random.Random, pv: PrivateValues):
        cfg = sim.cfg
        super().__init__(agent_id, _MarketView(sim), cfg.q_max, list(cfg.shade), cfg.hbl_L, cfg.pv_var,
                         cfg.lam_bg, pv=pv)
        self.sim = sim
        self.rng = rng
        self.n_hbl = 0
        self.n_zi = 0
        self.n_fail = 0
        self._w = None

    # -- information
    def estimate_fundamental(self):
        return self.sim.estimate(self.sim.t)

    def get_order_list(self):
        sim = self.sim
        boundary = min(min(b.time, s.time) for b, s, _ in sim.fill_pairs[-self.L:])
        recs = sim.window(boundary)
        self._prepare(recs)
        buys = [r for r in recs if r.side == BUY]
        sells = [r for r in recs if r.side == SELL]
        return recs, buys, sells

    def _prepare(self, recs: List[Rec]):
        now = self.sim.t
        g = self.grace_period
        side = np.fromiter((r.side for r in recs), dtype=np.int8, count=len(recs))
        price = np.fromiter((r.price for r in recs), dtype=np.float64, count=len(recs))
        qty = np.fromiter((r.qty for r in recs), dtype=np.float64, count=len(recs))
        rem = np.fromiter((r.rem for r in recs), dtype=np.float64, count=len(recs))
        alive = np.fromiter(((now if r.end is None else r.end) - r.time for r in recs), dtype=np.float64,
                            count=len(recs))
        # Upstream counts an order as rejected only if it never traded; with unit orders this equals
        # "unfilled remainder". For multi-unit orders we count the unfilled remainder.
        wrej = rem * np.minimum(1.0, alive / g)
        fb = [(p, q) for r in recs if r.side == BUY for (_, q, p) in r.fills]
        fs = [(p, q) for r in recs if r.side == SELL for (_, q, p) in r.fills]
        b, s = side == BUY, side == SELL
        self._w = dict(
            bp=price[b], bq=qty[b], brej=wrej[b], sp=price[s], sq=qty[s], srej=wrej[s],
            fbp=np.array([x[0] for x in fb]), fbq=np.array([x[1] for x in fb], dtype=np.float64),
            fsp=np.array([x[0] for x in fs]), fsq=np.array([x[1] for x in fs], dtype=np.float64),
        )

    def _acc_rej(self, p, side):
        w = self._w
        if side == BUY:
            acc = w["sq"][w["sp"] <= p].sum() + w["fbq"][w["fbp"] <= p].sum()   # AL + TBL
            rej = w["brej"][w["bp"] >= p].sum()                                  # RBG
        else:
            acc = w["bq"][w["bp"] >= p].sum() + w["fsq"][w["fsp"] >= p].sum()   # BG + TAG
            rej = w["srej"][w["sp"] <= p].sum()                                  # RAL
        return acc, rej

    def belief_function(self, p, side, orders=None):
        acc, rej = self._acc_rej(p, side)
        if acc == 0:
            return 0
        return acc / (acc + rej)

    def fast_belief_function(self, p, side, orders=None):
        acc, _ = self._acc_rej(p, side)
        return acc == 0

    def act(self, side: int, offset: float) -> float:
        sim = self.sim
        book = sim.book
        price = None
        used = False
        if len(sim.fill_pairs) >= self.L and book.bids and book.asks:
            try:
                price, _ = self.determine_optimal_price(side)
                used = True
            except Exception:
                self.n_fail += 1
                price = None
        if price is None or not math.isfinite(price):
            est = self.estimate_fundamental()
            price = est + self.pv.value_for_exchange(self.position, side) - side * offset
            used = False
        if used:
            self.n_hbl += 1
        else:
            self.n_zi += 1
        return float(price), used


class ZI:
    kind = "ZI"

    def __init__(self, agent_id, sim, rng, pv):
        self.agent_id, self.sim, self.rng, self.pv = agent_id, sim, rng, pv
        self.position = 0
        self.cash = 0.0

    def act(self, side: int, offset: float):
        est = self.sim.estimate(self.sim.t)
        return est + self.pv.value_for_exchange(self.position, side) - side * offset, False


def make_pv(q_max, pv_var, rng: np.random.Generator) -> PrivateValues:
    pv = PrivateValues.__new__(PrivateValues)
    pv.values = np.sort(rng.normal(0.0, math.sqrt(pv_var), size=2 * q_max))[::-1].copy()
    pv.offset = q_max
    pv.extra_buy = min(pv.values[-1], 0)
    pv.extra_sell = max(pv.values[0], 0)
    return pv


def _arrivals(rng: np.random.Generator, lam: float, T: int, start: int = 0, first_at_start=False):
    gaps = rng.geometric(lam, size=int((T - start) * lam * 3) + 50)
    times = start + np.cumsum(gaps) - (0 if first_at_start else 1)
    if first_at_start:
        times = np.concatenate([[start], times])
    return [int(x) for x in times[times < T]]


class Sim:
    def __init__(self, cfg: Config, seed: int):
        self.cfg, self.seed = cfg, seed
        ss = np.random.SeedSequence(seed)
        s_fund, s_agents, s_mm, s_sp, s_roles = ss.spawn(5)
        # fundamental
        rng = np.random.default_rng(s_fund)
        shocks = rng.normal(0.0, math.sqrt(cfg.shock_var), size=cfg.T + 1)
        f = np.empty(cfg.T + 1)
        f[0] = cfg.mean
        for t in range(1, cfg.T + 1):
            f[t] = max(0.0, cfg.kappa * cfg.mean + (1 - cfg.kappa) * f[t - 1] + shocks[t])
        self.fund = f
        rho = (1 - cfg.kappa) ** (cfg.T - np.arange(cfg.T + 1))
        self.est = (1 - rho) * cfg.mean + rho * f

        self.book = Book()
        self.ledger: List[Rec] = []
        self.ledger_times: List[int] = []
        self.fill_pairs: List[Tuple[Rec, Rec, int]] = []  # (buy rec, sell rec, time)
        self.trades: List[Tuple[int, float, int]] = []    # (time, price, qty)
        self.t = 0
        self._oid = 0
        self.pos: Dict = {}
        self.cash: Dict = {}

        roles = np.random.default_rng(s_roles)
        hbl_ids = set(roles.choice(cfg.n_bg, size=cfg.n_hbl, replace=False).tolist())
        self.agents = {}
        self.events: Dict[int, List] = {}
        for aid, child in enumerate(s_agents.spawn(cfg.n_bg)):
            nr = np.random.default_rng(child)
            pv = make_pv(cfg.q_max, cfg.pv_var, nr)
            arng = random.Random(int(nr.integers(2 ** 62)))
            for t in _arrivals(nr, cfg.lam_bg, cfg.T):
                self.events.setdefault(t, []).append(aid)
            self.agents[aid] = LedgerHBL(aid, self, arng, pv) if aid in hbl_ids else ZI(aid, self, arng, pv)
            self.pos[aid], self.cash[aid] = 0, 0.0
        if cfg.mm:
            for t in _arrivals(np.random.default_rng(s_mm), cfg.lam_mm, cfg.T):
                self.events.setdefault(t, []).append(MM)
            self.pos[MM], self.cash[MM] = 0, 0.0
        # spoofer arrivals are drawn even when it is absent, so every arm shares the same streams
        sp_times = _arrivals(np.random.default_rng(s_sp), cfg.lam_sp, cfg.T, start=cfg.sp_start,
                             first_at_start=True)
        # The spoofer's arrivals are scheduled in every arm (and skipped when it is absent) so that the
        # within-step shuffle of the other traders is identical across arms.
        for t in sp_times:
            self.events.setdefault(t, []).append(SPOOFER)
        self.pos[SPOOFER], self.cash[SPOOFER] = 0, 0.0
        self.order_seed = seed

        # spoofer state
        self.sp_parent_rem = 0
        self.sp_parent_price = None
        self.sp_parent_id = 0
        self.sp_stats = dict(arrivals=0, sold=0, bought=0, spoof_fills=0, spoof_orders=0, children=0,
                             capped=0, max_visible=0, to_HBL=0, to_ZI=0, to_MM=0,
                             other_sell_HBL=0, other_sell_ZI=0, other_sell_MM=0)
        self.diag: List[tuple] = []

    # -- helpers
    def _kind(self, owner) -> str:
        return "MM" if owner == MM else self.agents[owner].kind

    def estimate(self, t: int) -> float:
        return float(self.est[t])

    def window(self, boundary: int) -> List[Rec]:
        import bisect
        i = bisect.bisect_left(self.ledger_times, boundary)
        return self.ledger[i:]

    def submit(self, owner, side, price, qty, parent=None) -> Rec:
        self._oid += 1
        rec = Rec(self._oid, owner, side, price, qty, self.t, parent)
        self.ledger.append(rec)
        self.ledger_times.append(self.t)
        for resting, incoming, q, p in self.book.match(rec, self.t):
            buy, sell = (resting, incoming) if resting.side == BUY else (incoming, resting)
            self.fill_pairs.append((buy, sell, self.t))
            self.trades.append((self.t, p, q))
            for r in (buy, sell):
                self.pos[r.owner] += r.side * q
                self.cash[r.owner] -= r.side * q * p
                if r.owner == SPOOFER:
                    if r.side == SELL:
                        self.sp_stats["sold"] += q
                    else:
                        self.sp_stats["bought"] += q
                        self.sp_stats["spoof_fills"] += q
            if sell.owner == SPOOFER and sell.parent is not None:
                self.sp_parent_rem -= q
                self.sp_stats["to_" + self._kind(buy.owner)] += q
            elif sell.owner != SPOOFER and buy.owner != SPOOFER:
                # trades between other participants after the seller's entry, by seller kind
                if self.t >= self.cfg.sp_start:
                    self.sp_stats["other_sell_" + self._kind(sell.owner)] += q
        # replenish an exhausted iceberg child (new order id, new time stamp: back of the queue)
        self._replenish()
        return rec

    def _replenish(self):
        cfg = self.cfg
        if not (cfg.iceberg and self.sp_parent_rem > 0):
            return
        live = [r for r in self.book.asks if r.owner == SPOOFER and r.parent == self.sp_parent_id]
        if live:
            return
        q = min(cfg.peak, self.sp_parent_rem)
        self.sp_stats["children"] += 1
        self.submit(SPOOFER, SELL, self.sp_parent_price, q, parent=self.sp_parent_id)

    # -- agent actions
    def _bg(self, aid):
        agent = self.agents[aid]
        self.book.cancel_owner(aid, self.t)
        side = agent.rng.choice([BUY, SELL])
        offset = agent.rng.uniform(*self.cfg.shade)
        agent.position = self.pos[aid]
        price, used = agent.act(side, offset)
        if self.cfg.diag and agent.kind == "HBL" and used and self.t >= self.cfg.sp_start:
            self._record_diag(agent, side, price)
        self.submit(aid, side, price, 1)

    def _record_diag(self, agent: LedgerHBL, side, price):
        w = agent._w
        est = self.estimate(self.t)
        # share of the visible ask quantity in memory that belongs to the strategic seller
        recs = self.window(min(min(b.time, s.time) for b, s, _ in self.fill_pairs[-agent.L:]))
        sp_ask = sum(r.qty for r in recs if r.owner == SPOOFER and r.side == SELL)
        all_ask = float(w["sq"].sum())
        sp_bid = sum(r.qty for r in recs if r.owner == SPOOFER and r.side == BUY)
        all_bid = float(w["bq"].sum())
        self.diag.append((self.t, side, price - est, sp_ask, all_ask, sp_bid, all_bid))

    def _mm(self):
        cfg = self.cfg
        self.book.cancel_owner(MM, self.t)
        est = self.estimate(self.t)
        best_bid, best_ask = self.book.best(BUY), self.book.best(SELL)
        st = max(est + 0.5 * cfg.mm_omega, best_bid)
        bt = min(est - 0.5 * cfg.mm_omega, best_ask)
        for k in range(cfg.mm_K):
            self.submit(MM, BUY, bt - (k + 1) * cfg.mm_xi, 1)
            self.submit(MM, SELL, st + (k + 1) * cfg.mm_xi, 1)

    def _spoofer(self):
        cfg = self.cfg
        st = self.sp_stats
        st["arrivals"] += 1
        self.book.cancel_owner(SPOOFER, self.t)
        self.sp_parent_rem = 0
        # genuine leg first (as in the upstream MMSP wrapper), then the spoof order
        price = self.estimate(self.t) + cfg.sell_offset
        self.sp_parent_id += 1
        self.sp_parent_price = price
        if cfg.iceberg:
            self.sp_parent_rem = cfg.sell_qty
            st["children"] += 1
            self.submit(SPOOFER, SELL, price, min(cfg.peak, cfg.sell_qty), parent=self.sp_parent_id)
        else:
            self.sp_parent_rem = cfg.sell_qty
            self.submit(SPOOFER, SELL, price, cfg.sell_qty, parent=self.sp_parent_id)
        if cfg.spoof:
            bid = self.book.best(BUY, exclude=SPOOFER)
            if math.isfinite(bid):
                sp_price = bid - cfg.tick
                own_ask = min((r.price for r in self.book.asks if r.owner == SPOOFER), default=math.inf)
                if sp_price >= own_ask:  # never post a spoof bid that would lock against its own ask
                    sp_price = own_ask - cfg.tick
                    st["capped"] += 1
                st["spoof_orders"] += 1
                self.submit(SPOOFER, BUY, sp_price, cfg.spoof_qty)
        vis = sum(r.rem for r in self.book.asks if r.owner == SPOOFER)
        st["max_visible"] = max(st["max_visible"], vis)

    # -- main loop
    def run(self) -> dict:
        cfg = self.cfg
        mids = []
        grid = list(range(cfg.sp_start, cfg.T + 1, cfg.mid_grid))
        gi = 0
        for t in sorted(self.events):
            while gi < len(grid) and grid[gi] < t:
                mids.append(self._mid())
                gi += 1
            self.t = t
            who = list(self.events[t])
            random.Random(self.order_seed * 1_000_003 + t).shuffle(who)
            for a in who:
                if a == MM:
                    self._mm()
                elif a == SPOOFER:
                    if cfg.spoofer:
                        self._spoofer()
                else:
                    self._bg(a)
        while gi < len(grid):
            mids.append(self._mid())
            gi += 1
        return self._summary(np.array(mids, dtype=float), np.array(grid, dtype=float))

    def _mid(self):
        b, a = self.book.best(BUY, exclude=SPOOFER), self.book.best(SELL, exclude=SPOOFER)
        if math.isfinite(b) and math.isfinite(a):
            return 0.5 * (a + b)
        return math.nan

    def _summary(self, mids, grid):
        cfg = self.cfg
        final = float(self.fund[cfg.T])
        surplus = {"ZI": [], "HBL": []}
        hbl_used = hbl_tot = fails = 0
        for aid, ag in self.agents.items():
            pos = self.pos[aid]
            surplus[ag.kind].append(ag.pv.value_at_position(pos) + self.cash[aid] + pos * final)
            if ag.kind == "HBL":
                hbl_used += ag.n_hbl
                hbl_tot += ag.n_hbl + ag.n_zi
                fails += ag.n_fail
        sp = self.pos[SPOOFER] * final + self.cash[SPOOFER]
        mm = self.pos[MM] * final + self.cash[MM] if cfg.mm else 0.0
        tr = np.array(self.trades, dtype=float).reshape(-1, 3)
        post = tr[tr[:, 0] >= cfg.sp_start] if len(tr) else tr
        est_grid = self.est[grid.astype(int)]
        diag = np.array(self.diag, dtype=float).reshape(-1, 7)
        out = dict(
            seed=self.seed,
            sp_surplus=sp, sp_sold=self.sp_stats["sold"], sp_bought=self.sp_stats["bought"],
            sp_position=self.pos[SPOOFER], sp_children=self.sp_stats["children"],
            sp_capped=self.sp_stats["capped"], sp_max_visible=self.sp_stats["max_visible"],
            **{f"sp_{k}": self.sp_stats[k] for k in ("to_HBL", "to_ZI", "to_MM", "other_sell_HBL",
                                                      "other_sell_ZI", "other_sell_MM")},
            hbl_mean=float(np.mean(surplus["HBL"])) if surplus["HBL"] else math.nan,
            zi_mean=float(np.mean(surplus["ZI"])) if surplus["ZI"] else math.nan,
            mm_surplus=mm, mm_position=self.pos.get(MM, 0),
            total_bg=float(sum(sum(v) for v in surplus.values())),
            n_trades=int(tr[:, 2].sum()) if len(tr) else 0,
            n_trades_post=int(post[:, 2].sum()) if len(post) else 0,
            vwap_dev_post=float(np.sum((post[:, 1] - self.est[post[:, 0].astype(int)]) * post[:, 2]) / post[:, 2].sum())
            if len(post) else math.nan,
            mid_dev=float(np.nanmean(mids - est_grid)) if np.isfinite(mids).any() else math.nan,
            mids=mids, final_fund=final,
            hbl_share=hbl_used / hbl_tot if hbl_tot else math.nan, hbl_fail=fails,
            imbalance_pos=sum(self.pos.values()), imbalance_cash=sum(self.cash.values()),
            diag_n=len(diag),
            diag_buy_dev=float(diag[diag[:, 1] == BUY, 2].mean()) if (diag[:, 1] == BUY).any() else math.nan,
            diag_sell_dev=float(diag[diag[:, 1] == SELL, 2].mean()) if (diag[:, 1] == SELL).any() else math.nan,
            diag_sp_ask_share=float(np.mean(diag[:, 3] / np.maximum(diag[:, 4], 1))) if len(diag) else math.nan,
            diag_sp_bid_share=float(np.mean(diag[:, 5] / np.maximum(diag[:, 6], 1))) if len(diag) else math.nan,
        )
        return out


def run(args):
    cfg, seed = args
    return Sim(cfg, seed).run()
