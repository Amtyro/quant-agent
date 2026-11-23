"""Correctness checks for the A1 simulator.

Run from a1_iceberg_spoofing with PyMarketSim on PYTHONPATH: ``python -m pytest``.
"""
import math
from dataclasses import replace

from marketsim.agent.hbl_agent import HBLAgent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import MatchedOrder, Order

from experiments.a1sim import ENVS, SPOOFER, Book, Config, LedgerHBL, Rec, Sim


def _rec(oid, owner, side, price, qty, t=0):
    return Rec(oid, owner, side, price, qty, t)


def test_price_time_priority_and_level_pricing():
    b = Book()
    b.match(_rec(1, "a", SELL, 101, 1), 0)
    b.match(_rec(2, "b", SELL, 100, 1), 0)
    b.match(_rec(3, "c", SELL, 100, 1), 1)   # same price, later: behind order 2
    ex = b.match(_rec(4, "d", BUY, 101, 3), 2)
    assert [(r.oid, q, p) for r, _, q, p in ex] == [(2, 1, 100.0), (3, 1, 100.0), (1, 1, 101.0)]
    assert not b.asks and not b.bids


def test_partial_fill_and_self_trade_prevention():
    b = Book()
    big = _rec(1, "x", SELL, 100, 5)
    b.match(big, 0)
    b.match(_rec(2, "x", BUY, 100, 1), 1)          # own order: must not trade, rests
    assert big.filled == 0 and len(b.bids) == 1
    b.match(_rec(3, "y", BUY, 100, 2), 2)
    assert big.filled == 2 and big.rem == 3 and big.qty == 5   # displayed size in the record is unchanged
    b.cancel_owner("x", 3)
    assert big.end == 3 and not b.asks and not b.bids


def _cfg(**kw):
    xi, om = ENVS["B2"]
    base = Config(mm_xi=xi, mm_omega=om, spoofer=True, spoof=True, sell_qty=10, iceberg=True, peak=1,
                  sell_offset=50, T=4000)
    return replace(base, **kw)


def test_conservation_and_iceberg_invariants():
    for ice in (False, True):
        sim = Sim(_cfg(iceberg=ice), seed=3)
        out = sim.run()
        assert out["imbalance_pos"] == 0
        assert abs(out["imbalance_cash"]) < 1e-6
        if ice:
            assert out["sp_max_visible"] <= 1
        # no parent sells more than G units
        sold = {}
        for r in sim.ledger:
            if r.owner == SPOOFER and r.side == SELL:
                sold[r.parent] = sold.get(r.parent, 0) + r.filled
        assert max(sold.values()) <= 10


def test_iceberg_irrelevant_without_book_learners():
    """With no HBL traders nobody reads displayed size, so VIS and ICE must give the same market."""
    a = Sim(_cfg(n_zi=24, n_hbl=0, iceberg=False), seed=5).run()
    b = Sim(_cfg(n_zi=24, n_hbl=0, iceberg=True), seed=5).run()
    for k in ("sp_surplus", "sp_sold", "zi_mean", "mm_surplus", "n_trades"):
        assert a[k] == b[k], k


class _FakeMarket:
    def __init__(self, t, matched):
        self._t, self.matched_orders = t, matched

    def get_time(self):
        return self._t


def test_belief_matches_upstream_on_unit_orders():
    """No MM, no spoofer: every order is a unit order, so the ledger belief must equal the upstream one."""
    cfg = replace(_cfg(), mm=False, spoofer=False, spoof=False, T=6000, lam_bg=0.005)
    sim = Sim(cfg, seed=11)
    checked = [0]
    orig = LedgerHBL.act

    def act(self, side, offset):
        price, used = orig(self, side, offset)
        if used and checked[0] < 25:
            recs, _, _ = self.get_order_list()
            orders = [Order(price=r.price, order_type=r.side, quantity=r.qty, agent_id=r.owner, time=r.time,
                            order_id=r.oid) for r in recs]
            by_id = {o.order_id: o for o in orders}
            matched = []
            for bu, se, t in sim.fill_pairs:
                for r in (bu, se):
                    matched.append(MatchedOrder(r.fills[0][2], t, by_id.get(r.oid, Order(
                        price=r.price, order_type=r.side, quantity=r.qty, agent_id=r.owner, time=r.time,
                        order_id=r.oid))))
            fake = type("F", (), {})()
            fake.market = _FakeMarket(sim.t, matched)
            fake.grace_period = self.grace_period
            prices = sorted({o.price for o in orders})
            for p in prices[:: max(1, len(prices) // 15)]:
                for s in (BUY, SELL):
                    up = HBLAgent.belief_function(fake, p, s, orders)
                    mine = self.belief_function(p, s)
                    assert math.isclose(up, mine, rel_tol=1e-9, abs_tol=1e-12), (p, s, up, mine)
            checked[0] += 1
        return price, used

    LedgerHBL.act = act
    try:
        sim.run()
    finally:
        LedgerHBL.act = orig
    assert checked[0] >= 10


def test_arms_are_paired():
    """A seller nobody can see or trade with must leave the rest of the market unchanged."""
    base = Sim(_cfg(n_zi=24, n_hbl=0, spoofer=False), seed=7).run()
    ghost = Sim(_cfg(n_zi=24, n_hbl=0, spoofer=True, spoof=False, sell_offset=1e7), seed=7).run()
    for k in ("zi_mean", "mm_surplus", "n_trades", "mid_dev"):
        assert base[k] == ghost[k], k


def test_fill_time_valuation_identity():
    """Fill-time surplus = sum of side*q*(r~_t - p) over the ledger's fills; differs from the terminal-value surplus
    only by sum side*q*(r_T - r~_t); and the valuations net to zero across all traders."""
    sim = Sim(_cfg(), seed=5)
    out = sim.run()
    final = out["final_fund"]
    nr = noise = 0.0
    for r in sim.ledger:
        if r.owner != SPOOFER:
            continue
        for (t, q, p) in r.fills:
            nr += r.side * q * (sim.est[t] - p)
            noise += r.side * q * (final - sim.est[t])
    assert out["sp_bought"] + out["sp_sold"] > 0
    assert abs(out["sp_surplus_nr"] - nr) < 1e-6
    assert abs(out["sp_surplus"] - out["sp_surplus_nr"] - noise) < 1e-6
    assert abs(sum(sim.mark.values())) < 1e-6
