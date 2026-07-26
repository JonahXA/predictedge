import pandas as pd

from predictedge.export import _daily, _reliability


def test_reliability_devigs_market_within_event():
    """Market mids are normalized per event before pooling, so a book
    quoting 55/55 (10% vig) is scored as a 50/50 forecast, not 55/55."""
    n = 20
    bins = pd.DataFrame({
        "event_ticker": [f"E{i // 2}" for i in range(2 * n)],
        "p_model": [0.5] * 2 * n,
        "mid": [0.55] * 2 * n,  # each event's book sums to 1.10
        "result": ["yes", "no"] * n,
    })
    market = [r for r in _reliability(bins) if r["series"] == "market"]
    assert len(market) == 1
    assert market[0]["predicted"] == 0.5  # de-vigged, not 0.55
    assert market[0]["observed"] == 0.5


def test_reliability_bins_and_counts():
    n = 40
    bins = pd.DataFrame({
        "event_ticker": [f"E{i}" for i in range(n)],
        "p_model": [0.85] * n,
        "mid": [1.0] * n,
        "result": ["yes"] * 30 + ["no"] * 10,
    })
    rows = [r for r in _reliability(bins) if r["series"] == "model"]
    assert len(rows) == 1
    assert rows[0]["n"] == n
    assert rows[0]["predicted"] == 0.85
    assert rows[0]["observed"] == 0.75  # 30/40 resolved yes


def test_daily_averages_across_cities():
    events = pd.DataFrame({
        "date": ["2026-05-17", "2026-05-17", "2026-05-18"],
        "brier_model": [0.8, 0.6, 0.5],
        "brier_market": [0.4, 0.2, 0.1],
        "event_ticker": ["A", "B", "C"],
    })
    rows = _daily(events)
    assert rows[0] == {"date": "2026-05-17", "model_brier": 0.7, "market_brier": 0.3, "n": 2}
    assert rows[1]["n"] == 1


def test_drop_duplicate_members_keeps_first():
    """Alias model ids return identical series; keeping both would
    double-weight that model and understate the ensemble spread."""
    import pandas as pd

    from predictedge.weather import drop_duplicate_members

    df = pd.DataFrame({
        "gfs_seamless": [80.0, 81.0],
        "gfs_hrrr": [80.0, 81.0],  # alias of gfs
        "icon_seamless": [78.0, 79.0],
    })
    out = drop_duplicate_members(df)
    assert list(out.columns) == ["gfs_seamless", "icon_seamless"]
