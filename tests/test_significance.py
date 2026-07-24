import numpy as np

from predictedge.significance import cluster_bootstrap, diebold_mariano


def test_dm_detects_a_real_difference():
    rng = np.random.default_rng(1)
    d = rng.normal(0.5, 1.0, size=200)
    stat, p = diebold_mariano(d)
    assert stat > 0 and p < 0.001


def test_dm_accepts_null():
    rng = np.random.default_rng(2)
    d = rng.normal(0.0, 1.0, size=200)
    _, p = diebold_mariano(d)
    assert p > 0.05


def test_cluster_bootstrap_ci_covers_truth():
    rng = np.random.default_rng(3)
    clusters = np.repeat(np.arange(50), 6)  # 50 dates x 6 cities
    d = rng.normal(0.3, 1.0, size=300)
    lo, hi, p = cluster_bootstrap(d, clusters)
    assert lo < 0.3 < hi
    assert p < 0.05


def test_cluster_bootstrap_wider_than_iid_when_clustered():
    """Perfectly correlated within-date scores must widen the CI vs
    treating events as independent."""
    rng = np.random.default_rng(4)
    per_date = rng.normal(0.0, 1.0, size=40)
    d = np.repeat(per_date, 6)  # 6 identical events per date
    clusters = np.repeat(np.arange(40), 6)
    lo_cl, hi_cl, _ = cluster_bootstrap(d, clusters)
    lo_iid, hi_iid, _ = cluster_bootstrap(d, np.arange(len(d)))
    assert (hi_cl - lo_cl) > (hi_iid - lo_iid) * 1.5
