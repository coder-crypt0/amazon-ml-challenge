import itertools
import numpy as np
from ber.decision import expected_f05_prefix, resolve_exclusive


def test_exact_expected_utility_includes_unselected_truth_and_singletons():
    probabilities=np.array([.8,.5,.15])
    brute=np.zeros(4)
    for bits in itertools.product((0,1),repeat=3):
        mass=np.prod([p if bit else 1-p for p,bit in zip(probabilities,bits)])
        total=sum(bits)
        for k in range(4):
            value=1.25*sum(bits[:k])/(k+.25*total) if k or total else 1.
            brute[k]+=mass*value
    np.testing.assert_allclose(expected_f05_prefix(probabilities),brute)
    np.testing.assert_allclose(expected_f05_prefix([]),[1.])


def test_tied_exclusive_edges_abstain():
    assert resolve_exclusive([[0,7],[1,7]],[.9,.9]).tolist()==[]
    assert resolve_exclusive([[0,7],[1,7]],[.95,.9],margin=.01).tolist()==[0]


def test_assemble_policy_selects_exactly_what_the_tuner_scored():
    from ber import policy as tuner
    from ber.predict import select_matches
    rng = np.random.default_rng(3)
    queries, pairs, probs = [], [], []
    for qi in range(200):
        queries.append({"truth": ["x"] * int(rng.integers(0, 4)), "country": "India" if qi % 2 else "US"})
        for j in range(int(rng.integers(0, 12))):
            pairs.append((qi, j)); probs.append(float(rng.random()))
    probs, pairs = np.asarray(probs), np.asarray(pairs)
    ev = tuner.Evaluator(probs, pairs, np.zeros(len(probs), bool), queries)
    for rule in ({"t": .6, "t1": .6, "r": 0.}, {"t": .7, "t1": .2, "r": 0.}, {"t": .5, "t1": .1, "r": .5}):
        expected = ev.select(rule["t"], rule["t1"], rule["r"])
        actual = np.concatenate([select_matches(probs[pairs[:, 0] == qi], rule) for qi in range(len(queries))])
        assert np.array_equal(expected, actual), rule
