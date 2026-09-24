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
