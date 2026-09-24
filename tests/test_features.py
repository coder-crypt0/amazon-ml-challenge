import numpy as np
from ber.features import prepare_record, pair_features, FEATURE_NAMES


def record(name="Café École SARL",address="23 Rue du Centre",country="France",id="S1-1"):
    return dict(entity_id=id,business_name=name,business_address=address,country=country)


def test_features_preserve_unicode_and_ignore_ids():
    left=record(name="दिल्ली कैफे")
    prepared=prepare_record(left)
    assert prepared["_name"] and prepared["_unicode"]
    a=pair_features(left,record(name="दिल्ली कैफे",id="S2-200"))
    b=pair_features(record(name="दिल्ली कैफे",id="S1-999"),record(name="दिल्ली कैफे",id="S3-7"))
    assert len(a)==len(FEATURE_NAMES)
    np.testing.assert_array_equal(a,b)
    assert np.isfinite(a).all()


def test_missing_address_is_not_exact_match():
    x=dict(zip(FEATURE_NAMES,pair_features(record(address=""),record(address=""))))
    assert x["address_equal"]==0
    assert x["left_address_missing"]==1
    assert x["right_address_missing"]==1
