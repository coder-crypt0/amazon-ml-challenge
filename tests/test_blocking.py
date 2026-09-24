from src.ber.text import normalize, canonical
from src.ber.blocking import blocking_keys, BlockingIndex

def test_unicode_normalization():
    assert normalize("Café, दिल्ली") == "cafe दिल्ली"
    assert canonical("Acme Ltd") == "acme"

def test_country_prefixed_and_stable_keys():
    a = blocking_keys({"name": "Acme Ltd", "address": "12 Main Road"}, "France")
    b = blocking_keys({"name": "Acme Ltd", "address": "12 Main Road"}, "France")
    assert a == b and all(k.startswith("france|") for k in a)

def test_disk_index_caps_postings(tmp_path):
    records = [{"name": "Acme Ltd", "address": "1 Main St"}] * 20
    ix = BlockingIndex.build(records, tmp_path / "idx", country="FR", max_postings=3, chunk_size=4)
    assert len(ix.query(records[0], "FR")) == 0
