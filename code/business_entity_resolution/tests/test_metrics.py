from src.ber.model import macro_f05

def test_macro_includes_candidate_miss():
    assert macro_f05({1:{"a"},2:{"b"}}, {1:{"a"}}, query_ids=[1,2]) == .5
def test_empty_query_is_perfect():
    assert macro_f05({1:set(),2:set()}, {}, query_ids=[1,2]) == 1.0
def test_singleton_precision_penalty():
    score=macro_f05({"q":{"right"}}, {"q":{"wrong"}}, query_ids=["q"])
    assert score == 0.0
