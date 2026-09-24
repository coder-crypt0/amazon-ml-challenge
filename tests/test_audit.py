from pathlib import Path
from ber.audit import audit_outputs

def _write(p, text):
    p.write_text(text, encoding="utf-8")

def test_audit_accepts_france_and_rejects_subset_violation(tmp_path):
    td=tmp_path/'test'; td.mkdir()
    _write(td/'test_source1.tsv', 'entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tA\tX\tFrance\nS1-2\tB\tY\tUS\n')
    for n, ids in [('test_source2.tsv',['S2-1']),('test_source3.tsv',['S3-1'])]:
        _write(td/n, 'entity_id\tbusiness_name\tbusiness_address\tcountry\n' + ''.join(f'{x}\tN\tA\tFrance\n' for x in ids))
    m=tmp_path/'m.tsv'; c=tmp_path/'c.tsv'
    _write(m, 'source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\n')
    _write(c, 'source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1\nS1-2\t\n')
    assert audit_outputs(m,c,td)['ok']
    _write(c, 'source1_entity_id\tcandidate_entity_ids\nS1-1\t\nS1-2\t\n')
    assert not audit_outputs(m,c,td)['ok']
