import pytest
import pandas as pd

from src.models import hybrid as H

def test_interweave():
    df = pd.DataFrame({
        'val': [0, 2, 4, 1, 2, 3],
        'grp': ['A', 'A', 'A', 'B', 'B', 'B'],
    })

    result = H.interweave(df, by='grp')
    expected = ['A', 'B'] * 3

    assert result['grp'].tolist() == expected

# TODO: All the merge methods should satisfy this test
# and the next one
@pytest.mark.parametrize('merge', [H.merge_apply_avg, H.merge_apply_naive, H.merge_apply_prioritize_common])
def test_merge_apply_all_common(merge):
    df = pd.DataFrame({
        'itemID': ['0', '1', '2', '3', '0', '1', '2', '3'],
        'rec':    ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'prediction': pd.NA,
    })

    # res = H.merge_apply_avg(df, top_k=4)
    res = merge(df, top_k=4)
    
    assert res['rec'].tolist() == ['both'] * 4

@pytest.mark.parametrize('merge', [H.merge_apply_avg, H.merge_apply_naive, H.merge_apply_prioritize_common])
def test_merge_apply_none_common(merge):
    df = pd.DataFrame({
        'itemID': ['0', '1', '2', '3', '4', '5', '6', '7'],
        'rec':    ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'prediction': pd.NA,
    })

    # res = H.merge_apply_avg(df, top_k=4)
    res = merge(df, top_k=4)

    assert all(res['rec'] != 'both')
    assert res['rec'].tolist() == ['A', 'B', 'A', 'B']
    assert res.index.tolist() == ['0', '4', '1', '5']

def test_merge_apply_priority():
    df = pd.DataFrame({
        'itemID': ['a', 'b', 'c', 'd', 'd', 'e', 'f', 'g'],
        'rec':    ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'prediction': pd.NA,
    })

def test_merge_apply_naive():
    df = pd.DataFrame({
        'itemID': ['a', 'b', 'c', 'd', 'd', 'e', 'f', 'g'],
        'rec':    ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'prediction': pd.NA,
    })
    
    res = H.merge_apply_naive(df, top_k=4)

    assert res['rec'].tolist() == ['A', 'both', 'A', 'B']
    assert res.index.tolist() == ["a", "d", "b", "e"]

def test_merge_apply_avg():
    df = pd.DataFrame({
        'itemID': ['a', 'b', 'c', 'd', 'd', 'e', 'f', 'g'],
        'rec':    ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'prediction': pd.NA,
    })
    
    res = H.merge_apply_avg(df, top_k=4)

    assert res['rec'].tolist() == ['A', 'A', 'B', 'both']
    assert res.index.tolist() == ["a", "b", "e", "d"]

def test_merge_apply_avg_all():
    df = pd.DataFrame({
        'itemID': ['a', 'b', 'c', 'd', 'd', 'e', 'f', 'g'],
        'rec':    ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'prediction': pd.NA,
    })
    
    res = H.merge_apply_avg_all(df, top_k=4)

    assert res['rec'].tolist() == ['both', 'A', 'A', 'B']
    assert res.index.tolist() == ["d", "a", "b", "e"]

def test_merge_apply_avg_all2():
    df = pd.DataFrame({
        'itemID': [
            'A', 'B', 'C', 'D', 'E', 'F',
            'B', 'E', 'H', 'I', 'J', 'F',
        ],
        'rec':    ['A']*6 + ['B']*6,
        'prediction': pd.NA,
    })
    
    res = H.merge_apply_avg_all(df, top_k=6)

    assert res.index.tolist() == ["B", "E", "A", "F", "C", "H"]
    assert res['rec'].tolist() == ['both', 'both', 'A', 'both', 'A', 'B']

