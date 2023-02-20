import pytest
from dataset import Dataset
import values as v
import utils as ut


@pytest.fixture
def t_dataset():
    dataset = Dataset(v.PT_BR_DICTIONARY_FILE, item_addition_ratio=0.1)
    return dataset

@pytest.fixture
def t_dictionary():
    d = {"aalto":"'a w t u", "abba":"'a b 6", "abbade":"6 b 'a dZ i", "abbamonte":"6 b 6 m 'o~ tS i", 
    "abbate":"6 b 'a tS i", "abbondanza":"6 b o~ d '6~ z 6"}
    return d

@pytest.fixture
def t_list():
    l = [["aalto","'a w t u"], ["abba","'a b 6"], ["abbade","6 b 'a dZ i"], ["abbamonte","6 b 6 m 'o~ tS i"], 
    ["abbate","6 b 'a tS i"], ["abbondanza","6 b o~ d '6~ z 6"]]
    return l

def test_get_pairs(t_dataset, t_dictionary, t_list):
    dict_list = t_dataset.get_pairs(t_dictionary)
    assert dict_list == t_list
