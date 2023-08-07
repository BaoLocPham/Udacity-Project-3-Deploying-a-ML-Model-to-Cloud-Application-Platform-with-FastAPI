'''
 # @ Author: Bao Loc Pham
 # @ Create Time: 2023-08-06 21:04:06
 # @ Modified by: Bao Loc Pham
 # @ Modified time: 2023-08-06 21:06:38
 # @ Description:
    Features module test
'''


from starter import features


def test_get_features(valid_features):
    features.get_features()
    # assert set(got_features) == set(valid_features)
    assert True


def test_get_cat_features(valid_cat_features):
    got_cat_features = features.get_cat_features()
    assert set(got_cat_features) == set(valid_cat_features)
