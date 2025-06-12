from fxutil import scinum


def test_scinum_trailing_zeros():
    x = 340

    assert scinum(340, no_trailing_zeros=True) == "340\,"
    assert scinum(340, no_trailing_zeros=False) == "340.00\,"
