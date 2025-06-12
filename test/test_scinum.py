from fxutil import scinum


def test_scinum_trailing_zeros():
    assert scinum(340, no_trailing_zeros=True) == "340\,"
    assert scinum(340, no_trailing_zeros=False) == "340.00\,"
    assert scinum(340, no_trailing_zeros=False, ndigits=4) == "340.0000\,"
    assert (
        scinum(34000000, no_trailing_zeros=False, ndigits=4, force_mode="f")
        == "34\,000\,000.0000\,"
    )
    assert (
        scinum(34000000, no_trailing_zeros=False, ndigits=4) == r"3.4000\times 10^{7}\,"
    )
    assert scinum(34000000, no_trailing_zeros=True, ndigits=4) == r"3.4\times 10^{7}\,"
