from fxutil.typing import Combi, parse_combi_args


def test_combi_args():
    @parse_combi_args
    def func(a: Combi[int], b: Combi[str] | None = None):
        return {"a": a, "b": b}

    # breakpoint()
    res = func(1, "foo")

    assert res["a"] == (1,)
    assert res["b"] == ("foo",)

    @parse_combi_args(exceptions=["foo"])
    def func(a: Combi[int], b: Combi[str] | None = None):
        return {"a": a, "b": b}

    res = func(a=1, b="foo")

    assert res["a"] == (1,)
    assert res["b"] == "foo"

    res = func(a=1)
    assert res["a"] == (1,)
    assert res["b"] is None
