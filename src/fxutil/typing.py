import inspect

import functools as ft

from typing import Annotated, Callable, Iterable, TypeVar, Union, get_args, get_origin

T = TypeVar("T")


class _CombiTag:
    pass


Combi = Annotated[Union[T, Iterable[T], str], _CombiTag]


def is_combi_ann(ann) -> bool:
    return get_origin(ann) is Annotated and _CombiTag in get_args(ann)[1:]


def parse_combi_argument(arg, exceptions=None):
    if arg in (exceptions or []):
        return arg
    else:
        if isinstance(arg, Iterable) and not isinstance(arg, str):
            return arg
        else:
            return (arg,)


def parse_combi_args(func: Callable | None = None, exceptions: list = None):
    def _decorate(func):
        sig = inspect.signature(func)

        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for name, value in bound.arguments.items():
                if is_combi_ann(sig.parameters[name].annotation):
                    # print(f"{name} is Combi")
                    new_val = parse_combi_argument(value, exceptions)
                    # print(f"Replacing {value} with {new_val}")
                    bound.arguments[name] = new_val

            return func(*bound.args, **bound.kwargs)

        return wrapper

    if func:
        return _decorate(func)

    return _decorate
