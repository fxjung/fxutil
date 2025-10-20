import functools as ft
import inspect
from typing import Annotated, Callable, Iterable, TypeVar, Union, get_args, get_origin

T = TypeVar("T")


class _CombiTag:
    pass


Combi = Annotated[Union[T, Iterable[T], str], _CombiTag]


def is_combi_ann(ann) -> bool:
    """
    Check if the given annotation is a Combi annotation.

    Parameters
    ----------
    ann
        Annotation to check.

    Returns
    -------
    True if the annotation is believed to be a Combi annotation, False otherwise.

    """
    if ann is None:
        # Not annotated
        return False
    elif get_origin(ann) is None:
        # Not a generic type
        return False
    elif get_origin(ann) is Union:
        # Union type: check all args
        return any(is_combi_ann(a) for a in get_args(ann))
    else:
        # Check if Annotated with _CombiTag
        return get_origin(ann) is Annotated and _CombiTag in get_args(ann)[1:]


def parse_combi_argument(arg, exceptions=None):
    if arg is None or arg in (exceptions or []):
        # Received None or value from the exceptions list, return as is
        return arg
    else:
        if isinstance(arg, Iterable) and not isinstance(arg, str):
            # Already an iterable (but not a string), return as is
            return arg
        else:
            # Otherwise: wrap as a one-tuple
            return (arg,)


def parse_combi_args(func: Callable | None = None, exceptions: list = None):
    """
    Decorator to parse Combi arguments of a function.

    Usage example:

    ```python
    @parse_combi_args
    def my_function(param: Combi[int] | None):
        ...
    ```

    This will ensure that `param` is always treated as an iterable of integers (or None)
    when `my_function` is called, even if a single integer is passed.


    Parameters
    ----------
    func
    exceptions

    Returns
    -------

    """

    def _decorate(func):
        sig = inspect.signature(func)

        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for name, value in bound.arguments.items():
                if is_combi_ann(sig.parameters[name].annotation):
                    # Possibly use
                    # get_type_hints(func, include_extras=True)
                    # instead?
                    # Note that a very weird bug occurs with the inspect annotation
                    # return value (produces a string instead of an annotation object)
                    # if typing is imported from __future__.
                    # See also <https://stackoverflow.com/q/72510518/23551601>

                    # print(f"{name} is Combi")
                    new_val = parse_combi_argument(value, exceptions)
                    # print(f"Replacing {value} with {new_val}")
                    bound.arguments[name] = new_val

            return func(*bound.args, **bound.kwargs)

        return wrapper

    if func:
        return _decorate(func)

    return _decorate
