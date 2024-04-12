from collections import defaultdict
from typing import Any, Optional

def nested_dict(n: int = 2, type_: Any = list, obj: Optional[dict] = None):
    
    """
    Creates nested dicts (`defaultdict`s of any depth with any type).
    Additionally, can take an arbitrarily deep nested dictionary and
    turn it into its `defaultdict` correspondence.

    :param n: Number of depth levels.
    :param type_: Type of the nested dict
    :param obj: The nested dict to turn into a `defaultdict`.
    :return:
    """
    if n <= 0:  # Base case
        return obj if obj else type_()

    if obj:  # Recursive case
        return defaultdict(lambda: type_(), {k: nested_dict(n=n - 1, type_=type_, obj=v) for k, v in obj.items()})

    return defaultdict(lambda: nested_dict(n=n - 1, type_=type_))

def flatten(lst: list[list]) -> list:
    """
    Flattens a list of lists to a single list.
    """
    return [item for sublist in lst for item in sublist]
