"""
Type hints for spiceypy as it isn't typed.
"""
from typing import Iterable, Tuple, Union

from numpy import ndarray

def spkpos(
    targ: str, et: Union[float, ndarray], ref: str, abcorr: str, obs: str
) -> Union[Tuple[ndarray, float], Tuple[ndarray, ndarray]]: ...


def furnsh(path: Union[str, Iterable[str]]) -> None: ...


def ktotal(kind: str) -> int: ...


def unload(filename: Union[str, Iterable[str]]) -> None: ...


def spkezr(
    targ: str, et: Union[ndarray, float], ref: str, abcorr: str, obs: str
) -> Union[Tuple[ndarray, float], Tuple[Iterable[ndarray], Iterable[float]]]: ...


def str2et(time: str) -> float: ...


def bodvrd(bodynm: str, item: str, maxn: int) -> Tuple[int, ndarray]: ...