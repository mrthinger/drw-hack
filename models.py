from numpy import ndarray
from datetime import datetime
from typing import TypedDict

class Observation(TypedDict):
    id: int | None
    what: str  # The core economic observation
    what_embedding: ndarray

    how: str  # The communication analysis
    how_embedding: ndarray

    citations: list[str]  # The quoted texts
    source: str  # link source
    date: datetime  # Datetime object for the statement
