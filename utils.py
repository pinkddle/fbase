from __future__ import annotations
from enum import Enum
import pandas as pd

from typing import Iterable

class Dataset(Enum):
    SYNTHETIC_TEST = "path/to/data"
    SYNTHETIC_BIG = "path/to/data"
    

def get_dataset(ds: Dataset) -> pd.DataFrame:
    pass
