import numpy as np

from typing import Tuple
from typing import Dict
from typing import List
from typing import TypeVar
from typing import Optional

from enum        import Enum
from dataclasses import dataclass
from pandas      import DataFrame, Series

Number = TypeVar('Number', int, float)
Array  = TypeVar('Array', List, np.array)


class DetName(Enum):
	new     = 1
	next100 = 2
	next500 = 3


@dataclass
class VolumeDim:
	z_min : float
	z_max : float
	rad   : float
