import numpy as np

from typing import Tuple
from typing import Dict
from typing import List
from typing import TypeVar
from typing import Optional

from enum        import Enum
from dataclasses import dataclass
#from pandas      import DataFrame, Series

Number = TypeVar('Number', int, float)
Array  = TypeVar('Array', List, np.array)


class DetName(Enum):
    demo          = 1
    new           = 2
    next100       = 3
    flex100       = 4
    next500       = 5
    next_2x2      = 6
    next_3x3      = 7
    next_hd       = 8
    next100_alaHD = 9


@dataclass
class VolumeDim:
	z_min : float
	z_max : float
	rad   : float
