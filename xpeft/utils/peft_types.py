import enum
from typing import Optional


class PeftType(str, enum.Enum):
    PLORA = "PLORA"
    MOELORA = "MOELORA"


class TaskType(str, enum.Enum):
    CAUSAL_LM = "CAUSAL_LM"
