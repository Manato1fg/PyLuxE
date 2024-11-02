from enum import Enum


class Medium(Enum):
    LIGHT = 0
    ELECTRON = 1


class InitializeType(Enum):
    ZEROS = 0
    GAUSSIAN = 1
    RANDOM = 2
