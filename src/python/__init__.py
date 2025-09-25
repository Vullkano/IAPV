"""
IAPV - AI for Virtual Characters
Python implementation
"""

__version__ = "1.0.0"
__author__ = "IAPV Contributors"
__description__ = "AI framework for virtual characters covering planning, learning, locomotion, and more"

from .common import math_utils
from .locomotion import steering_behaviors
from .planning import pathfinding

__all__ = [
    'math_utils',
    'steering_behaviors', 
    'pathfinding'
]