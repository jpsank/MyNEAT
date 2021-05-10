"""
Contains configuration for realtime NEAT
"""

from neat.base import BaseConfig
from neat.base.config import FloatConfig, BoolConfig, StringConfig


class Config(BaseConfig):
    """
    Contains configuration and counters for a rt-NEAT simulation
    """

    # real-time NEAT
    #   Based on law of eligibility: n = m/(P*I)
    #       n: replacement frequency (number of ticks between replacements)
    #       m: minimum age (minimum time alive, in ticks, before an agent is eligible to be removed)
    #       P: population size
    #       I: preferred ineligibility fraction (fraction of population at any given time that should be ineligible)
    pop_size = 100
    minimum_age = 500
    ineligibility_fraction = 0.5
    replacement_frequency = round(minimum_age / (pop_size * ineligibility_fraction))
    reorganization_frequency = 5  # adjust compat threshold & reassign species every _ replacements (=5 in NERO)


