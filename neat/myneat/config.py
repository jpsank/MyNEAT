"""
Contains configuration for my NEAT
"""

from neat.base.config import FloatConfig, BoolConfig, StringConfig, BaseConfig


class Config(BaseConfig):
    """
    Contains configuration and counters for my NEAT simulation
    """

    # body gene options
    compatibility_body_coefficient = 1.0
    body_gene_config = FloatConfig(
        init_mean=0.5,
        init_stdev=0.2,
        max_value=1.0,
        min_value=0.0,
        mutate_power=0.1,
        mutate_rate=0.5,
        replace_rate=0.0
    )

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


