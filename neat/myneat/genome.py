from neat.base.genome import BaseGenome, GeneIndex

from dataclasses import dataclass
import random


@dataclass
class BodyGene:
    """
    Base class for body genes
    """
    key: int  # gene primary key
    value: float

    @staticmethod
    def new(key, config):
        """ Create a new gene with values initialized by config. """
        return BodyGene(key=key,
                        value=config.body_gene_config.new())

    @staticmethod
    def crossover(gene1, gene2):
        """ Create a new gene randomly inheriting attributes from its parents. """
        assert gene1.key == gene2.key, "You can only crossover matching/homologous genes"
        return BodyGene(key=gene1.key,
                        value=gene1.value if random.random() < .5 else gene2.value)

    def mutate(self, config):
        self.value = config.body_gene_config.mutate(self.value)

    def copy(self):
        return BodyGene(self.key, self.value)

    @staticmethod
    def distance(gene1, gene2, config):
        d = abs(gene1.value - gene2.value)
        return d * config.compatibility_body_coefficient


class Genome(BaseGenome):
    def __init__(self, gid, config):
        super().__init__(gid, config)
        self.body_genes = GeneIndex(BodyGene)

    def init(self):
        super().init()
        # Initialize body genes
        self.body_genes.new(1, self.config)
        self.body_genes.new(2, self.config)
        self.body_genes.new(3, self.config)

    def crossover(self, parent1, parent2):
        super().crossover(parent1, parent2)
        # Inherit body genes
        self.body_genes = GeneIndex.crossover(parent1.body_genes, parent2.body_genes)

        return self

    def mutate(self):
        super().mutate()
        # Do not add or delete body genes, just mutate their parameters
        self.body_genes.mutate(self.config)

    @staticmethod
    def distance(genome1, genome2, config):
        brain_distance = super().distance(genome1, genome2, config)

        body_distance, disjoint = GeneIndex.compare(genome1.connections, genome2.connections, config)
        body_distance += disjoint * config.compatibility_disjoint_coefficient
        body_distance /= max(len(genome1.connections), len(genome2.connections))

        return brain_distance + body_distance
