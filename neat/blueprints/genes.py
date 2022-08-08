"""
Defines blueprints for configuration and functions applied to node and connection genes, plus counter for node id's.
"""

from itertools import count
from dataclasses import dataclass, field
from typing import *

from neat.model import Gene, NodeGene, ConnGene
from neat.blueprints.primitives import Blueprint, FloatBP, BoolBP, StringBP


# --------------- GENE CONFIGURABLES ---------------

class GeneBP(Blueprint):
    """ Abstract base class for all gene blueprints. """
    
    __constructor__ = None
    __primary_keys__ = ()
    
    def __are_homologous(self, a: Gene, b: Gene) -> bool:
        return all(getattr(a, k) == getattr(b, k) for k in self.__primary_keys__)

    def __iter_configs(self) -> Iterator[Tuple[str, Blueprint]]:
        """ Iterate over all configurable attributes of this gene. """
        for k, t in self.__annotations__.items():
            if issubclass(t, Blueprint):
                yield k, getattr(self, k)
    
    def create(self, **kwargs) -> Gene:
        """
        Create a new gene with the given attributes.
        Pre-condition: all primary keys are specified
        """
        assert all(k in kwargs for k in self.__primary_keys__), "You can only create a gene with all primary keys specified"

        for k, cfg in self.__iter_configs():
            if k not in kwargs:
                kwargs[k] = cfg.create()
        
        return self.__constructor__(**kwargs)

    def mutate(self, gene: Gene):
        """ Mutate a gene by mutating each of its configurable attribute. """
        for k, cfg in self.__iter_configs():
            v = getattr(gene, k)
            setattr(gene, k, cfg.mutate(v))

    def copy(self, gene: Gene) -> Gene:
        """ Copy a gene by copying each of its configurable attribute. """
        kwargs = {k: getattr(gene, k) for k in self.__primary_keys__}
        for k, cfg in self.__iter_configs():
            kwargs[k] = cfg.copy(getattr(gene, k))
        return self.__constructor__(**kwargs)

    def crossover(self, a: Gene, b: Gene) -> Gene:
        """ 
        Create a new gene randomly inheriting attributes from its parents. 
        Pre-condition: a and b are homologous
        """
        assert self.__are_homologous(a, b), "You can only crossover matching/homologous genes"

        kwargs = {k: getattr(a, k) for k in self.__primary_keys__}
        for k, cfg in self.__iter_configs():
            kwargs[k] = cfg.crossover(getattr(a, k), getattr(b, k))
        return self.__constructor__(**kwargs)

    def distance(self, a: Gene, b: Gene) -> float:
        """
        Calculate the genomic distance between two genes. 
        Pre-condition: a and b are homologous
        """
        assert self.__are_homologous(a, b), "You can only find distance between matching/homologous genes"

        return sum(cfg.distance(getattr(a, k), getattr(b, k)) for k, cfg in self.__iter_configs())


@dataclass
class NodeBP(GeneBP):
    """ Blueprint for node genes. """

    __constructor__ = NodeGene
    __primary_keys__ = ("id",)

    response: FloatBP
    bias: FloatBP
    activation: StringBP
    aggregation: StringBP

    __id_counter: count = field(default_factory=count)

    def create(self, **kwargs) -> NodeGene:
        """ Create a new node gene with the given attributes. """

        if "id" not in kwargs:
            kwargs["id"] = next(self.__id_counter)
        
        return super().create(**kwargs)

@dataclass
class ConnBP(GeneBP):
    """ Blueprint for connection genes. """

    __constructor__ = ConnGene
    __primary_keys__ = ("in_node", "out_node")

    weight: FloatBP
    enabled: BoolBP