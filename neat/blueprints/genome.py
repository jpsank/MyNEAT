"""
Defines blueprint for configuration and functions applied to genomes, plus counter for genome id's.
Holds methods for creating, mutating, copying, crossover, and calculating genomic distance.
"""

from itertools import count
from dataclasses import dataclass, field
import random
from typing import *
import itertools

from neat.model import *
from neat.blueprints.primitives import Blueprint
from neat.blueprints.genes import GeneBP, NodeBP, ConnBP


# --------------- GENOME CONFIGURABLES ---------------

@dataclass
class GenomeBP(Blueprint):
    """ Contains genome configuration and counters for a simulation """

    __constructor__ = Genome
    __primary_keys__ = ("id",)

    node: NodeBP
    conn: ConnBP

    # Network initialization options
    num_inputs: int
    num_outputs: int

    # Network mutation options
    conn_add_prob: float  # Connection add rate
    conn_delete_prob: float   # Connection remove rate
    node_add_prob: float  # Node add rate
    node_delete_prob: float  # Node remove rate

    # Structural mutations
    single_structural_mutation: bool  # If enabled, only one structural mutation per genome per "generation"
    structural_mutation_surer: bool  # If enabled, perform alternative structural mutations on failure

    # Genome compatibility options
    compatibility_disjoint_coefficient: float # c2, takes the place of both c1 and c2
    compatibility_weight_coefficient: float  # c3

    # Genome ID counter and input/output node IDs
    __id_counter: count = field(default_factory=count)
    input_ids: list = field(init=False)
    output_ids: list = field(init=False)

    def __post_init__(self):
        # By convention, input pins have negative keys, and the output pins have keys 0,1,...
        self.input_ids = [-i - 1 for i in range(self.num_inputs)]
        self.output_ids = [i for i in range(self.num_outputs)]

    def create(self, **kwargs) -> Genome:
        """ 
        Create a new genome with the given attributes.
        Input and output nodes are connected by default. 
        """

        if "id" not in kwargs:
            kwargs["id"] = next(self.__id_counter)
        
        # Create input and output nodes
        if "nodes" not in kwargs:
            kwargs["nodes"] = {i: self.node.create(id=i) for i in self.input_ids + self.output_ids}

        # Connect input nodes to output nodes
        if "conns" not in kwargs:
            kwargs["conns"] = {(i1, i2): self.conn.create(in_node=i1, out_node=i2) for i1, i2 in itertools.product(self.input_ids, self.output_ids)}
                
        return Genome(**kwargs)

    def __mutate_add_node(self, genome: Genome):
        """
        Attempt to add a new node by splitting a connection.
        Surer: if no connections are available, add a connection.
        """
        if not genome.conns:
            # Mutation FAIL if there are no connections to split
            # Alternative mutation: add connection instead of node
            if self.structural_mutation_surer:
                return self.__mutate_add_conn(genome)
            return

        # Mutation SUCCESS
        (i, o), conn_to_split = random.choice(list(genome.conns.items()))
        conn_to_split.enabled = False

        node = self.node.create()
        genome.nodes[node.id] = node
        genome.conns[(i, node.id)] = self.conn.create(in_node=i, out_node=node.id, weight=1)
        genome.conns[(node.id, o)] = self.conn.create(in_node=node.id, out_node=o, weight=conn_to_split.weight)
        return

    def __mutate_add_conn(self, genome: Genome):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        Fails if the randomly generated connection already exists.
        Surer: If randomly generated connection already exists, but is disabled,
        enable it.
        """
        # Note: This allows for nodes to connect to themselves.

        # Any node (input, hidden, or output) can be the in node
        possible_inputs = list(genome.nodes.keys())
        # Only output and hidden nodes may be the out node
        possible_outputs = list(set(possible_inputs) - set(self.input_ids))

        in_node = random.choice(possible_inputs)
        out_node = random.choice(possible_outputs)
        key = (in_node, out_node)

        if key in genome.conns:
            # Mutation FAIL if connection already exists
            # Alternative mutation: set existing connection enabled instead of adding a new connection
            if self.structural_mutation_surer:
                genome.conns[key].enabled = True
            return

        if in_node in self.output_ids and out_node in self.output_ids:
            # Mutation FAIL if tried to connect two output nodes (not allowed)
            # No alternative mutation
            return

        # Mutation SUCCESS
        genome.conns[key] = self.conn.create(in_node=in_node, out_node=out_node)
        return key

    def __mutate_delete_node(self, genome: Genome):
        """ Attempt to delete a random hidden node. Fails if no hidden nodes exist. """
        # NOTE: This may? delete the only connection

        available_nodes = [i for i in genome.nodes.keys()
                           if i not in self.output_ids and i not in self.input_ids]
        if not available_nodes:
            # Mutation FAIL if no hidden nodes to delete
            # No alternative mutation
            return -1

        # Mutation SUCCESS
        del_id = random.choice(available_nodes)

        conns_to_delete = set()
        for conn in genome.conns.values():
            if conn.in_node == del_id or conn.out_node == del_id:
                conns_to_delete.add(conn.key)

        for key in conns_to_delete:
            del genome.conns[key]
        del genome.nodes[del_id]

        return del_id

    def __mutate_delete_conn(self, genome: Genome):
        """ Attempt to delete a random connection. Fails if no connections exist. """
        # NOTE: This may? leave nodes with no connections
        # NOTE: This may delete the only connection
        if genome.conns:
            # Mutation SUCCESS
            key = random.choice(list(genome.conns.keys()))
            del genome.conns[key]
            return key
        # Mutation FAIL if no connections to delete
        return -1

    def mutate(self, genome: Genome):
        """ Mutate a genome. """

        # Structural mutations
        mutations = (self.__mutate_add_node, self.__mutate_delete_node, self.__mutate_add_conn, self.__mutate_delete_conn)
        probs = (self.node_add_prob, self.node_delete_prob, self.conn_add_prob, self.conn_delete_prob)
        
        if self.single_structural_mutation:
            div = max(1, sum(probs))
            cum = 0
            r = random.random()
            for mut, prob in zip(mutations, probs):
                if r < (cum := cum + prob / div):
                    mut(genome)
                    break
        else:
            for mut, prob in zip(mutations, probs):
                if random.random() < prob:
                    mut(genome)

        # Parameter/weight mutations
        for conn in genome.conns.values():
            self.conn.mutate(conn)
        for node in genome.nodes.values():
            self.node.mutate(node)
        
    def copy(self, genome: Genome) -> Genome:
        """ Copy a genome. """
        return Genome(
            id=genome.id,
            nodes={k: self.node.copy(node) for k, node in genome.nodes.items()},
            conns={k: self.conn.copy(conn) for k, conn in genome.conns.items()},
        )
    
    def __crossover_genes(self, a: 'dict[Any, Gene]', b: 'dict[Any, Gene]', gene_bp: GeneBP) -> 'dict[Any, Gene]':
        """
        Crossover two maps of the same type of gene.
        Pre-condition: a is from the parent with higher fitness
        """
        c = {}
        for id_, gene1 in a.items():
            if (gene2 := b.get(id_)) is None:
                # Excess or disjoint gene; copy from the fittest parent
                c[id_] = gene_bp.copy(gene1)
            else:
                # Matching/Homologous gene; combine genes from both parents.
                c[id_] = gene_bp.crossover(gene1, gene2)
        return c

    def crossover(self, a: Genome, b: Genome) -> Genome:
        """
        Create a new genome by crossover from two parent genomes.
        Pre-condition: a is fitter than b
        """

        conns = self.__crossover_genes(a.conns, b.conns, self.conn)
        nodes = self.__crossover_genes(a.nodes, b.nodes, self.node)
        return self.create(conns=conns, nodes=nodes)

    def __compare_genes(self, a: 'dict[Any, Gene]', b: 'dict[Any, Gene]', gene_bp: GeneBP):
        """ Compare two maps of the same type of gene. """
        homologous_distance, disjoint = 0.0, 0
        if a or b:
            for i in b.keys():
                if i not in a:
                    disjoint += 1

            for i, gene1 in a.items():
                if (gene2 := b.get(i)) is None:
                    disjoint += 1
                else:
                    # Homologous genes compute their own distance value.
                    homologous_distance += gene_bp.distance(gene1, gene2)
        return homologous_distance, disjoint

    def distance(self, a: Genome, b: Genome) -> float:
        """
        Returns the genetic distance between two genomes. This distance value
        is used to compute genome compatibility for speciation.
        """

        total_distance = 0
        for a_genes, b_genes, field in ( (a.nodes, b.nodes, self.node), (a.conns, b.conns, self.conn) ):
            if a_genes or b_genes:
                dist, disjoint = self.__compare_genes(a_genes, b_genes, field)
                dist += disjoint * self.compatibility_disjoint_coefficient
                total_distance += dist / max(len(a_genes), len(b_genes))
        
        return total_distance
