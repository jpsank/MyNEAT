""" Handles genomes. """

from dataclasses import dataclass
import random
from collections import defaultdict

from myneat.config import Config, default_config


@dataclass
class NodeGene:
    """
    Base class for CPPN node genes
    out = activation(bias + response * aggregation(inputs))
    """
    id: int  # node id (primary key)
    type: str  # input, hidden, output
    response: float
    bias: float
    activation: str
    aggregation: str

    @staticmethod
    def new(config, type_):
        return NodeGene(config.node_id_counter(), type_, config.response_config.new(), config.bias_config.new(),
                        config.activation_config.new(), config.aggregation_config.new())

    def mutate(self, config):
        self.response = config.response_config.mutate(self.response)
        self.bias = config.bias_config.mutate(self.bias)
        self.activation = config.activation_config.mutate(self.activation)
        self.aggregation = config.aggregation_config.mutate(self.aggregation)

    def copy(self):
        return NodeGene(self.id, self.type, self.response, self.bias, self.activation, self.aggregation)

    @staticmethod
    def distance(node1, node2, config):
        d = abs(node1.bias - node2.bias) + abs(node1.response - node2.response)
        if node1.activation != node2.activation:
            d += 1.0
        if node1.aggregation != node2.aggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient


@dataclass
class ConnectionGene:
    """
    Base class for CPPN connection genes
    """
    innovation: int  # innovation number (primary key)
    in_node: int
    out_node: int
    weight: float
    enabled: bool

    def disable(self):
        self.enabled = False

    @staticmethod
    def new(config, in_node, out_node, weight=None):
        if weight is None:
            weight = config.weight_config.new()
        return ConnectionGene(config.innovation_counter(), in_node, out_node, weight, config.enabled_config.new())

    def mutate(self, config):
        self.weight = config.weight_config.mutate(self.weight)
        self.enabled = config.enabled_config.mutate(self.enabled)

    def copy(self):
        return ConnectionGene(self.innovation, self.in_node, self.out_node, self.weight, self.enabled)

    @staticmethod
    def distance(conn1, conn2, config):
        d = abs(conn1.weight - conn2.weight)
        if conn1.enabled != conn2.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient


class Genome:
    def __init__(self, gid, config: Config):
        self.id = gid
        self.config: Config = config

        self.nodes: {int: NodeGene} = {}
        self.connections: {int: ConnectionGene} = {}
        self.node_ids_by_type: {str: [int]} = defaultdict(list)

        self.fitness = None

    @staticmethod
    def new(config: Config):
        return Genome(config.genome_id_counter(), config)

    def init(self):
        input_nodes = [self.create_node("input") for _ in range(self.config.num_inputs)]
        output_nodes = [self.create_node("output") for _ in range(self.config.num_outputs)]
        for node1 in input_nodes:
            for node2 in output_nodes:
                self.create_connection(node1.id, node2.id)

    def size(self):
        """ Returns genome 'complexity', taken to be number of nodes + number of connections """
        return len(self.nodes) + len(self.connections)

    def add_connection(self, conn):
        self.connections[conn.innovation] = conn

    def add_node(self, node):
        self.nodes[node.id] = node
        self.node_ids_by_type[node.type].append(node.id)

    def create_connection(self, node_in_id, node_out_id, weight=None):
        """
        Create a new connection gene in the genome
        """
        new_conn = ConnectionGene.new(self.config, node_in_id, node_out_id, weight)
        self.add_connection(new_conn)
        return new_conn

    def create_node(self, node_type="hidden", ):
        """
        Create a new node gene in the genome
        """
        new_node = NodeGene.new(self.config, node_type)
        self.add_node(new_node)
        return new_node

    def mutate(self):
        """ Mutates this genome """

        if random.random() < self.config.node_add_prob:
            self.mutate_add_node()

        if random.random() < self.config.node_delete_prob:
            self.mutate_delete_node()

        if random.random() < self.config.conn_add_prob:
            self.mutate_add_connection()

        if random.random() < self.config.conn_delete_prob:
            self.mutate_delete_connection()

        # Mutate connection genes
        for conn in self.connections.values():
            conn.mutate(self.config)

        # Mutate node genes (bias, response, etc.)
        for node in self.nodes.values():
            node.mutate(self.config)

    def mutate_add_connection(self, max_attempts=10):
        # TODO: this allows for nodes to connect to themselves. I think that's bad
        # TODO: also, maybe this should be allowed to take input from an output node?
        node1_id = random.choice(self.node_ids_by_type["input"] + self.node_ids_by_type["hidden"])
        node2_id = random.choice(self.node_ids_by_type["hidden"] + self.node_ids_by_type["output"])

        for conn in self.connections.values():
            if (conn.in_node == node1_id and conn.out_node == node2_id) \
                    or (conn.in_node == node2_id and conn.out_node == node1_id):
                # Proposed connection already exists, so try again until reaching max attempts
                if max_attempts == 1:
                    return -1
                return self.mutate_add_connection(max_attempts=max_attempts-1)

        new_conn = self.create_connection(node1_id, node2_id)
        return new_conn.innovation

    def mutate_add_node(self):
        conn = random.choice(list(self.connections.values()))
        conn.disable()

        new_node = self.create_node("hidden")
        self.create_connection(conn.in_node, new_node.id, 1)
        self.create_connection(new_node.id, conn.out_node, conn.weight)
        return new_node.id

    def mutate_delete_connection(self):
        # TODO: This can leave nodes with no connections; this should not be allowed?
        if len(self.connections) > 1:  # Don't delete the only connection
            innovation_nbr = random.choice(list(self.connections.keys()))
            del self.connections[innovation_nbr]
            return innovation_nbr
        return -1

    def mutate_delete_node(self):
        # Do nothing if there are no hidden nodes
        available_nodes = self.node_ids_by_type["hidden"]
        if not available_nodes:
            return -1

        del_id = random.choice(available_nodes)

        conns_to_delete = set()
        for innovation_nbr, conn in self.connections.items():
            if del_id in (conn.in_node, conn.out_node):
                conns_to_delete.add(innovation_nbr)

        # We can't have it deleting all the connections
        if len(conns_to_delete) == len(self.connections):
            return -1

        for innovation_nbr in conns_to_delete:
            del self.connections[innovation_nbr]

        del self.nodes[del_id]

        return del_id

    def copy(self):
        new_genome = Genome.new(self.config)
        new_genome.nodes = self.nodes
        new_genome.connections = self.connections
        new_genome.node_ids_by_type = self.node_ids_by_type
        return new_genome

    @staticmethod
    def distance(genome1, genome2, config):
        """
        Compute compatibility distance between two genomes
        """
        matching_node_genes = 0
        matching_conn_genes = 0
        node_difference = 0
        conn_difference = 0
        disjoint_genes = 0
        excess_genes = 0

        highest_id1 = max(genome1.nodes.keys())
        highest_id2 = max(genome2.nodes.keys())
        highest_id = max(highest_id1, highest_id2)

        i = 0
        while i <= highest_id:
            node1 = genome1.nodes.get(i)
            node2 = genome2.nodes.get(i)
            if node1 is None and highest_id1 < i and node2 is not None:
                # we are past genome1's highest innovation yet genome2 has this gene, therefore genome2 has excess
                excess_genes += 1
            elif node2 is None and highest_id2 < i and node1 is not None:
                # we are past genome2's highest innovation yet genome1 has this gene, therefore genome1 has excess
                excess_genes += 1
            elif node1 is None and highest_id1 > i and node2 is not None:
                # genome1 lacks this particular gene but has more at higher innovation, therefore disjoint
                disjoint_genes += 1
            elif node2 is None and highest_id2 > i and node1 is not None:
                # genome2 lacks this particular gene but has more at higher innovation, therefore disjoint
                disjoint_genes += 1
            elif node1 is not None and node2 is not None:
                # both genome1 and genome2 have this gene, therefore matching
                matching_node_genes += 1
                node_difference += NodeGene.distance(node1, node2, config)
            i += 1

        highest_innovation1 = max(genome1.connections.keys())
        highest_innovation2 = max(genome2.connections.keys())
        highest_innovation = max(highest_innovation1, highest_innovation2)

        i = 0
        while i <= highest_innovation:
            conn1 = genome1.connections.get(i)
            conn2 = genome2.connections.get(i)
            if conn1 is None and highest_innovation1 < i and conn2 is not None:
                # we are past genome1's highest innovation yet genome2 has this gene, therefore genome2 has excess
                excess_genes += 1
            elif conn2 is None and highest_innovation2 < i and conn1 is not None:
                # we are past genome2's highest innovation yet genome1 has this gene, therefore genome1 has excess
                excess_genes += 1
            elif conn1 is None and highest_innovation1 > i and conn2 is not None:
                # genome1 lacks this particular gene but has more at higher innovation, therefore disjoint
                disjoint_genes += 1
            elif conn2 is None and highest_innovation2 > i and conn1 is not None:
                # genome2 lacks this particular gene but has more at higher innovation, therefore disjoint
                disjoint_genes += 1
            elif conn1 is not None and conn2 is not None:
                # both genome1 and genome2 have this gene, therefore matching
                matching_conn_genes += 1
                conn_difference += ConnectionGene.distance(conn1, conn2, config)
            i += 1

        average_difference = (node_difference + conn_difference) / (matching_node_genes + matching_conn_genes)
        larger_size = max(genome1.size(), genome2.size())

        return (config.compatibility_excess_coefficient * excess_genes) / larger_size +\
               (config.compatibility_disjoint_coefficient * disjoint_genes) / larger_size +\
               (config.compatibility_weight_coefficient * average_difference)

    @staticmethod
    def crossover(parent1, parent2):
        """
        Assumes parent1 is more fit than parent2 and both parents have same config
        """
        child = Genome.new(parent1.config)
        for node1 in parent1.nodes.values():
            child.add_node(node1.copy())

        for innovation_nbr, conn1 in parent1.connections.items():
            if innovation_nbr in parent2.connections.keys():  # matching gene
                conn2 = parent2.connections[innovation_nbr]
                child.add_connection(conn1.copy() if random.random() < 0.5 else conn2.copy())
            else:  # disjoint or excess gene
                child.add_connection(conn1.copy())
        return child


if __name__ == '__main__':
    genome = Genome.new(default_config)
    genome.init()

