"""
Handles genomes.
Heavily influenced by NEAT-Python.
"""

import random

from myneat.myneat.genes import NodeGene, ConnectionGene
from myneat.myneat.config import Config


class Genome:
    def __init__(self, gid, config: Config):
        self.id = gid
        self.config: Config = config

        self.nodes: {int: NodeGene} = {}
        self.connections: {int: ConnectionGene} = {}

    @staticmethod
    def new(config: Config):
        """ Create an altogether new genome for the given configuration. """
        return Genome(next(config.genome_id_counter), config)

    def init(self):
        """ Connect a genome based on its configuration. """
        for k in self.config.input_keys:
            self.nodes[k] = self.create_node(k)
        for k in self.config.output_keys:
            self.nodes[k] = self.create_node(k)

        for k1 in self.config.input_keys:
            for k2 in self.config.output_keys:
                self.create_connection((k1, k2))

    @staticmethod
    def crossover(parent1, parent2):
        """
        Create a new genome by crossover from two parent genomes
        :param parent1: fitter parent
        :param parent2: less fit parent
        :return: newly generated child genome
        """

        child = Genome.new(parent1.config)

        # Inherit connection genes
        for key, conn1 in parent1.connections.items():
            conn2 = parent2.connections.get(key)
            if conn2 is None:
                # Excess or disjoint gene; copy from the fittest parent.
                child.add_connection(conn1.copy())
            else:
                # Matching/Homologous gene; combine genes from both parents.
                child.add_connection(ConnectionGene.crossover(conn1, conn2))

        # Inherit node genes
        for key, node1 in parent1.nodes.items():
            node2 = parent2.nodes.get(key)
            assert key not in child.nodes
            if node2 is None:
                # Extra gene; copy from the fittest parent
                child.add_node(node1.copy())
            else:
                # Matching/Homologous gene; combine genes from both parents.
                child.add_node(NodeGene.crossover(node1, node2))

        return child

    def add_connection(self, conn):
        self.connections[conn.key] = conn

    def add_node(self, node):
        self.nodes[node.key] = node

    def create_connection(self, key, weight=None):
        """ Create a new connection gene in the genome. """
        new_conn = ConnectionGene.new(self.config, key, weight)
        self.add_connection(new_conn)
        return new_conn

    def create_node(self, key=None):
        """ Create a new node gene in the genome. """
        new_node = NodeGene.new(self.config, key)
        self.add_node(new_node)
        return new_node

    def mutate(self):
        """ Mutates this genome """

        if self.config.single_structural_mutation:
            div = max(1, (self.config.node_add_prob + self.config.node_delete_prob +
                          self.config.conn_add_prob + self.config.conn_delete_prob))

            cum = 0
            r = random.random()
            if r < (cum := cum + self.config.node_add_prob) / div:
                self.mutate_add_node()
            elif r < (cum := cum + self.config.node_delete_prob) / div:
                self.mutate_delete_node()
            elif r < (cum := cum + self.config.conn_add_prob) / div:
                self.mutate_add_connection()
            elif r < (cum + self.config.conn_delete_prob) / div:
                self.mutate_delete_connection()
        else:
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

    def mutate_add_node(self):
        if not self.connections:
            # Mutation FAIL if there are no connections to split
            # Alternative mutation: add connection instead of node
            if self.config.structural_mutation_surer:
                self.mutate_add_connection()
            return

        # Mutation SUCCESS
        conn_to_split = random.choice(list(self.connections.values()))
        conn_to_split.disable()

        new_node = self.create_node()
        i, o = conn_to_split.key
        self.create_connection((i, new_node.key), 1)
        self.create_connection((new_node.key, o), conn_to_split.weight)

    def mutate_add_connection(self):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        # Note: This allows for nodes to connect to themselves.

        # Any node (input, hidden, or output) can be the in node
        possible_inputs = list(self.nodes)
        in_node = random.choice(possible_inputs)
        # Only output and hidden nodes may be the out node
        possible_outputs = list(set(possible_inputs) - set(self.config.input_keys))
        out_node = random.choice(possible_outputs)

        key = (in_node, out_node)
        if key in self.connections:
            # Mutation FAIL if connection already exists
            # Alternative mutation: set existing connection enabled instead of adding a new connection
            if self.config.structural_mutation_surer:
                self.connections[key].enabled = True
            return

        if in_node in self.config.output_keys and out_node in self.config.output_keys:
            # Mutation FAIL if tried to connect two output nodes (not allowed)
            # No alternative mutation
            return

        # Mutation SUCCESS
        self.create_connection(key)
        return key

    def mutate_delete_node(self):
        # NOTE: This may? delete the only connection

        available_nodes = [k for k in self.nodes.keys()
                           if k not in self.config.output_keys
                           and k not in self.config.input_keys]
        if not available_nodes:
            # Mutation FAIL if no hidden nodes to delete
            # No alternative mutation
            return -1

        # Mutation SUCCESS
        del_key = random.choice(available_nodes)

        conns_to_delete = set()
        for conn in self.connections.values():
            if del_key in conn.key:
                conns_to_delete.add(conn.key)

        for key in conns_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        return del_key

    def mutate_delete_connection(self):
        # NOTE: This may? leave nodes with no connections
        # NOTE: This may delete the only connection
        if self.connections:
            # Mutation SUCCESS
            key = random.choice(list(self.connections.keys()))
            del self.connections[key]
            return key
        # Mutation FAIL if no connections to delete
        return -1

    def copy(self):
        new_genome = Genome.new(self.config)
        new_genome.nodes = self.nodes
        new_genome.connections = self.connections
        return new_genome

    @staticmethod
    def distance(genome1, genome2, config):
        """
        Returns the genetic distance between two genomes. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if genome1.nodes or genome2.nodes:
            disjoint_nodes = 0
            for k2 in genome2.nodes:
                if k2 not in genome1.nodes:
                    disjoint_nodes += 1

            for k1, n1 in genome1.nodes.items():
                n2 = genome2.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += NodeGene.distance(n1, n2, config)

            max_nodes = max(len(genome1.nodes), len(genome2.nodes))
            node_distance = (node_distance + (config.compatibility_disjoint_coefficient * disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if genome1.connections or genome2.connections:
            disjoint_connections = 0
            for k2 in genome2.connections:
                if k2 not in genome1.connections:
                    disjoint_connections += 1

            for k1, c1 in genome1.connections.items():
                c2 = genome2.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += ConnectionGene.distance(c1, c2, config)

            max_conns = max(len(genome1.connections), len(genome2.connections))
            connection_distance = (connection_distance +
                                   (config.compatibility_disjoint_coefficient * disjoint_connections)) / max_conns

        distance = node_distance + connection_distance
        return distance

    def size(self):
        """ Returns genome 'complexity', taken to be number of nodes + number of connections """
        return len(self.nodes) + len(self.connections)

