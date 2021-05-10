"""
Handles genomes.
Heavily influenced by NEAT-Python.
"""

import random

from neat.base.genes import NodeGene, ConnectionGene
from neat.base.data import Index


class GeneIndex(Index):
    def __init__(self, gene_type):
        super().__init__("key")
        self.gene_type = gene_type

    def new(self, *args, **kwargs):
        """ Create and add a new item with given parameters. """
        item = self.gene_type.new(*args, **kwargs)
        self.add(item)
        return item

    @staticmethod
    def compare(genes1, genes2, config):
        homologous_distance = 0.0
        disjoint = 0
        if genes1 or genes2:
            for k2 in genes2.keys():
                if k2 not in genes1:
                    disjoint += 1

            for k1, g1 in genes1.items():
                g2 = genes2.get(k1)
                if g2 is None:
                    disjoint += 1
                else:
                    # Homologous genes compute their own distance value.
                    homologous_distance += type(g1).distance(g1, g2, config)
        return homologous_distance, disjoint

    @staticmethod
    def crossover(parent1, parent2):
        """
        Initialize new gene list by crossover from two parent gene indexes
        :param parent1: fitter parent gene index
        :param parent2: less fit parent gene index
        :return: new child gene index
        """
        assert (gene_type := parent1.gene_type) == parent2.gene_type
        child = GeneIndex(gene_type)
        for key, gene1 in parent1.items():
            gene2 = parent2.get(key)
            assert key not in child
            if gene2 is None:
                # Excess or disjoint gene; copy from the fittest parent
                child.add(gene1.copy())
            else:
                # Matching/Homologous gene; combine genes from both parents.
                child.add(gene_type.crossover(gene1, gene2))
        return child

    def mutate(self, config):
        for gene in self.values():
            gene.mutate(config)


class BaseGenome:
    def __init__(self, gid, config):
        self.id = gid
        self.config = config

        # self.nodes = GeneList(NodeGene)
        # self.connections = GeneList(ConnectionGene)
        self.nodes = GeneIndex(NodeGene)
        self.connections = GeneIndex(ConnectionGene)

    def init(self):
        """ Initialize a genome by connecting input and output nodes. """
        # Create input and output nodes
        for k in self.config.input_keys:
            self.nodes.new(k, self.config)
        for k in self.config.output_keys:
            self.nodes.new(k, self.config)

        # Connect input nodes to output nodes
        for k1 in self.config.input_keys:
            for k2 in self.config.output_keys:
                self.connections.new((k1, k2), self.config)

    def crossover(self, parent1, parent2):
        """
        Initialize a genome by crossover from two parent genomes.
        :param parent1: Fitter parent.
        :param parent2: Less fit parent.
        :return: This new child genome.
        """

        # Inherit connection genes
        self.connections = GeneIndex.crossover(parent1.connections, parent2.connections)
        # Inherit node genes
        self.nodes = GeneIndex.crossover(parent1.nodes, parent2.nodes)

        return self

    def mutate(self):
        """ Mutate this genome. """

        # Structural mutations
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

        # Parameter/weight mutations
        self.connections.mutate(self.config)
        self.nodes.mutate(self.config)

    def mutate_add_node(self):
        """
        Attempt to add a new node by splitting a connection.
        Surer: if no connections are available, add a connection.
        """
        if not self.connections:
            # Mutation FAIL if there are no connections to split
            # Alternative mutation: add connection instead of node
            if self.config.structural_mutation_surer:
                self.mutate_add_connection()
            return

        # Mutation SUCCESS
        conn_to_split = random.choice(list(self.connections.values()))
        conn_to_split.disable()

        new_node = self.nodes.new(next(self.config.node_key_indexer), self.config)
        i, o = conn_to_split.key
        self.connections.new((i, new_node.key), self.config, weight=1)
        self.connections.new((new_node.key, o), self.config, weight=conn_to_split.weight)

    def mutate_add_connection(self):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        Fails if the randomly generated connection already exists.
        Surer: If randomly generated connection already exists, but is disabled,
        enable it.
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
        self.connections.new(key, self.config)
        return key

    def mutate_delete_node(self):
        """ Attempt to delete a random hidden node. Fails if no hidden nodes exist. """
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
        """ Attempt to delete a random connection. Fails if no connections exist. """
        # NOTE: This may? leave nodes with no connections
        # NOTE: This may delete the only connection
        if self.connections:
            # Mutation SUCCESS
            key = random.choice(list(self.connections.keys()))
            del self.connections[key]
            return key
        # Mutation FAIL if no connections to delete
        return -1

    @staticmethod
    def distance(genome1, genome2, config):
        """
        Returns the genetic distance between two genomes. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0
        if genome1.nodes or genome2.nodes:
            node_distance, disjoint_nodes = GeneIndex.compare(genome1.nodes, genome2.nodes, config)
            node_distance += disjoint_nodes * config.compatibility_disjoint_coefficient
            node_distance /= max(len(genome1.nodes), len(genome2.nodes))

        # Compute connection gene differences.
        connection_distance = 0
        if genome1.connections or genome2.connections:
            connection_distance, disjoint_connections = GeneIndex.compare(genome1.connections, genome2.connections, config)
            connection_distance += disjoint_connections * config.compatibility_disjoint_coefficient
            connection_distance /= max(len(genome1.connections), len(genome2.connections))

        return node_distance + connection_distance

    def size(self):
        """ Returns genome 'complexity', taken to be number of nodes + number of connections """
        return len(self.nodes) + len(self.connections)

