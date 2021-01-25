""" Phenome generation (not yet implemented). """

from collections import defaultdict

from myneat.myneat.genome import Genome
from myneat.myneat.activations import activation_defs
from myneat.myneat.aggregations import aggregation_defs


def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.
    Returns a set of identifiers of required nodes, not including inputs.
    """

    # This process is analogous to an infection, starting at the output nodes and iterating backwards,
    # "infecting" nodes that connect to other infected nodes. Thus, nodes that do not connect to an
    # eventual output will avoid infection; these nodes are Not Required.

    infected = set(outputs)
    while True:
        # Newly infect nodes that connect to an already-infected node
        nodes_to_infect = set(a for (a, b) in connections if a not in infected and b in infected)
        if not nodes_to_infect:
            # No more new nodes that connect to an eventual output. Done iterating backwards
            break

        infected = infected.union(nodes_to_infect)

    return infected - set(inputs)


class RecurrentNetwork:
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        self.i_values = {}
        for node_key in list(inputs) + list(outputs):
            self.i_values[node_key] = 0.0
        for node_key, _, _, _, _, node_inputs in self.node_evals:
            self.i_values[node_key] = 0.0
            for in_key, _ in node_inputs:
                self.i_values[in_key] = 0.0
        self.o_values = dict(self.i_values)

    @staticmethod
    def new(genome: Genome):
        """ Receives a genome and returns its phenotype (a RecurrentNetwork). """
        required = required_for_output(genome.config.input_keys, genome.config.output_keys, genome.connections.keys())

        # Gather inputs and expressed connections for each output node.
        node_to_inputs = defaultdict(list)
        for conn in genome.connections.values():
            if not conn.enabled:
                continue

            in_node, out_node = conn.key
            if in_node not in required and out_node not in required:
                continue

            node_to_inputs[out_node].append((in_node, conn.weight))

        node_evals = []
        for node_key, inputs in node_to_inputs.items():
            node = genome.nodes[node_key]
            activation_function = activation_defs.get(node.activation)
            aggregation_function = aggregation_defs.get(node.aggregation)
            node_evals.append((node_key, activation_function, aggregation_function, node.bias, node.response, inputs))

        return RecurrentNetwork(genome.config.input_keys, genome.config.output_keys, node_evals)

    def reset(self):
        self.i_values = dict((k, 0.0) for k in self.i_values)
        self.o_values = dict((k, 0.0) for k in self.o_values)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for input_key, v in zip(self.input_nodes, inputs):
            self.i_values[input_key] = v
            self.o_values[input_key] = v

        for node, activation, aggregation, bias, response, node_inputs in self.node_evals:
            node_inputs = [self.i_values[in_key] * weight for in_key, weight in node_inputs]
            agg = aggregation(node_inputs)
            self.o_values[node] = activation(bias + response * agg)

        outputs = [self.o_values[i] for i in self.output_nodes]
        # Switch so output values are the inputs for next activation
        self.i_values, self.o_values = self.o_values, self.i_values

        return outputs

