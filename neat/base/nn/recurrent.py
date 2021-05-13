""" Heavily influenced by NEAT-Python """

from collections import defaultdict

from neat.base.genome import BaseGenome
from neat.base.activations import activation_defs
from neat.base.aggregations import aggregation_defs

from .graphs import required_for_output


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
    def create(genome: BaseGenome):
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

