"""
Directed graph algorithm implementations.
"""


def creates_cycle(connections, test):
    """
    Returns true if the addition of the 'test' connection would create a cycle,
    assuming that no cycle already exists in the graph represented by 'connections'.
    """
    i, o = test
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for a, b in connections:
            if a in visited and b not in visited:
                if b == i:
                    return True

                visited.add(b)
                num_added += 1

        if num_added == 0:
            return False


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


def feed_forward_layers(inputs, outputs, connections):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    """

    required = required_for_output(inputs, outputs, connections)

    layers = []
    s = set(inputs)
    while 1:
        next_nodes = set()
        # Find candidate nodes c for the next layer.  These nodes should connect
        # a node in s to a node not in s.
        candidates = set(b for (a, b) in connections if a in s and b not in s)
        # Keep only the used nodes whose entire input set is contained in s.
        for candidate in candidates:
            input_set = (i for (i, o) in connections if o == candidate)
            if candidate in required and all(i in s for i in input_set):
                next_nodes.add(candidate)

        if not next_nodes:
            break

        layers.append(next_nodes)
        s = s.union(next_nodes)

    return layers


