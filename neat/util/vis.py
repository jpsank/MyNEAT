import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

from neat.model import Genome


def plt_population(pop, config, out="net.html"):
    best_of_each_species = [s.get_best() for s in pop.species.values()]
    best_of_each_species = sorted(best_of_each_species, key=lambda a: a.fitness, reverse=True)
    plt_genomes({a.species_id: a.genome for a in best_of_each_species}, config.population.genome.input_ids, config.population.genome.output_ids)


def required_for_output(inputs, outputs, connections):
    """ This is kinda like breadth-first search """
    stack = [[i] for i in inputs]
    required = set()
    while len(stack) > 0:
        nodes = stack.pop()
        curr = nodes[-1]
        if curr in outputs:
            required.update(nodes)
        else:
            for (a, b) in connections:
                if a == curr and b not in nodes:
                    stack.append([*nodes, b])
    return required


def plt_genomes(genome_map: 'dict[int, Genome]', input_ids, output_ids, out="net.html"):
    net = Network(height="100%", width="100%", bgcolor="#222222", font_color="white")
    x, y = 0, 0
    x_step, y_step = 1000, 500
    for gid, g in genome_map.items():

        get_nid = lambda key: f"{gid}:{key}"
        get_ntype = lambda key: (
            is_in := key in input_ids, 
            is_out := key in output_ids,
            "input" if is_in else "output" if is_out else "hidden")

        def add_node(key):
            node = g.nodes[key]
            nid = get_nid(key)
            is_input, is_output, ntype = get_ntype(key)
            kwargs = {
                "label": f"{nid}\n{ntype}",
                "title": f"={node.activation}({node.bias:.2f} + {node.response:.2f} * {node.aggregation}(inputs))",
                "color": "green" if is_input else "blue" if is_output else "red",
                "shape": "box" if is_input else "ellipse" if is_output else "circle",
                "x": x,
                "y": y,
            }
            if is_input or is_output:
                kwargs.update({
                    "x": x + (-200 if is_input else 200),
                    "y": y + abs(key)*100 - (100 if is_input else 0),
                    "fixed": {"x": True, "y": True},
                })

            net.add_node(nid, **kwargs)

        required = required_for_output(input_ids, output_ids, g.conns.keys())

        # Add nodes
        for key in required:
            add_node(key)
        
        # Add edges
        for conn in g.conns.values():
            a, b = conn.key
            if a not in required or b not in required:
                continue
            kwargs = {
                "weight": conn.weight,
                "label": f"{conn.weight:.2f}",
                "arrows": {"to": {"enabled": True}},
                "color": "grey" if not conn.enabled else "red" if conn.weight < 0 else "#2B7CE9"
            }
            net.add_edge(get_nid(a), get_nid(b), **kwargs)
        
        x += x_step
        if x >= x_step*4:
            x = 0
            y += y_step
    
    # net.toggle_physics(False)
    net.show(out)


def plt_genome(genome, out="net.html"):
    plt_genomes({0: genome}, out)
