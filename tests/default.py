from neat.blueprints import *


default_blueprint = GenerationalBP(

    # Population blueprint
    population = PopulationBP(
        
        # General parameters
        pop_size = 150,

        # Species blueprint
        species = SpeciesBP(
            # Dynamic compatibility threshold
            compat_threshold_initial = 3.0,
            compat_threshold_modifier = 0.1,
            compat_threshold_min = 0.1,
            target_num_species = 50,

            # Stagnation
            species_fitness_func = "mean",
            max_stagnation = 15,
            species_elitism = 0,
            reset_on_extinction = True,
        ),

        # Genome blueprint
        genome = GenomeBP(

            # Node options
            node = NodeBP(
                activation = StringBP(
                    default="tanh",
                    mutate_rate=0.05,
                    options=["sigmoid", "tanh", "sin", "gauss", "identity"]
                ),
                aggregation = StringBP(
                    default="sum",
                    mutate_rate=0.05,
                    options=["sum", "product", "max", "min"]
                ),
                bias = FloatBP(
                    init_mean=0.0,
                    init_stdev=1.0,
                    max_value=30.0,
                    min_value=-30.0,
                    mutate_power=0.5,
                    mutate_rate=0.7,
                    replace_rate=0.1
                ),
                response = FloatBP(
                    init_mean=1.0,
                    init_stdev=0.0,
                    max_value=30.0,
                    min_value=-30.0,
                    mutate_power=0.0,
                    mutate_rate=0.0,
                    replace_rate=0.0
                ),
            ),

            # Connection options
            conn = ConnBP(
                enabled = BoolBP(
                    default=True,
                    mutate_rate=0.01
                ),
                weight = FloatBP(
                    init_mean=0.0,
                    init_stdev=1.0,
                    max_value=30.0,
                    min_value=-30.0,
                    mutate_power=0.5,
                    mutate_rate=0.8,
                    replace_rate=0.1
                ),
            ),

            # Network initialization options
            num_inputs = 2,
            num_outputs = 1,

            # Network mutation options
            conn_add_prob = 0.5,
            conn_delete_prob = 0.5,
            node_add_prob = 0.2,
            node_delete_prob = 0.2,

            # Genome compatibility options
            compatibility_disjoint_coefficient = 1.0,
            compatibility_weight_coefficient = 0.5,

            # Structural mutations
            single_structural_mutation = False,
            structural_mutation_surer = False,
        ),

    ),

)
