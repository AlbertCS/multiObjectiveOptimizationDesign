class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, elitism_count):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count

    def init_population(self, chromosome_length):
        pass

    def calc_fitness(self, individual):
        pass

    def eval_population(self, population):
        pass

    def is_termination_condition_met(self, generations_count, max_generations):
        pass

    def select_parent(self, population):
        pass

    def crossover_population(self, population):
        pass

    def mutate_population(self, population):
        pass

    def create_new_population(self, population):
        pass

    def run(self):
        pass
