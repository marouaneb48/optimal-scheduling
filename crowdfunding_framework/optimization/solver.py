import pygad
import numpy as np

class GeneticSolver:
    """
    A Genetic Algorithm solver using PyGAD with schedule-aware operators.
    """
    def __init__(self, problem, population_size=80, generations=200, mutation_rate=0.15, crossover_rate=0.8):
        self.problem = problem
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.initial_individual = None

    def set_initial_individual(self, individual):
        self.initial_individual = individual

    def fitness_func(self, ga_instance, solution, solution_idx):
        ind = solution
        if isinstance(solution, np.ndarray):
            ind = solution.tolist()
        ind = [int(x) for x in ind]
        return self.problem.evaluate(ind)

    def _build_initial_population(self):
        """
        Builds a population seeded around the original schedule.
        - Slot 0: exact original
        - ~60% of pop: small perturbations (shift 10-30% of projects by ±1-2 weeks)
        - ~20% of pop: medium perturbations (shift 30-60% of projects by ±1-3 weeks)
        - ~20% of pop: random for diversity
        """
        low, high = self.problem.bounds
        orig = np.array(self.initial_individual)
        pop = [orig.tolist()]

        n_small = int(self.pop_size * 0.6)
        n_medium = int(self.pop_size * 0.2)
        n_random = self.pop_size - 1 - n_small - n_medium

        # Small perturbations: shift 10-30% of genes by ±1-2
        for _ in range(n_small):
            child = orig.copy()
            n_mutate = np.random.randint(max(1, int(0.1 * len(child))), max(2, int(0.3 * len(child))) + 1)
            indices = np.random.choice(len(child), size=n_mutate, replace=False)
            shifts = np.random.choice([-2, -1, 1, 2], size=n_mutate)
            child[indices] = np.clip(child[indices] + shifts, low, high)
            pop.append(child.tolist())

        # Medium perturbations: shift 30-60% of genes by ±1-3
        for _ in range(n_medium):
            child = orig.copy()
            n_mutate = np.random.randint(max(1, int(0.3 * len(child))), max(2, int(0.6 * len(child))) + 1)
            indices = np.random.choice(len(child), size=n_mutate, replace=False)
            shifts = np.random.choice([-3, -2, -1, 1, 2, 3], size=n_mutate)
            child[indices] = np.clip(child[indices] + shifts, low, high)
            pop.append(child.tolist())

        # Random for diversity
        for _ in range(n_random):
            pop.append(np.random.randint(low, high + 1, size=self.problem.N).tolist())

        return pop

    def run(self):
        low, high = self.problem.bounds
        gene_space = list(range(low, high + 1))

        initial_pop = None
        if self.initial_individual is not None:
            initial_pop = self._build_initial_population()

        # Vectorized mutation: shift genes by small amounts
        def mutation_func(offspring, ga_instance):
            mask = np.random.random(offspring.shape) < self.mutation_rate
            shifts = np.random.choice([-2, -1, 1, 2], size=offspring.shape)
            offspring = offspring + mask * shifts
            np.clip(offspring, low, high, out=offspring)
            return offspring

        ga_instance = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=max(2, int(self.pop_size * 0.3)),
            fitness_func=self.fitness_func,
            sol_per_pop=self.pop_size,
            num_genes=self.problem.N,
            gene_space=gene_space,
            gene_type=int,
            crossover_type="uniform",
            crossover_probability=self.crossover_rate,
            mutation_type=mutation_func,
            initial_population=initial_pop,
            suppress_warnings=True,
            keep_elitism=max(2, int(self.pop_size * 0.05)),
            parent_selection_type="tournament",
            K_tournament=5,
            save_solutions=False,
            save_best_solutions=False,
        )

        ga_instance.run()

        best_solution, best_fitness, solution_idx = ga_instance.best_solution()

        if isinstance(best_solution, np.ndarray):
            best_solution = best_solution.tolist()
        best_solution = [int(x) for x in best_solution]

        history = ga_instance.best_solutions_fitness

        return best_solution, best_fitness, history
