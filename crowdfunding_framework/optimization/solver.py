import pygad
import numpy as np

class GeneticSolver:
    """
    A generic Genetic Algorithm solver using PyGAD.
    Decoupled from the specific problem domain.
    """
    def __init__(self, problem, population_size=10, generations=30, mutation_rate=0.1, crossover_rate=0.5):
        self.problem = problem
        self.pop_size = population_size
        self.generations = generations
        # PyGAD handles mutation/crossover differently, but we map these loosely or use defaults
        # mutation_rate in PyGAD is usually mutation_probability or percent_genes
        self.mutation_rate = mutation_rate 
        self.initial_individual = None

    def set_initial_individual(self, individual):
        """
        Sets the starting point for the optimization (seeding).
        """
        self.initial_individual = individual 
        
    def fitness_func(self, ga_instance, solution, solution_idx):
        """
        Wrapper for PyGAD fitness function.
        PyGAD expects (instance, solution, idx).
        Our problem.evaluate expects just the individual (solution).
        """
        # Ensure solution is treated as a list of integers
        # PyGAD might pass numpy arrays
        ind = solution
        if isinstance(solution, np.ndarray):
            ind = solution.tolist()
            
        # Convert floats to ints if necessary (gene_space usually handles this but safety check)
        ind = [int(x) for x in ind]
        
        return self.problem.evaluate(ind)

    def run(self):
        # Define Gene Space based on problem bounds
        # problem.bounds is (low, high) inclusive for our custom solver, 
        # let's assume it means [low, high] integers.
        low, high = self.problem.bounds
        # PyGAD gene_space can be a list of ranges or a single range dict
        # Since each gene (project start week) has the same domain [1, T], we can use one range.
        # gene_space = {'low': low, 'high': high} will generate floats by default in some versions,
        # but coupled with int_genes=True (if available) or simply range list works best.
        # range(low, high+1) creates a list of valid discrete values.
        gene_space = list(range(low, high + 1))
        

        initial_pop = None
        if self.initial_individual is not None:
            # Generate full population manually
            # 1. Seed
            initial_pop = [self.initial_individual]
            
            # 2. Random rest
            # bounds are [low, high] inclusive
            low, high = self.problem.bounds
            
            # Generate remaining
            for _ in range(self.pop_size - 1):
                # Random individual
                rand_ind = np.random.randint(low, high + 1, size=self.problem.N).tolist()
                initial_pop.append(rand_ind)
                
        ga_instance = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=max(2, int(self.pop_size * 0.2)), # 20% parents
            fitness_func=self.fitness_func,
            sol_per_pop=self.pop_size,
            num_genes=self.problem.N,
            gene_space=gene_space,
            gene_type=int, # Force integer genes
            mutation_type="random",
            mutation_probability=self.mutation_rate,
            initial_population=initial_pop, # Pass seeded population
            suppress_warnings=True
        )
        
        ga_instance.run()
        
        # Extract best solution
        best_solution, best_fitness, solution_idx = ga_instance.best_solution()
        
        # Convert numpy array back to list for compatibility
        if isinstance(best_solution, np.ndarray):
            best_solution = best_solution.tolist()
        best_solution = [int(x) for x in best_solution]
        
        # Helper to get history (fitness over generations)
        # PyGAD stores this in best_solutions_fitness
        history = ga_instance.best_solutions_fitness
        
        return best_solution, best_fitness, history
