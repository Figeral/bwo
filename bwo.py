import numpy as np

class BlackWidowOptimization:
    def __init__(self, func, bounds, dim, pop_size=30, max_iter=100,
                 procreating_rate=0.6, cannibalism_rate=0.44, mutation_rate=0.4):
        self.func = func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.procreating_rate = procreating_rate
        self.cannibalism_rate = cannibalism_rate
        self.mutation_rate = mutation_rate

        # Initialize population
        self.pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        
        # Sort population
        sort_idx = np.argsort(self.fitness)
        self.pop = self.pop[sort_idx]
        self.fitness = self.fitness[sort_idx]
        
        self.best_solution = self.pop[0].copy()
        self.best_fitness = self.fitness[0]
        
        self.curr_iter = 0

    def get_state(self):
        return {
            'pop': self.pop.copy(),
            'best_solution': self.best_solution.copy(),
            'best_fitness': self.best_fitness,
            'curr_iter': self.curr_iter
        }

    def procreate_and_cannibalize(self):
        # Calculate number of reproducing variables (spiders)
        nr = int(self.procreating_rate * self.pop_size)
        
        # Ensure nr is even to form pairs
        if nr % 2 != 0:
            nr = max(2, nr - 1)
            
        if nr < 2:
            return np.array([])
            
        # Normally in BWO, parents mate randomly, but better fitness individuals are chosen.
        # Since population is sorted, we can pick the best `nr` individuals to be parents
        parents = self.pop[:nr].copy()
        parents_fitness = self.fitness[:nr].copy()
        
        # Randomly pair parents
        indices = np.random.permutation(nr)
        new_pop = []
        
        for i in range(0, nr, 2):
            idx1, idx2 = indices[i], indices[i+1]
            p1, p2 = parents[idx1], parents[idx2]
            
            # Crossover to create multiple offspring
            num_offspring = max(4, 2 * self.dim) # Arbitrary number of offspring, usually related to dimension
            offspring = []
            for _ in range(num_offspring // 2):
                alpha = np.random.uniform(0, 1, self.dim)
                y1 = alpha * p1 + (1 - alpha) * p2
                y2 = alpha * p2 + (1 - alpha) * p1
                offspring.extend([y1, y2])
            
            offspring = np.array(offspring)
            # Clip bounds bounds
            offspring = np.clip(offspring, self.bounds[0], self.bounds[1])
            
            # Evaluate offspring fitness
            offspring_fitness = np.apply_along_axis(self.func, 1, offspring)
            
            # Cannibalism
            # 1. Sexual cannibalism: best parent survives, worst dies.
            if parents_fitness[idx1] < parents_fitness[idx2]:
                surviving_parent = p1
            else:
                surviving_parent = p2
                
            # 2. Sibling cannibalism: keep the strongest according to Cannibalism Rate (CR)
            sort_off_idx = np.argsort(offspring_fitness)
            offspring = offspring[sort_off_idx]
            
            num_survivors = max(1, int(len(offspring) * (1 - self.cannibalism_rate)))
            surviving_offspring = offspring[:num_survivors]
            
            new_pop.append(surviving_parent)
            new_pop.extend(surviving_offspring)
            
        return np.array(new_pop)
        
    def mutate(self):
        nm = int(self.mutation_rate * self.pop_size)
        if nm == 0:
            return np.array([])
            
        # Select random individuals to mutate from the entire pop
        indices = np.random.choice(self.pop_size, nm, replace=False)
        mutants = self.pop[indices].copy()
        
        # Mutate one random variable per mutant
        for i in range(nm):
            dim_to_mutate = np.random.randint(self.dim)
            mutants[i, dim_to_mutate] = np.random.uniform(self.bounds[0], self.bounds[1])
            
        return mutants

    def step(self):
        if self.curr_iter >= self.max_iter:
            return False
            
        # 1. Procreation and Cannibalism
        new_pop_from_reproduction = self.procreate_and_cannibalize()
        
        # 2. Mutation
        mutants = self.mutate()
        
        # 3. Update Population
        valid_pops = [self.pop]
        if len(new_pop_from_reproduction) > 0:
            valid_pops.append(new_pop_from_reproduction)
        if len(mutants) > 0:
            valid_pops.append(mutants)
            
        combined_pop = np.vstack(valid_pops)
            
        # Clip
        combined_pop = np.clip(combined_pop, self.bounds[0], self.bounds[1])
        combined_fitness = np.apply_along_axis(self.func, 1, combined_pop)
        
        # Sort and select best pop_size individuals
        sort_idx = np.argsort(combined_fitness)
        self.pop = combined_pop[sort_idx][:self.pop_size]
        self.fitness = combined_fitness[sort_idx][:self.pop_size]
        
        # Update best solution
        if self.fitness[0] < self.best_fitness:
            self.best_fitness = self.fitness[0]
            self.best_solution = self.pop[0].copy()
            
        self.curr_iter += 1
        return True
