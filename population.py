"""
Population class. Contains the Population class and relevant methods.
In the NSGA2 algorithm, a population is a set of individuals.
"""""
import dill

import individual


def load(filename):
    """Loads a Population from a Dill/Pickle file."""
    with open('./pops/' + str(filename), 'rb') as pop_file:
        generation, list_of_inds = dill.load(pop_file)

        result = Population()
        result.generation = generation
        for ind_data in list_of_inds:
            new_ind = individual.Individual()
            genes, fitness1, fitness2, age = ind_data
            new_ind.genes = genes
            new_ind.fitness1 = fitness1
            new_ind.fitness2 = fitness2
            new_ind.age = age
            new_ind.keras_model = new_ind.create_keras_model()
            result.individuals.append(new_ind)
    return result


def crowding_distance_assigment(individuals):
    """Determines for each individual the distance to its nearest neighbours.
    Subroutine of NSGA2."""
    for i in individuals:
        i.distance = 0
    individuals.sort(key=lambda x: x.fitness1, reverse=False)
    min_f1 = individuals[0].fitness1
    max_f1 = individuals[-1].fitness1
    if min_f1 != max_f1:
        individuals[0].distance = 9999999999999
        individuals[-1].distance = 9999999999999
        for idx, i in enumerate(individuals):
            # if idx != 0 and idx != len(individuals) - 1:
            if idx not in [0, len(individuals) - 1]:
                i.distance += (individuals[idx + 1].fitness1 -
                               individuals[idx].fitness1) / (max_f1 - min_f1)

    # TODO refactor duplicate code for fitness2
    individuals.sort(key=lambda x: x.fitness2, reverse=False)
    min_f2 = individuals[0].fitness2
    max_f2 = individuals[-1].fitness2
    if min_f2 == max_f2:
        return
    individuals[0].distance = 9999999999999
    individuals[-1].distance = 9999999999999
    for idx, i in enumerate(individuals):
        # if idx != 0 and idx != len(individuals) - 1:
        if idx not in [0, len(individuals) - 1]:
            i.distance += (individuals[idx + 1].fitness2 -
                           individuals[idx].fitness2) / (max_f2 - min_f2)


class Population:
    """"In the NSGA2 algorithm, a population is a set of individuals."""

    def __init__(self):
        self.individuals = []
        self.generation = -1

    def fill_population(self, pop_size):
        """Initializes 'pop_size' (int) of Individual objects."""
        self.individuals = [individual.Individual() for _ in range(pop_size)]

    def print_stats(self):
        """"Prints stats."""
        for ind in self.individuals:
            print([ind.fitness1, ind.fitness2, ind.age, ind.rank])

    def get_fittest(self):
        """Returns the Individuals with the best (lowest) fitness1 and fitness2."""
        best_f1, best_f2 = self.individuals[0], self.individuals[0]
        for ind in self.individuals[1:]:
            if ind.fitness1 < best_f1.fitness1:
                best_f1 = ind
            if ind.fitness2 < best_f2.fitness2:
                best_f2 = ind
        return best_f1, best_f2

    def worst_fitness(self):
        """Returns the values of the Individuals with the best (lowest) fitness1 and fitness2."""
        worst_f1 = self.individuals[0].fitness1
        worst_f2 = self.individuals[0].fitness2
        for ind in self.individuals[1:]:
            if ind.fitness1 > worst_f1:
                worst_f1 = ind.fitness1
            if ind.fitness2 > worst_f2:
                worst_f2 = ind.fitness2
        return worst_f1, worst_f2

    def save(self, filename):
        """Saves this Population to a Dill/Pickle file."""
        best_f1, _ = self.get_fittest()
        best_val_acc = 1 - best_f1.fitness1
        with open('./pops/' + str(filename) + "_valacc_{:.3f}".format(best_val_acc), 'wb') as file:
            list_of_inds = []
            for ind in self.individuals:
                temp_ind = ind.deepcopy()
                for lay in temp_ind.genes:
                    lay.keras_tensor = -1
                list_of_inds.append([temp_ind.genes, ind.fitness1, ind.fitness2, ind.age])
            result = [self.generation, list_of_inds]
            dill.dump(result, file)

    def deepcopy(self):
        """Returns a deepcopy of this Population."""
        pop_copy = Population()
        pop_copy.__dict__ = self.__dict__.copy()
        pop_copy.individuals = []
        pop_copy.generation = self.generation
        for ind in self.individuals:
            pop_copy.individuals.append(ind.deepcopy(true_copy=False))
        return pop_copy

    def fast_non_dominated_sort(self):
        """Divides population P into pareto fronts. A subroutine of NSGA2."""
        fronts = [[]]
        for p in self.individuals:
            p.S = []
            p.n = 0
            # Update p.S and n_p
            for q in self.individuals:
                if p.dominates(q):
                    p.S.append(q)
                elif q.dominates(p):
                    p.n += 1

            # Check if p belongs to the first front
            if p.n == 0:
                p.rank = 0
                fronts[0].append(p)

        # Determine all fronts and ranks
        i = 0
        while len(fronts[i]) > 0:
            Q = []
            for p in fronts[i]:
                for q in p.S:
                    q.n -= 1
                    if q.n == 0:
                        q.rank = i + 1
                        Q.append(q)
            i += 1
            fronts.append(Q)
        return fronts
