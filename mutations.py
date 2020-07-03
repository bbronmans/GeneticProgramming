""""
Mutations class. Contains the Option, Options and MutationOptions classes and relevant methods.
A MutationOptions object contains Options objects which contain Option objects. A MutationOptions
object is used to monitor the effects of potential mutation options of Individuals in the NSGA2
algorithm and adjusts the corresponding distribution probabilities accordingly.
"""
import dill
import numpy.random as rand

import params

rand.seed(42)

MAIN_OPTIONS = ["topology", "layer"]
TOPOLOGY_OPTIONS = ["add_layer", "add_shortcut", "remove"]
ADD_OPTIONS = params.ADD_OPTIONS
REPEAT_OPTIONS = [1, 3, 5]


def load(filename):
    """Loads Mutation settings from a Dill/Pickle file."""
    with open('./mutations/' + str(filename), 'rb') as mut_file:
        result = dill.load(mut_file)
    return result


class Option:
    """Contains info about an (mutation) option."""

    def __init__(self, name):
        self.name = name
        self.probability = 0
        self.expected_value = 1
        self.age = 1

    def update(self, value):
        """Updates the expected value and age of this (mutatation) option."""
        self.expected_value = params.MUTATION_MOMENTUM * self.expected_value \
                              + (1 - params.MUTATION_MOMENTUM) * value
        self.age = 0

    def increase_age(self):
        """Updates the age."""
        self.age += 1


class Options:
    """Contains a set of (mutation) Options."""

    def __init__(self, name, option_types):
        self.name = name
        self.options = [Option(otype) for otype in option_types]
        self.probabilities = []
        self.update_probs()

    def update(self, selected_opt, value):
        """Updates the attributes of its Option set."""
        for opt in self.options:
            if opt.name == selected_opt:
                opt.update(value)
            else:
                opt.increase_age()
        self.update_probs()

    def update_probs(self):
        """Updates the probabilities """
        sum_exp_vals = 0
        sum_ages = 0
        for opt in self.options:
            sum_exp_vals += opt.expected_value
            sum_ages += opt.age
        for opt in self.options:
            opt.probability = (1 - params.MUTATION_AGE_WEIGHT) * \
                              (opt.expected_value / (sum_exp_vals + 1e-9)) + \
                              params.MUTATION_AGE_WEIGHT * (opt.age / (sum_ages + 1e-9))
        self.probabilities = [opt.probability for opt in self.options]
        # Ensure that probabilities add up to exactly 1 to compensate rounding errors.
        self.probabilities[0] += (1 - sum(self.probabilities))

    def select_option(self):
        """Draws an option from the set of Options."""
        names = [opt.name for opt in self.options]
        name = rand.choice(names, p=self.probabilities)
        return name

    def print(self):
        """Prints current values of attributes of the set of Options."""
        print("Options " + str(self.name) + ": ")
        for opt in self.options:
            print("\t" + str(opt.name) +
                  "\t exp_val: {:03f}\tprob: {:03f}".format(opt.expected_value, opt.probability))


class MutationOptions:
    """An object containing nested Options objects. Is used to track the effectiveness of
    different mutation options and update the probabilities accordinly."""

    def __init__(self):
        # Primary options
        self.main = Options("main", MAIN_OPTIONS)

        # Secondary options
        self.topology = Options("topology", TOPOLOGY_OPTIONS)

        # Tertiary options (topology)
        self.add_layer = Options("add_layer", ADD_OPTIONS)

        # self.repeats = Options("repeats", REPEAT_OPTIONS)

    def save(self, filename):
        """Saves the MutationOptions object to file."""
        with open('./mutations/' + str(filename), 'wb') as mut_file:
            dill.dump(self, mut_file)

    def print(self):
        """Prints the current values of attributes of the Options objects."""
        self.main.print()
        self.topology.print()
        self.add_layer.print()

    def update(self, mutations, value):
        """Updates all relevant attributes of the Options objects."""
        self.main.update(mutations[0], value)
        if mutations[0] == "topology":
            self.topology.update(mutations[1], value)
            if mutations[1] == "add_layer":
                self.add_layer.update(mutations[2], value)
            # etc etc
