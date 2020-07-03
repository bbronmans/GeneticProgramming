# pylint: disable=wrong-import-position
""""
Main class. Runs the NSGA2 algorithm (https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf)
and contains the evaluation environment for the Convolutional Neural Networks.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['PYTHONHASHSEED'] = str(0)
import gc
import numpy as np
import keras as K
from tensorflow import ConfigProto
from tensorflow import logging
from keras.utils import np_utils
from keras.datasets import cifar100
from keras.utils import plot_model
from tqdm import tqdm

np.random.seed(42)
import tensorflow as tf

tf.set_random_seed(42)
import mutations
import population
import params
import preprocess_img


class SaveBestCallback(K.callbacks.Callback):
    """Callback that saves only the model with the highest val acc of all evaluations."""

    def __init__(self):
        super(SaveBestCallback, self).__init__()
        self.filepath = './GA2/{val_acc:.3f}_{epoch:02d}.h5'
        self.best_val_acc = -1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        current_val_acc = logs.get('val_acc')
        if current_val_acc > self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.model.save(filepath, overwrite=True)
        gc.collect()


def init_eval_environment():
    """Prepares the dataset and datagenerators used to train a model."""
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    logging.set_verbosity(logging.ERROR)  # Supress all kind of deprecation warnings

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Preprocess
    mean = np.mean(x_train, axis=(0, 1, 2))  # Per channel normalization
    std = np.std(x_train, axis=(0, 1, 2))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    y_train = np_utils.to_categorical(y_train, params.NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test, params.NUM_CLASSES)

    # Data input
    class Dataset:
        """Class required for retrieving and preprocessing a single image."""

        def __init__(self, img_data, label_data, augmentation=False):
            self.img_data = img_data
            self.label_data = label_data
            self.augmentation = augmentation

        def __getitem__(self, i):
            label = self.label_data[i]
            image = self.img_data[i]
            if self.augmentation:
                image = preprocess_img.augment_img(image)
            return image, label

        def __len__(self):
            return len(self.label_data)

    class Dataloader(K.utils.Sequence):
        """Class required for iterating over batches of processed images."""

        def __init__(self, dataset, batch_size, shuffle=False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indexes = np.arange(len(dataset))
            self.on_epoch_end()

        def __getitem__(self, i):
            start = i * self.batch_size
            stop = (i + 1) * self.batch_size
            data = []
            for j in range(start, stop):
                data.append(self.dataset[j])

            # Transpose list of lists
            batch = [np.stack(samples, axis=0) for samples in zip(*data)]
            return batch

        def __len__(self):
            """Denotes the number of batches per epoch"""
            return len(self.indexes) // self.batch_size

        def on_epoch_end(self):
            """Callback function to shuffle indexes each epoch"""
            if self.shuffle:
                self.indexes = np.random.permutation(self.indexes)

    train_dataset = Dataset(x_train, y_train, augmentation=True)
    val_dataset = Dataset(x_test, y_test)

    train_dataloader = Dataloader(train_dataset, batch_size=params.BATCH_SIZE, shuffle=True)
    val_dataloader = Dataloader(val_dataset, batch_size=params.BATCH_SIZE, shuffle=False)
    return train_dataloader, val_dataloader


def evaluate_individual(individual, train_dataloader, val_dataloader, save_best_callback):
    """Trains the Keras model of an Individual to evaluate its fitness."""
    individual.keras_model = individual.create_keras_model()
    model = individual.keras_model
    opt_sgd = K.optimizers.SGD(lr=params.LEARNING_RATE,
                               momentum=params.LR_MOMENTUM,
                               decay=params.LR_DECAY,
                               nesterov=params.NESTEROV)

    callbacks = [
        K.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.005, patience=25),
        save_best_callback
    ]
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt_sgd,
                  metrics=['accuracy'])
    history = model.fit_generator(train_dataloader,
                                  steps_per_epoch=params.TRAIN_SAMPLES // params.BATCH_SIZE,
                                  epochs=params.TRAIN_EPOCHS,
                                  validation_data=val_dataloader,
                                  validation_steps=params.VAL_SAMPLES // params.BATCH_SIZE,
                                  callbacks=callbacks,
                                  verbose=0)
    val_acc = history.history['val_acc']
    best_val_acc = np.amax(val_acc)
    individual.fitness1 = 1 - best_val_acc
    individual.fitness2 = model.count_params()


def compare(ind1, ind2):
    """Returns the relative rank of two Individuals in the same NSGA2 front."""
    if ind1.rank < ind2.rank or (ind1.rank == ind2.rank and
                                 ind1.distance - ind1.fitness1 > ind2.distance - ind2.fitness1):
        return 1
    return -1


def nsga2_step(old_p, train_dataloader, val_dataloader, save_best_callback, mutation_options):
    """Performs one epoch of an NSGA2 epoch. Sorts input population 'old_p'
    to select new population 'new_p', creates and evaluates mutated offspring 'new_q'.
    Performs one epoch of the main routine of NSGA2. """
    # Select parents
    print("Enter nsga2step")
    fronts = old_p.fast_non_dominated_sort()
    new_p = population.Population()
    i = 0

    # Add all individuals from the best fronts as long as the entire front fits
    while len(new_p.individuals) + len(fronts[i]) < params.POPULATION_SIZE:
        print("Size front " + str(i) + ": " + str(len(fronts[i])))
        fronts[i].sort(key=lambda x: x.fitness1, reverse=False)
        # population.crowding_distance_assigment(fronts[i])  # Actually belongs below while loop
        new_p.individuals.extend(fronts[i])
        i += 1
    # Add the best ranked individuals from the front that only partially fits
    fronts[i].sort(key=lambda x: x.fitness1, reverse=False)
    # sorted_front = sorted(fronts[i], key=functools.cmp_to_key(compare), reverse=True)
    new_p.individuals.extend(fronts[i][:params.POPULATION_SIZE - len(new_p.individuals)])
    print("Parents")
    new_p.print_stats()

    # Update model size restrictions
    best_f1, _ = new_p.get_fittest()
    worst_f1, _ = new_p.worst_fitness()

    # Update administration
    new_p.generation = old_p.generation + 1
    for ind in new_p.individuals:
        ind.age += 1

    # Create offspring
    new_q = new_p.deepcopy()
    for ind in tqdm(new_q.individuals):
        n_mutations = np.random.choice(mutations.REPEAT_OPTIONS)
        while len(ind.mutations) == 0:
            for _ in range(n_mutations):
                ind.mutate(mutation_options)
        ind.trim()
        evaluate_individual(ind, train_dataloader, val_dataloader, save_best_callback)
        if ind.fitness1 > worst_f1:
            new_q.individuals.remove(ind)
        for parent in new_p.individuals:
            if ind.dominates(parent):
                ind.dom_count += 1

        print("f1, f2: " + str(ind.fitness1) + ", " + str(ind.fitness2))
        print("Dom count: " + str(ind.dom_count))
        print("Mutations: " + str(ind.mutations))
        if ind.fitness1 < best_f1.fitness1 and n_mutations == 1:
            mutation_options.update(ind.mutations, ind.dom_count + 5)
        elif n_mutations == 1:
            mutation_options.update(ind.mutations, ind.dom_count)
            mutation_options.print()
        mutation_options.save("latest")
    return new_p, new_q


def save_best_models(best_f1_ind, best_f2_ind, generation):
    """Saves the model with the highest val acc and the model with the smallest model size"""
    best_f1_ind.keras_model = best_f1_ind.create_keras_model()
    best_f1_ind.keras_model.save(
        "./GA2/best_val_acc/{:03d}_{:.3f}".format(generation, 1 - best_f1_ind.fitness1))
    best_f2_ind.keras_model = best_f2_ind.create_keras_model()
    best_f2_ind.keras_model.save(
        "./GA2/best_size/{:03d}_{:.3f}".format(generation, best_f2_ind.fitness1))


def print_best_models(best_f1_ind, best_f2_ind, generation):
    """Visualizes the model with the highest val acc and the model with the smallest model size"""
    plot_model(best_f1_ind.keras_model,
               to_file="./best_val_acc/{:03d}_{:.3f}_{:06d}.png".format(generation,
                                                                        1 - best_f1_ind.fitness1,
                                                                        best_f1_ind.fitness2),
               show_shapes=True)
    plot_model(best_f2_ind.keras_model,
               to_file="./best_size/{:03d}_{:.3f}_{:06d}.png".format(generation,
                                                                     1 - best_f2_ind.fitness1,
                                                                     best_f2_ind.fitness2),
               show_shapes=True)
    print("Best val_acc model ({:.3f}):".format(1 - best_f1_ind.fitness1))
    best_f1_ind.print_genes()


def main():
    """Main loop."""

    # Prepare evaluation environment
    train_dataloader, val_dataloader = init_eval_environment()
    save_best_callback = SaveBestCallback()

    # Initialize the mutation operations
    load_mut = False
    load_mut_file = "latest"
    mutation_options = mutations.MutationOptions()
    if load_mut:
        mutation_options = mutations.load(load_mut_file)

    # TODO remove, temp code
    mutation_options.main.options[0].expected_value += 10
    mutation_options.main.update_probs()
    mutation_options.topology.options[0].expected_value += 10
    mutation_options.topology.update_probs()
    # mutation_options.add_layer.options[-2].expected_value += 10
    # mutation_options.add_layer.update_probs()

    # Initialize the population
    pop = population.Population()
    load_pop = True
    load_pop_file = "pop_gen021_valacc_0.537"
    # load_pop_file = "pop_gen083_valacc_0.146"

    if load_pop:
        pop = population.load(load_pop_file)
    else:
        pop.fill_population(params.POPULATION_SIZE)
        for ind in tqdm(pop.individuals):
            while ind.fitness1 < params.MAX_INIT_FITNESS1:
                ind.genes = []
                ind.init_architecture()
                ind.fitness1 = 0.96
                ind.fitness2 = ind.keras_model.count_params()
                for i in range(params.INITIAL_MUTATIONS):
                    ind.mutate(mutation_options)
                if params.INITIAL_MUTATIONS > 0:
                    evaluate_individual(ind, train_dataloader, val_dataloader, save_best_callback)

    # Plot current best individuals
    best_f1_ind, best_f2_ind = pop.get_fittest()
    print_best_models(best_f1_ind, best_f2_ind, pop.generation)

    parents = pop
    pop.save("pop_gen-1")

    # Run the main NSGA2 loop
    for nsga_epoch in range(params.NSGA_EPOCHS):
        # Create new offspring and select new parents
        parents, children = nsga2_step(parents,
                                       train_dataloader,
                                       val_dataloader,
                                       save_best_callback,
                                       mutation_options)
        print("Finished epoch " + str(nsga_epoch))
        print("Children")
        children.print_stats()

        # Save and plot best individuals
        parents.individuals.extend(children.individuals)
        parents.save("pop_gen{:03d}".format(parents.generation))
        mutation_options.save("mut_gen{:03d}".format(parents.generation))
        best_f1_ind, best_f2_ind = parents.get_fittest()
        save_best_models(best_f1_ind, best_f2_ind, parents.generation)
        print_best_models(best_f1_ind, best_f2_ind, parents.generation)


if __name__ == "__main__":
    main()
