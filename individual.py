"""
Individual class. Contains the Individual class and relevant methods.
In the NSGA2 algorithm, an individual is a unique solution. In this case,
an individual is a neural network architecture.
"""""
import dill
import keras as K
import numpy.random as rand
import tensorflow as tf

import layer
import params

tf.set_random_seed(42)
rand.seed(42)


def load(filename):
    """Loads an Individual from a Dill/Pickle file."""
    with open('./GA2/' + str(filename), 'rb') as ind_file:
        ind_data = dill.load(ind_file)
        new_ind = Individual()
        genes, fitness1, fitness2, age = ind_data
        new_ind.genes = genes
        new_ind.fitness1 = fitness1
        new_ind.fitness2 = fitness2
        new_ind.age = age
        new_ind.keras_model = new_ind.create_keras_model()
    return new_ind


class Individual:
    """"In the NSGA2 algorithm, an individual is a unique solution. In this case,
    an individual is a neural network architecture."""

    def init_architecture(self):
        """Creates a minimum viable model."""
        conv_layer = layer.ConvLayer(layer_in=None, n_channels=32, lsubtype="conv", stride=2)
        self.genes.append(conv_layer)
        mbconv_layer = layer.MBConvLayer(layer_in=conv_layer)
        self.genes.append(mbconv_layer)
        global_pool_layer = layer.GlobalPoolingLayer(layer_in=mbconv_layer)
        self.genes.append(global_pool_layer)
        dense_layer = layer.DenseLayer(layer_in=global_pool_layer, n_channels=params.NUM_CLASSES,
                                       final_layer=True)
        self.genes.append(dense_layer)

    def __init__(self):
        self.genes = []
        self.fitness1 = 2
        self.fitness2 = params.MAX_MODEL_PARAMS + 1
        self.age = 0
        self.init_architecture()
        self.keras_model = self.create_keras_model()

        self.S = []  # Used in fast_non_dominated_sort  # TODO change into dict?
        self.n = 0  # Used in fast_non_dominated_sort
        self.rank = -1  # Used in fast_non_dominated_sort
        self.distance = -1  # Used in crowding_distance_assigment

        self.mutations = []
        self.dom_count = 0

    def dominates(self, ind2):
        """Checks whether this individual dominates 'ind2' according to NSGA2."""
        return self.fitness1 < ind2.fitness1 and self.fitness2 <= ind2.fitness2

    def save(self, filename):
        """Saves the individual to a Dill/Pickle file."""
        with open('./GA/saved_models/' + str(filename), 'wb') as file_writer:
            temp = self.deepcopy()
            for lay in temp.genes:
                lay.keras_layer = -1
                lay.keras_tensor = -1
            dill.dump([temp.genes, self.fitness1, self.fitness2, self.age], file_writer)

    def deepcopy(self, true_copy=False):
        """Returns a deepcopy of the individual."""
        ind_copy = Individual()
        ind_copy.__dict__ = self.__dict__.copy()
        ind_copy.genes = []
        if not true_copy:
            ind_copy.fitness1, ind_copy.fitness2 = 2, params.MAX_MODEL_PARAMS + 1
            ind_copy.rank = -1
            ind_copy.age, ind_copy.dom_count = 0, 0
            ind_copy.mutations = []

        # Copy first layer, then other layers and manually reference their input layers
        ind_copy.genes.append(self.genes[0].deepcopy())
        for lay in self.genes[1:]:
            lay_copy = lay.deepcopy()
            layer_in_idx = self.genes.index(lay.layer_in)
            lay_copy.layer_in = ind_copy.genes[layer_in_idx]
            if lay.ltype == "merge":
                layer2_in_idx = self.genes.index(lay.layer2_in)
                lay_copy.layer2_in = ind_copy.genes[layer2_in_idx]

            ind_copy.genes.append(lay_copy)
        ind_copy.update_layer_dims()
        ind_copy.keras_model = ind_copy.create_keras_model()
        return ind_copy

    def print_genes(self):
        """Prints an overview of the model architecture."""
        for lay in self.genes:
            lay.print()

    def create_keras_model(self):
        """Create a Keras model based on the current gene set."""
        # Clear Keras environment and update the required Keras layers
        K.backend.clear_session()
        tf.set_random_seed(42)  # Reset random seed

        model_input = K.layers.Input(shape=[params.INPUT_IMG_SIZE, params.INPUT_IMG_SIZE, 3])
        # Update the required Keras tensors
        self.genes[0].keras_tensor = self.genes[0].to_keras(model_input)
        for lay in self.genes[1:]:
            if lay.ltype == "merge":
                lay.keras_tensor = lay.to_keras([lay.layer_in.keras_tensor,
                                                 lay.layer2_in.keras_tensor])
            else:
                lay.keras_tensor = lay.to_keras(lay.layer_in.keras_tensor)

        # Create the Keras model
        model_output = self.genes[-1].keras_tensor
        model = K.models.Model(inputs=[model_input], outputs=[model_output])
        return model

    def mutate(self, mutation_options):
        """Mutate either the topology or a random layer."""
        main_type = mutation_options.main.select_option()
        self.mutations.append(main_type)
        if main_type == "topology":
            print("Mutate topology")
            self.mutate_topology(mutation_options)
        elif main_type == "layer":
            print("Mutate layer")
            self.mutate_layer()
        else:
            raise Exception("Mutationtype " + str(main_type) +
                            " should be one of mutation.MAIN_OPTIONS")
        self.update_layer_dims()

    def mutate_topology(self, mutation_options):
        """Mutate the topology by adding or removing a random layer or shortcut."""
        topology_type = mutation_options.topology.select_option()
        self.mutations.append(topology_type)
        if topology_type == "add_layer":
            print("Add layer")
            self.add_random_layer(mutation_options)
        elif topology_type == "add_shortcut":
            print("Add shortcut")
            self.add_random_shortcut()
        elif topology_type == "remove":
            print("Remove layer")
            self.remove_random_layer()
        else:
            raise Exception("Mutationtype " + str(topology_type) +
                            " should be one of mutation.TOPOLOGY_OPTIONS")

    def add_random_layer(self, mutation_options):
        """Adds a random new layer, randomly inserted sequentially."""
        # Find all suitable (ltype, location) options for a new layer
        options = []
        for lay in self.genes[:-2]:
            suitable_out_ltypes = layer.suitable_output_ltypes(lay.ltype)
            output_layers, output_merge_layers = self.get_output_layers(lay)
            output_layers.extend(output_merge_layers)
            for out_lay in output_layers:
                suitable_in_ltypes = layer.suitable_input_ltypes(out_lay.ltype)
                for ltype in params.ADD_OPTIONS:
                    if ltype in suitable_in_ltypes and ltype in suitable_out_ltypes:
                        options.append([ltype, lay, out_lay])

        # Select a valid option and insert it
        option = self.select_valid_layer_option(options, mutation_options)
        new_layer, in_layer, out_layer, new_ltype = option
        if new_layer is not None:
            self.mutations.append(new_ltype)
            self.genes.insert(self.genes.index(in_layer) + 1, new_layer)
            out_layer.layer_in = new_layer
            print("New layer added:")
            new_layer.print()
        else:
            self.mutations.clear()

    def add_random_shortcut(self):
        """Adds a random identity shortcut."""
        # List all suitable (ltype, location) options for a new layer
        options = []
        for idx, lay_in in enumerate(self.genes[:-2]):
            for lay in self.genes[idx + 2:]:
                if lay.dims_in == lay_in.dims_out():
                    options.append((lay_in, lay))

        if len(options) == 0:
            print("No shortcut added")
            return

        # Add a random shortcut
        rand.shuffle(options)
        while len(options) > 0:
            in_layer, out_layer = options[0]
            merge_layer = layer.create_merge_layer(layer1_in=out_layer.layer_in, layer2_in=in_layer)
            merge_layer_idx = max(self.genes.index(in_layer),
                                  self.genes.index(out_layer.layer_in)) + 1
            self.genes.insert(merge_layer_idx, merge_layer)
            out_layer.layer_in = merge_layer
            if self.has_valid_dims():
                self.trim()
                print("Shortcut added:")
                merge_layer.print()
                return
            # Remove the shortcut if not valid
            out_layer.layer_in = merge_layer.layer_in
            self.genes.remove(merge_layer)
            options.pop(0)
        print("No shortcut added")

    def remove_random_layer(self):
        """Removes a random layer."""
        # List all suitable layers to remove
        options = []
        for lay in self.genes[1:-2]:
            options.append(lay)

        # Remove a random layer
        rand.shuffle(options)
        while len(options) > 0:
            del_layer = options[0]
            out_layers, merge_out_layers = self.get_output_layers(del_layer)
            temp = del_layer.layer_in
            del_layer_idx = self.genes.index(del_layer)
            self.genes.remove(del_layer)
            for out_lay in out_layers:
                out_lay.layer_in = temp
            for merge_out_lay in merge_out_layers:
                merge_out_lay.layer2_in = temp
            if self.has_valid_dims():
                self.trim()
                return
            # Undo the removal if not valid
            self.genes.insert(del_layer_idx, del_layer)
            for out_lay in out_layers:
                out_lay.layer_in = del_layer
            for merge_out_lay in merge_out_layers:
                merge_out_lay.layer2_in = del_layer
            options.remove(del_layer)
        print("No layer removed")

    def mutate_layer(self):
        """Selects a random mutable layer and mutate it."""
        mutable_layers = [lay for lay in self.genes[:-3] if lay.is_mutable]
        rand.shuffle(mutable_layers)
        while len(mutable_layers) > 0:
            lay = mutable_layers[0]
            backup = lay.__dict__.copy()
            lay.mutate()
            if self.has_valid_dims():
                return
            lay.__dict__ = backup
            mutable_layers.remove(lay)
        print("No layer mutation performed")

    def update_layer_dims(self):
        """Updates the output dimensions of all layers."""
        self.genes[0].dims_in = [params.INPUT_IMG_SIZE, params.INPUT_IMG_SIZE, 3]
        self.genes[0].layer_in = None
        for lay in self.genes[1:]:
            lay.dims_in = lay.layer_in.dims_out()

    def get_output_layers(self, layer_in):
        """Returns all layers that receive input from 'layer_in'."""
        out_layers = [lay for lay in self.genes if lay.layer_in is layer_in]
        out_merge_layers = [lay for lay in self.genes if
                            lay.ltype == "merge" and
                            lay.layer2_in is layer_in]
        return out_layers, out_merge_layers

    def select_valid_layer_option(self, options, mutation_options):
        """Select a valid new layer option to add to the model."""
        rand.shuffle(options)

        def select_option(layer_options):
            """Select a new layer option to add to the model."""
            available_ltypes = list(set([opt[0] for opt in layer_options]))
            sum_probs = 0
            ltypes = []
            probs = []
            for opt in mutation_options.add_layer.options:
                if opt.name in available_ltypes:
                    sum_probs += opt.probability
                    ltypes.append(opt.name)
                    probs.append(opt.probability)
            probs = [p / sum_probs for p in probs]
            chosen_ltype = rand.choice(ltypes, p=probs)
            chosen_ltype_options = [opt for opt in options if opt[0] == chosen_ltype]
            rand.shuffle(chosen_ltype_options)
            return chosen_ltype_options[0]

        def has_valid_rank(lay):
            """Select a new layer option to add to the model."""
            if lay.ltype in ["conv", "2x2pool", "global_pool", "mbconv"]:
                return len(lay.dims_in) == 3 and lay.dims_in[0] > 0
            return True

        def check_model_validity(new_lay, in_lay, out_lay):
            """Temporarily inserts the proposed new layer to validate the resulting model."""
            self.genes.insert(self.genes.index(out_lay), new_lay)
            out_lay.layer_in = new_lay
            validity = self.has_valid_dims()
            out_lay.layer_in = in_lay
            self.genes.remove(new_lay)
            return validity

        while len(options) > 0:
            option = select_option(options)
            new_ltype, in_layer, out_layer = option
            new_layer = layer.create_layer(in_layer, new_ltype)

            if has_valid_rank(new_layer) and \
                    len(new_layer.dims_out()) == len(out_layer.dims_in) and \
                    check_model_validity(new_layer, in_layer, out_layer):
                return [new_layer, in_layer, out_layer, new_ltype]
            options.remove(option)

        print("No valid new layer options")  # TODO: nicer fix than print + returning Nones
        return [None, None, None, None]

    def has_valid_dims(self):
        """Performs several checks to see if the current architecture is valid."""
        self.update_layer_dims()

        # Prevent OOM errors by limiting the max number of params in the first layer:
        if (params.INPUT_IMG_SIZE ** 2) * self.genes[0].n_channels > 33000:
            print("Exit1 first layer max params exceeded")
            return False

        for lay in self.genes[1:]:
            # Prevent OOM errors by limiting the max number of params per layer
            if len(lay.dims_out()) == 3 and \
                    lay.dims_out()[0] * lay.dims_out()[1] * lay.dims_out()[2] > 32000:  # Was 33
                lay.print()
                print("Exit1 max layer params exceeded")
                return False
            if lay.ltype == "mbconv" and lay.max_params() > 32000:  # Was 33
                lay.print()
                print("Exit2 max MBconv layer params exceeded")
                return False
            # Check if input of merge layers remain consistent
            if lay.ltype == "merge":
                if len(lay.layer_in.dims_out()) == 3 and \
                        lay.layer_in.dims_out()[0] != lay.layer2_in.dims_out()[0]:
                    print("Exit3 unequal merge layer dimensions")
                    return False
            # Ensure spatial resolution dimensions remains > 0
            if len(lay.dims_out()) == 3 and lay.dims_out()[0] < 1:
                print("Exit4 image dimension < 1")
                return False

        # Check if model size isn't too large
        self.keras_model = self.create_keras_model()
        model_size = self.keras_model.count_params()
        if model_size > params.MAX_MODEL_PARAMS:
            print("Exit5 max model params exceeded")
            return False
        print("Exit6 validation succesful")
        return True

    def trim(self):
        """Removes any layer that has become disconnected from the main graph."""
        needs_trimming = True
        while needs_trimming:
            needs_trimming = False
            for lay in self.genes[:-2]:
                out_layers, out_merge_layers = self.get_output_layers(lay)
                all_out_layers = out_layers + out_merge_layers
                if len(all_out_layers) == 0:
                    self.genes.remove(lay)
                    needs_trimming = True
                    continue

                if lay.ltype == "merge":
                    if lay.layer_in is lay.layer2_in:
                        for out_lay in out_layers:
                            out_lay.layer_in = lay.layer_in
                        for out_merge_lay in out_merge_layers:
                            out_merge_lay.layer2_in = lay.layer_in
                        self.genes.remove(lay)
                        needs_trimming = True
