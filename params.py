""""
Parameter class. Defines all hyperparameters.
"""
# Data params
INPUT_IMG_SIZE = 32
NUM_CLASSES = 100
TRAIN_SAMPLES = 50000
VAL_SAMPLES = 50000

# NSGA2 parameters
POPULATION_SIZE = 25
MAX_MODEL_PARAMS = 250000
NSGA_EPOCHS = 1000

# Model training parameters
LEARNING_RATE = 0.1
LR_MOMENTUM = 0.99
LR_DECAY = 0.03
NESTEROV = True
TRAIN_EPOCHS = 50
BATCH_SIZE = 512
WEIGHT_DECAY = 0.005

# Mutation parameters
INITIAL_MUTATIONS = 0
MAX_INIT_FITNESS1 = 0.97
LAYER_TYPE_OPTIONS = ["conv", "FC", "2x2pool", "mbconv", "global_pool"]
ADD_OPTIONS = ["mbconv", "2x2pool"]
MUTATION_MOMENTUM = 0.8
MUTATION_AGE_WEIGHT = 0.3
