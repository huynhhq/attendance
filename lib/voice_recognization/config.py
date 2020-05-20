import os

# PATH
DATASET_NAME = "Data"
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(ROOT_PATH, DATASET_NAME)
MODELS_PATH = os.path.join(ROOT_PATH, "Models")
MODEL_FILE_PATH = os.path.join(ROOT_PATH, "Models/model_4.h5")
DATA_TEST_PATH = os.path.join(ROOT_PATH, "Test")
FEATURE_SET_FILE = os.path.join(MODELS_PATH, "all_targets_mfcc_sets_4.npz")

# CONTAINS
NUM_RECORD = 5
DURATION = 5  # seconds
RECORD_DURATION = 2  # seconds

GMM_EXT = ".gmm"

# THRESHOLD
THRESHOLD = 0.93 #0.998