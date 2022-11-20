from pathlib import Path
import signal
from nni.experiment import Experiment
search_space = {
    # 'num_leaves': {'_type': 'choice', '_value': [31, 28, 24, 20]},
    # 'bagging_freq': {'_type': 'choice', '_value': [1, 2, 4, 8, 10]},
    # 'Regulizers': {'_type': 'uniform', '_value': [11, 12]},
    'weight1': {'_type': 'uniform', '_value': [0, 0.2]},
    'weight2': {'_type': 'uniform', '_value': [0, 0.2]},
    'weight3': {'_type': 'uniform', '_value': [0, 0.2]},
    'weight4': {'_type': 'uniform', '_value': [0, 0.2]},
    'weight5': {'_type': 'uniform', '_value': [0, 0.2]},
    'weight6': {'_type': 'uniform', '_value': [0, 0.2]},
    'dropout_rate': {'_type': 'uniform', '_value': [0, 1]},
    'learning_rate': {'_type': 'uniform', '_value': [0, 0.1]},
}
experiment = Experiment('local')
experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Metis'
experiment.config.tuner.class_args = {
    'optimize_mode': 'maximize'
}
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2
experiment.run(port=8000, wait_completion=False)

print('Experiment is running. Press Ctrl-C to quit.')
signal.pause()