# Standard imports
import argparse
import os
import random
import shutil
from datetime import datetime

# Third-party imports
import petname

# Custom imports
from src.train import train_function
from src.infer import infer_function
from src.eval import eval_function
from src.config import load_configuration


def parse_args():
    parser = argparse.ArgumentParser(description='Insect Recognition')

    parser.add_argument('--configuration', type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    configuration = load_configuration(args.configuration)

    if configuration.train:
        # Define the experiment name
        random_name = petname.Generate(words=2, separator="-")
        random_number = random.randint(1000, 9999)
        run_name = f"{configuration['train'].datasets[0].name}-{configuration['train'].task}-{datetime.now().strftime('%y%m%d-%H%M%S')}-{random_name}"
        run_path = f'runs/train/{run_name}'
        os.makedirs(run_path, exist_ok=True)
        shutil.copy(args.configuration, run_path)
        print(f'Experiment name: {run_name}')
        print('--------------------------------------------------------')

        # Train loop
        train_function(configuration['train'], run_name)

    elif configuration.eval:
        # Define the experiment name
        random_name = petname.Generate(words=2, separator="-")
        random_number = random.randint(1000, 9999)
        run_name = f"{configuration['eval'].datasets[0].name}-{configuration['eval'].task}-{datetime.now().strftime('%y%m%d-%H%M%S')}-{random_name}"
        run_path = f'runs/eval/{run_name}'
        os.makedirs(run_path, exist_ok=True)
        shutil.copy(args.configuration, run_path)
        print(f'Experiment name: {run_name}')
        print('--------------------------------------------------------')

        # Eval loop
        eval_function(configuration['eval'], run_name)

    elif configuration.infer:
        # Define the experiment name
        random_name = petname.Generate(words=2, separator="-")
        random_number = random.randint(1000, 9999)
        run_name = f"{configuration['infer'].target}-{datetime.now().strftime('%y%m%d-%H%M%S')}-{random_name}"
        run_path = f'runs/infer/{run_name}'
        os.makedirs(run_path, exist_ok=True)
        shutil.copy(args.configuration, run_path)
        print(f'Experiment name: {run_name}')
        print('--------------------------------------------------------')

        # Eval loop
        infer_function(configuration['infer'], run_name)

