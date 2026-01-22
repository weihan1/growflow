# Use this script to officially render all imgs for metric computation
# The idea is we instantiate the Runner class but we run the full_evaluator class instead
from runner import Runner
import numpy as np
import torch
import os
from tqdm import tqdm 
import gsplat
import tyro
from configs.captured_config import Config
import yaml
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from glob import glob

def load_from_yaml(yaml_path):
    """Load a config from a YAML file and return a Config object.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.unsafe_load(file)
    
    # Create a new Config object with default values
    config = Config()
    
    # Update the config with values from the YAML file
    for key, value in yaml_data.items():
        if hasattr(config, key):
            # Handle special cases like strategy which might need instantiation
            if key == 'strategy' and isinstance(value, dict):
                strategy_type = value.pop('type', 'default')
                if strategy_type.lower() == 'mcmc':
                    setattr(config, key, MCMCStrategy(**value))
                else:
                    setattr(config, key, DefaultStrategy(**value))
            else:
                setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter '{key}' in YAML file")
    
    return config

def display_config(cfg):
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    table = Table(title="Configuration Settings")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    config_dict = vars(cfg)
    for key, value in sorted(config_dict.items()):
        # Convert complex objects to a reasonable string
        if isinstance(value, (list, dict)) and len(str(value)) > 100:
            value = f"{type(value).__name__} with {len(value)} items"
        table.add_row(key, str(value))
    
    console.print("\n")
    console.print(table)
    console.print("\n")


def is_default_value(cfg, key, value):
    # This is a simple implementation - you may need to adjust based on how tyro works
    # If tyro provides a way to check if a value was explicitly set, use that instead
    default_config = Config()  # Create a default config object
    if hasattr(default_config, key):
        return getattr(default_config, key) == value
    return False


def main(cfg):
    display_config(cfg)
    runner = Runner(cfg)
    runner.generate_gt()

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    #This will just enable default pruning strategy
    main(cfg)







