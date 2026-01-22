from runner import Runner
import tyro
import torch
from glob import glob
import time
from configs.captured_config import Config
from gsplat.strategy import DefaultStrategy, MCMCStrategy
import os
import yaml

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

def main(cfg: Config):
    # if cfg.dynamic_ckpt is not None:
    #     dynamic_ckpt = cfg.dynamic_ckpt[0]
    #     print("found dynamic checkpoint, searching for existing config")
    #     two_levels_up = os.path.dirname(os.path.dirname(dynamic_ckpt)) 
    #     existing_configs = glob(os.path.join(two_levels_up, "*.yml"))
    #     try:
    #         # Store the original CLI args that should override the loaded config
    #         cli_overrides = {}
    #         for key, value in vars(cfg).items(): #this overwrites everything, so you need to specify the config values explicitly
    #             if key != "load_from_cfg": 
    #                 cli_overrides[key] = value
            
    #         # Load the config from file
    #         loaded_cfg = load_from_yaml(existing_configs[0])
            
    #         # Update the cfg with loaded values
    #         for key, value in vars(loaded_cfg).items():
    #             if key not in cli_overrides:  # Don't override CLI-provided values
    #                 setattr(cfg, key, value)
    #         print("Config loaded successfully with CLI overrides preserved")
    #     except Exception as e:
    #         print(f"Could not open config due to {e}, rolling back to default config settings...") 
    
    display_config(cfg)
    runner = Runner(cfg)
    runner.run()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    #Only supports Default densification
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01, 
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    print(f"training on {cfg.data_dir}")
    main(cfg)