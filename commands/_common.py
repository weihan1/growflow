"""Shared CLI behavior for the synthetic and captured entry points."""

import sys

import tyro
from gsplat.strategy import DefaultStrategy, MCMCStrategy


def config_type_for(dataset):
    if dataset == "synthetic":
        from configs.blender_config import Config
    elif dataset == "captured":
        from configs.captured_config import Config
    else:
        raise SystemExit("Choose a dataset: synthetic or captured")
    return Config


def dataset_args(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("synthetic", "captured"):
        raise SystemExit(
            "Choose a dataset: synthetic or captured. "
            "Example: python -m commands.train synthetic default --help"
        )
    return args[0], args[1:]


def training_config(config_type, args):
    """Keep the original default/mcmc Tyro presets and step scaling."""
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            config_type(strategy=DefaultStrategy(verbose=True)),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            config_type(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs, args=args)
    cfg.adjust_steps(cfg.steps_scaler)
    return cfg


def render_config(config_type, args):
    cfg = tyro.cli(config_type, args=args)
    cfg.adjust_steps(cfg.steps_scaler)
    return cfg


def display_config(cfg):
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Configuration Settings")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    for key, value in sorted(vars(cfg).items()):
        if isinstance(value, (list, dict)) and len(str(value)) > 100:
            value = f"{type(value).__name__} with {len(value)} items"
        table.add_row(key, str(value))
    console = Console()
    console.print("\n")
    console.print(table)
    console.print("\n")
