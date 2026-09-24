"""Train synthetic or captured scenes: python -m commands.train DATASET PRESET ..."""

import time

from commands._common import config_type_for, dataset_args, display_config, training_config


def main(argv=None):
    dataset, args = dataset_args(argv)
    cfg = training_config(config_type_for(dataset), args)
    print(f"training on {cfg.data_dir}")
    display_config(cfg)
    from trainers.runner import Runner

    runner = Runner(cfg)
    runner.run()
    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    main()
