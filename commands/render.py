"""Render model or ground truth: python -m commands.render DATASET ..."""

from commands._common import config_type_for, dataset_args, display_config, render_config


def main(argv=None):
    dataset, args = dataset_args(argv)
    ground_truth = "--ground-truth" in args
    if ground_truth:
        args.remove("--ground-truth")
    cfg = render_config(config_type_for(dataset), args)
    display_config(cfg)
    if not ground_truth:
        assert cfg.dynamic_ckpt is not None, "please specify dynamic ckpt"
        print(f"Running full eval on {cfg.dynamic_ckpt}")
    from trainers.runner import Runner

    runner = Runner(cfg, load_test_cameras_only=(dataset == "captured" and not ground_truth))
    if ground_truth:
        runner.generate_gt()
    else:
        runner.full_eval()


if __name__ == "__main__":
    main()
