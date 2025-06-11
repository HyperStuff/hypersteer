import tempfile
from pathlib import Path

from hydra import compose, initialize

# Import the train main function (Hydra-decorated)
from hypersteer.scripts import train as train_script


def test_save_and_resume_e2e():
    # 1. Compose config with Hydra to resolve all defaults and inheritance
    with initialize(config_path="../config"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=hypersteer",
                "wandb.log=False",
                "wandb.log_code=False",
                "train.n_steps=2",
                "train.n_epochs=1",
                "train.save_interval=1",
                "train.val_interval=2",
                "train.batch_size=1",
                "train.test_batch_size=1",
                "dataset.train.max_concepts=2",
                "dataset.train.num_of_examples=4",
                "debug=True",
            ],
        )

    # Use a temp directory for dump_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["dump_dir"] = tmpdir
        cfg["train"]["save_dir"] = tmpdir

        # 2. Run training for a few steps (should save checkpoint)
        train_script.main(cfg)

        # 3. Assert checkpoint exists
        train_dir = Path(tmpdir) / "train"
        # Find the latest step dir
        step_dirs = [d for d in train_dir.glob("step_*") if d.is_dir()]
        assert step_dirs, "No checkpoint step directories found after first run."
        # Get the highest step number
        last_step = max(int(d.name.split("_")[-1]) for d in step_dirs)
        last_ckpt_dir = train_dir / f"step_{last_step}"
        # Check for a model weight file
        weight_files = list(last_ckpt_dir.glob("*_weight.safetensors"))
        assert weight_files, f"No model weight file found in {last_ckpt_dir}"

        # 4. Resume from checkpoint: set train.resume_from to the checkpoint dir
        with initialize(config_path="../config"):
            cfg_resume = compose(
                config_name="config",
                overrides=[
                    "experiment=hypersteer",
                    "wandb.log=False",
                    "wandb.log_code=False",
                    f"train.n_steps={last_step + 2}",
                    "train.n_epochs=1",
                    "train.save_interval=1",
                    "train.val_interval=2",
                    "train.batch_size=1",
                    "train.test_batch_size=1",
                    "dataset.train.max_concepts=2",
                    "dataset.train.num_of_examples=4",
                    f"dump_dir={tmpdir}",
                    f"train.save_dir={tmpdir}",
                    f"train.resume_from={str(last_ckpt_dir)}",
                    "debug=True",
                ],
            )

        # 5. Run training again (should resume)
        train_script.main(cfg_resume)

        # 6. Assert new checkpoint exists and step increased
        step_dirs_after = [d for d in train_dir.glob("step_*") if d.is_dir()]
        assert len(step_dirs_after) > len(step_dirs), "No new checkpoint after resume."
        new_last_step = max(int(d.name.split("_")[-1]) for d in step_dirs_after)
        assert new_last_step > last_step, "Global step did not increase after resume."

        # 7. (Optional) Check that new weights file exists
        new_ckpt_dir = train_dir / f"step_{new_last_step}"
        new_weight_files = list(new_ckpt_dir.glob("*_weight.safetensors"))
        assert new_weight_files, f"No model weight file found in {new_ckpt_dir}"
