import argparse
import json
import os
import sys

src_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(src_dir)

import hw_asr.storage as storage_module
from hw_asr.storage.experiments_storage import ExperimentsStorage
from hw_asr.storage.external_storage import ExternalStorage
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.utils.util import ROOT_PATH


# the best checkpoint
BEST_CHECKPOINT = {
    "run": "kaggle_deepspeech2_1+6_finetuning:finetuned1",
    "checkpoint": "model_best"
}


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=ROOT_PATH / "gdrive_storage" / "external_storage.json",
        type=str,
        help="external storage config file path (default: gdrive storage)",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List all checkpoints in the storage")

    download_parser = subparsers.add_parser("download", help="Download a checkpoint or config from the storage")
    download_parser.add_argument(
        "-p",
        "--path",
        default="saved/models",
        type=str,
        help="local directory for checkpoints",
    )
    download_parser.add_argument(
        "-r",
        "--run",
        type=str,
        default=None,
        help="run to download in format '{exp_name}:{run_name}'",
    )
    download_parser.add_argument(
        "checkpoints",
        nargs='*',
        help="checkpoints to download in format '{checkpoint_name}', 'latest', or 'best' (config will always be downloaded even if no checkpoints are specified)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config_filepath = args.config
    with open(config_filepath, 'r') as file:
        config = json.load(file)

    external_storage: ExternalStorage = ConfigParser.init_obj(config["external_storage"], storage_module)

    if args.command == 'list':
        print("The output format is as follows:")
        print(json.dumps({
            "experiment name":
            {
                "run name": {
                    "with_config": "whether there is a config for this run",
                    "checkpoints": [
                        {
                            "name": "...",
                            "creation_date": "..."
                        }
                    ]
                }
            }
        }, indent=4))
        print(external_storage.list_content())
    elif args.command == "download":
        exps_storage = ExperimentsStorage(args.path)
        checkpoints = set(args.checkpoints)
        if len(checkpoints) != len(args.checkpoints):
            print("Specifying duplicate checkpoints will have no effect.")

        if "best" in checkpoints:
            assert args.run is None, "Do not pass \"--run\" if you want to download the best checkpoint as the run for the best checkpoint is hardcoded."
            assert len(checkpoints) == 1, f"The \"best\" checkpoint should be the only one specified for download (while you passed {len(checkpoints)} checkpoints)."
            args.run = BEST_CHECKPOINT["run"]
            checkpoints = [BEST_CHECKPOINT["checkpoint"]]

        if args.run is None:
            raise RuntimeError("Pass the run name via the \"--run\" argument or run `python3 model_loader.py download best` to download the best checkpoint.")

        exp_name, run_name = args.run.split(':')
        run_storage = exps_storage.get_run(exp_name, run_name, create_run_if_no=True)

        if "latest" in checkpoints:
            run_checkpoints = external_storage.get_available_runs()[exp_name][run_name].checkpoints
            if any(checkpoint.creation_date is None for checkpoint in run_checkpoints):
                raise RuntimeError("Cannot determine the latest checkpoint because some checkpoints for the run do not have the creation date.")
            latest_checkpoint = run_checkpoints[0]
            for checkpoint in run_checkpoints:
                if checkpoint.creation_date > latest_checkpoint.creation_date:
                    latest_checkpoint = checkpoint
            latest_index = checkpoints.index("latest")
            checkpoints[latest_index] = latest_checkpoint.name

        if not os.path.exists(run_storage.get_config_filepath()):
            print(f'Loading config for run "{run_storage.get_run_id()}"...')
            external_storage.import_config(run_storage)
            print(f'Successfully loaded config for run "{run_storage.get_run_id()}"')
        else:
            print(f'Config for run "{run_storage.get_run_id()}" already loaded')

        local_checkpoints = run_storage.get_checkpoints_filepaths()
        for checkpoint_name in checkpoints:
            if checkpoint_name not in local_checkpoints:
                print(f'Loading checkpoint "{checkpoint_name}" for run "{run_storage.get_run_id()}"...')
                external_storage.import_checkpoint(run_storage, checkpoint_name)
                print(f'Successfully loaded checkpoint "{checkpoint_name}" for run "{run_storage.get_run_id()}" to "{run_storage.get_checkpoints_filepaths()[checkpoint_name]}"')
            else:
                print(f'Checkpoint "{checkpoint_name}" for run "{run_storage.get_run_id()}" already loaded to "{run_storage.get_checkpoints_filepaths()[checkpoint_name]}"')
    else:
        raise RuntimeError(f'Command "{args.command}" is not supported.')
