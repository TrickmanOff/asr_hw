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
LATEST_RUN_NAME = "kaggle_deepspeech2_1+6_finetuning:finetuned1"


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
        default="latest",
        help="run to load in format '{exp_name}:{run_name}' or 'latest'",
    )
    download_parser.add_argument(
        "checkpoints",
        nargs='*',
        help="'{checkpoint_name}', or 'latest'; config will always be downloaded"
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
    else:
        exps_storage = ExperimentsStorage(args.path)

        if args.run == "latest":
            args.run = LATEST_RUN_NAME

        exp_name, run_name = args.run.split(':')
        run_storage = exps_storage.get_run(exp_name, run_name, create_run_if_no=True)

        if "latest" in args.checkpoints:
            run_checkpoints = external_storage.get_available_runs()[exp_name][run_name].checkpoints
            latest_checkpoint = run_checkpoints[0]
            for checkpoint in run_checkpoints:
                if checkpoint.creation_date > latest_checkpoint.creation_date:
                    latest_checkpoint = checkpoint
            latest_index = args.checkpoints.index("latest")
            args.checkpoints[latest_index] = latest_checkpoint.name

        if not os.path.exists(run_storage.get_config_filepath()):
            print(f'Loading config for run "{run_storage.get_run_id()}"...')
            external_storage.import_config(run_storage)
            print(f'Successfully loaded config for run "{run_storage.get_run_id()}"')
        else:
            print(f'Config for run "{run_storage.get_run_id()}" already loaded')

        local_checkpoints = run_storage.get_checkpoints_filepaths()
        for checkpoint_name in args.checkpoints:
            if checkpoint_name not in local_checkpoints:
                print(f'Loading checkpoint "{checkpoint_name}" for run "{run_storage.get_run_id()}"...')
                external_storage.import_checkpoint(run_storage, checkpoint_name)
                print(f'Successfully loaded checkpoint "{checkpoint_name}" for run "{run_storage.get_run_id()}" to "{run_storage.get_checkpoints_filepaths()[checkpoint_name]}"')
            else:
                print(f'Checkpoint "{checkpoint_name}" for run "{run_storage.get_run_id()}" already loaded to "{run_storage.get_checkpoints_filepaths()[checkpoint_name]}"')
