# Script for loading checkpoints

Trained checkpoints may be stored in an external storage along with their configurations. This script allows you to download them locally, for example, for the purpose of inference.

> Currently, only Google Drive is supported as an external storage.

To download the best checkpoint (picked by the author based on metrics on the LibriSpeech test data), simply run:
```
python3 model_loader.py download best
```

You can specify a local path where the downloaded files will be stored using the `--path` option (the default is "saved/models").

---

There are:
- experiments
- possibly multiple runs with separate configurations for each experiment
- possibly multiple checkpoints for each run

To download only the configuration for a run, you can use the following command:
```
python3 model_loader.py download --run=$EXP_NAME:$RUN_NAME
```

Instead of a run name, you can use "latest" as an alias for the best one (chosen manually).

To download a checkpoint or several of them, you can list their names:
```
python3 model_loader.py download --run=$EXP_NAME:$RUN_NAME checkpoint_1_name checkpoint_2_name ...
```

Instead of a checkpoint name, you can use "latest" as an alias for the most recent one (with the latest creation date).

To list all checkpoints in the storage, you can run:
```
python3 model_loader.py list
```

---

All the commands above would work only for author's checkpoints situated in the folder specified in [this config](../../gdrive_storage/external_storage.json). To use the script for any other publicly shared folder, specify its ID in the config passed as the `-c/--config` argument (by default, `{REPO_ROOT}/gdrive_storage/external_storage.json`).

If you want to access hidden checkpoints in your personal Google Drive, follow the instructions [here](../../docs/gdrive_storage.md#access-to-a-personal-google-drive).