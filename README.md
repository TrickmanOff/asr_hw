![](https://img.shields.io/badge/python-3.10-blue.svg)

[The original README in Russian](./README_ru.md)

# Automatic Speech Recognition Project

This project was completed as part of the deep learning course in audio processing: [assignment](https://github.com/markovka17/dla/tree/2024/hw1_asr).

This repository includes a training pipeline for ASR models with logging, checkpoint saving (including Google Drive support), and metrics calculation. The following describes how to run a pre-trained model from this pipeline on arbitrary data and how to reproduce it.

The [DeepSpeech 2](https://arxiv.org/abs/1512.02595) architecture was used, but the pipeline can also be applied to other architectures.

You can find detailed experiment logs, metrics, and inference samples in the [WandB report](https://wandb.ai/trickman/asr_project/reports/ASR-HW--VmlldzoxMDQwODgwMQ).

## Preparation

The code was tested with Python 3.10.

First, install all required packages:
```bash
pip install -r requirements.txt
```

## Running the Pre-Trained Model

### Loading Weights

To load the weights, you can use the script [model_loader.py](./scripts/model_loader):
```bash
python3 scripts/model_loader/model_loader.py download best
```
This will download the best checkpoint selected from a series of experiments:

|                         | CER    | WER    |
|-------------------------|--------|--------|
| LibriSpeech: test-clean | 0.0703 | 0.1605 |
| LibriSpeech: test-other | 0.1904 | 0.3595 |

> Metrics were calculated using beam search with a language model for generating text from model outputs.

You can also specify a custom directory for saving the checkpoint using the `-p, --path` argument (default: `saved/models`).

If there are any issues with the Google Drive API, the model can be downloaded manually via [this link](https://drive.google.com/drive/folders/1k7JkQV9ZBwQTKEYfJqt78gI5ko6NtYN-?usp=drive_link) or [this one](https://disk.yandex.ru/d/oE_i4O-2daqYKQ).

### Model Inference

For inference, use the script [`test.py`](test.py), where:

- Argument `-r, --resume`: path to the model checkpoint.
> The full path to the checkpoint is displayed in the console when using `model_loader.py`.

- Argument `-c, --config`: path to an additional config file (inference configuration).\
  The main config will be taken from the same directory as the specified checkpoint.\
  The final config is the main one combined with the additional one (fields in the additional config take precedence).\
  The inference config specifies parameters such as the language model used for text generation from the model's logits and the batch size, as well as optionally the data for inference. Examples of such configs can be found in [`hw_asr/configs/eval_metrics_configs`](hw_asr/configs/eval_metrics_configs).

> In the configs [`test-clean.json`](hw_asr/configs/eval_metrics_configs/test-clean.json) and [`test-other.json`](hw_asr/configs/eval_metrics_configs/test-other.json), parameters for the language model used for decoding are provided, optimized for the LibriSpeech dev-clean dataset. These parameters are expected to perform well on other datasets as well.

Data for speech recognition inference can be specified:
- As a dataset in the inference config, e.g., [`test-clean.json`](hw_asr/configs/eval_metrics_configs/test-clean.json) (supported datasets can be found [here](hw_asr/datasets)).
- Using the `-t, --test-data-folder` argument, pointing to a directory with the following structure:
```plaintext
test_dir
|-- audio
|    |-- voice1.[wav|mp3|flac|m4a]
|    |-- voice2.[wav|mp3|flac|m4a]
|-- transcriptions
|    |-- voice1.txt
|    |-- voice2.txt
```

The `transcriptions` subdirectory and its files are optional and are only needed for calculating CER and WER.

Example run on the LibriSpeech dataset (test-clean):
```bash
python3 test.py \
   --config=hw_asr/configs/eval_metrics_configs/test-clean.json \
   --resume=saved/models/$EXP/$RUN/$CHECKPOINT.pth
```

Example run on a set of audio recordings:
```bash
python3 test.py \
   --config=hw_asr/configs/eval_metrics_configs/test-clean.json \
   --resume=saved/models/$EXP/$RUN/$CHECKPOINT.pth \
   --test-data-folder=test_data
```

> The language model and dataset (when inferring on one of the datasets) will be loaded automatically.

If ground truth transcriptions are provided, the script outputs the average CER and WER metrics for all audio recordings under different decoding methods (argmax, beam search, beam search + LM). Predictions, along with ground truth from the dataset, will be saved to the file specified by the `-o, --output` argument (default: `"output.json"`).

> For beam search-generated transcriptions, the script also saves the likelihoods of predictions according to the model's logits.

## Reproducing Training

### General Information

The entire training process is defined by a JSON configuration file ([examples](hw_asr/configs/)). Training can be started by running [train.py](./train.py) and specifying the config file path as the `-c, --config` argument.

To fine-tune a model, specify the checkpoint path using the `-r, --resume` argument. Initial model weights will be loaded from the checkpoint instead of being randomly initialized.

### Reproducing the Best Checkpoint

Training was performed using the command:
```bash
python3 train.py \
   --config=hw_asr/configs/1+6/kaggle_deepspeech2_1+6_bidir_gru.json
```

Followed by fine-tuning on LibriSpeech test-other:
```bash
python3 test.py \
   --config=hw_asr/configs/1+6/kaggle_deepspeech2_1+6_other_finetuning.json \
   --resume=saved/models/kaggle_deepspeech2_1+6/$RUN_NAME/model_best.pth
```
where `$RUN_NAME` is the run name from the first step.

> All required datasets will be downloaded automatically.

Some paths to datasets (`data_dir`, `index_dir`) in the configs correspond to the author's environment (training on Kaggle). These can be removed or replaced with your own paths.

The configs specify checkpoint saving to Google Drive ("external storage" section). To use it with your personal account, follow the instructions [here](docs/gdrive_storage.md#access-to-a-personal-google-drive) and update the "external storage" section with your folder ID and credentials file path. Alternatively, you can disable exporting to Google Drive by removing the "external storage" section.

[WandB](https://wandb.ai/) is used for logging. You must either log in using `wandb login` or set your token (authentication key) via the [`WANDB_API_KEY`](https://docs.wandb.ai/guides/track/environment-variables/) environment variable.

## Testing the Implementation

Tests for verifying some parts of the code are located in [`hw_asr/tests`](hw_asr/tests). All tests can be run with:
```bash
python3 -m unittest discover hw_asr/tests
```

## Author

Egor Egorov:
- Telegram: [@TrickmanOff](https://t.me/TrickmanOff)
- Email: yegyegorov@gmail.com