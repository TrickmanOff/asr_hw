import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

import hw_asr.metric as module_metric
import hw_asr.model as module_model
from hw_asr.metric.utils import calc_cer, calc_wer
from hw_asr.trainer import MetricTracker, Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    metrics_names = []
    for decoder_type in ['argmax', 'beam_search', 'lm_beam_search']:
        for met in ['CER', 'WER']:
            metrics_names.append(f'{met}_{decoder_type}')
    metrics_tracker = MetricTracker(*metrics_names)
    calculated_metrics = set()

    results = []

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            if type(output) is dict:
                batch.update(output)
            else:
                batch["logits"] = output
            batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["probs"] = batch["log_probs"].exp().cpu()
            batch["argmax"] = batch["probs"].argmax(-1)
            lm_beam_search_text_preds = None
            if hasattr(text_encoder, "batch_lm_ctc_decode"):
                    logits_list = [
                        logits.cpu().numpy()[:int(length)]
                        for logits, length in zip(batch["logits"], batch["log_probs_length"])
                    ]
                    lm_beam_search_text_preds: List[str] = text_encoder.batch_lm_ctc_decode(logits_list)
                    
            
            for i in range(len(batch["text"])):
                argmax = batch["argmax"][i]
                argmax = argmax[: int(batch["log_probs_length"][i])]
                results.append(
                    {
                        "ground_truth": batch["text"][i],
                        "pred_text_argmax": text_encoder.ctc_decode(argmax.cpu().numpy()),
                        # "pred_text_beam_search": text_encoder.ctc_beam_search(
                        #     batch["probs"][i], batch["log_probs_length"][i], beam_size=10
                        # )[:10],
                        # "pred_text_beam_search": text_encoder.ctc_beam_search(
                        #     batch["probs"][i], batch["log_probs_length"][i], beam_size=10
                        # )[:100],
                    }
                )
                if lm_beam_search_text_preds is not None:
                    results[-1]["pred_text_lm_beam_search"] = lm_beam_search_text_preds[i]

                ground_truth = batch["text"][i]
                for pred_name, pred in results[-1].items():
                    if not pred_name.startswith('pred_text'):
                        continue
                    if isinstance(pred, list):
                        pred = pred[0].text
                    decoding_type = pred_name.removeprefix('pred_text_')
                    calculated_metrics.add('CER_' + decoding_type)
                    metrics_tracker.update('CER_' + decoding_type, calc_cer(ground_truth, pred))
                    calculated_metrics.add('WER_' + decoding_type)
                    metrics_tracker.update('WER_' + decoding_type, calc_wer(ground_truth, pred))

    with Path(out_file).open("w") as f:
        json.dump(results, f, indent=2)

    for metric_name in metrics_tracker.keys():
        if metric_name in calculated_metrics:
            avg_val = metrics_tracker.avg(metric_name)
            print(f'avg {metric_name}: \t{avg_val:.5f}')


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            args_config = json.load(f)
            # override data
            if "data" in args_config:
                config.config["data"] = args_config["data"]
            config.config.update(args_config)

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio"),
                            "transcription_dir": str(
                                test_data_folder / "transcriptions"
                            ),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
