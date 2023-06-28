import os
import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.model import FaceModel
from src.preprocess.functional import save_json
from src.utils.evaluate import evaluate
from src.utils.test import predict

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

warnings.filterwarnings("ignore")
pl.seed_everything(2023)


@hydra.main(version_base=None, config_path="config", config_name="train")
def train_test(cfg: DictConfig) -> None:
    model = FaceModel(cfg)
    if hasattr(cfg, "ckpt_path"):
        model = FaceModel.load_from_checkpoint(cfg.ckpt_path, cfg=cfg)
    dm = hydra.utils.instantiate(cfg.datamodule)
    callbacks = [hydra.utils.instantiate(x) for _, x in cfg.callbacks.items()]
    trainer = hydra.utils.instantiate(cfg.trainer)(callbacks=callbacks)
    trainer.fit(model, dm)
    best_ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
    if best_ckpt_path:
        model = FaceModel.load_from_checkpoint(best_ckpt_path, cfg=cfg)

    print("Model evaluation")
    dm.setup("test")
    predictions = predict(model, dm.test_data, cfg.accelerator)
    save_path = os.path.join(cfg.paths.output_dir, "predictions.json")
    save_json(predictions, save_path)

    print("Plots generation")
    dlib_predictions_path = os.path.join(
        cfg.datamodule.data_folder, "dlib_test_predictions.json"
    )
    prefix_path = Path(save_path).parent.as_posix()
    evaluate(dlib_predictions_path, save_path, prefix_path)


if __name__ == "__main__":
    train_test()
