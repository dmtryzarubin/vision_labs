import typing as ty

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig


class FaceModel(pl.LightningModule):
    logger_mapping = {"train": 0, "valid": 1}

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = hydra.utils.instantiate(cfg.model)
        self.criterion = hydra.utils.instantiate(cfg.criterion)
        self.outputs = {
            "train": [],
            "valid": [],
        }
        self.metrics = {
            "train": hydra.utils.instantiate(cfg.metrics),
            "valid": hydra.utils.instantiate(cfg.metrics),
        }

    def configure_optimizers(
        self,
    ) -> ty.Dict[str, ty.Any]:
        optimizer = hydra.utils.instantiate(self.cfg.optim)(self.model.parameters())
        if hasattr(self.cfg, "lr_scheduler"):
            lr_scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler)(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": "loss/valid",
            }
        return {"optimizer": optimizer}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        output = self.forward(x)
        B, P = output.shape
        output = output.view(B, P // 2, 2)
        if self.cfg.normalize_keypoints:
            output[..., 0] *= self.cfg.width
            output[..., 1] *= self.cfg.height
        return output

    def common_step(self, batch: ty.Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        x, y = batch["image"], batch["keypoints"]
        preds = self.model(x)
        metrics = self.metrics[mode](preds.cpu(), y.cpu())
        loss = self.criterion(preds, y)
        metrics["loss"] = loss
        self.outputs[mode].append(metrics)
        return metrics

    def _avg_losses(self, losses):
        keys = losses[0].keys()
        metricDict = {}
        for key in keys:
            for key in keys:
                metricDict[key] = torch.stack([x[key] for x in losses]).mean()
        return metricDict

    def epoch_end(self, mode: str):
        losses = self._avg_losses(self.outputs[mode])
        self.outputs[mode].clear()
        self.log_dict(
            {f"{k}/{mode}": v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        if mode == "train":
            losses["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.loggers[self.logger_mapping[mode]].log_metrics(
            losses, step=self.current_epoch
        )

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, mode="train")

    def on_train_epoch_end(self):
        return self.epoch_end(mode="train")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, mode="valid")

    def on_validation_epoch_end(self):
        return self.epoch_end(mode="valid")
