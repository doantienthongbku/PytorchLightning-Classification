import pytorch_lightning as lightning
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from data import LitDataModule
from model import LitModel

dm = LitDataModule(
    data_dir="cats_dogs_light",
    split_ratio=0.2,
    batch_size=2,
    num_worker=4,
    prefetch_factor=16
)
dm.setup(stage="fit")
checkpoint_callback = ModelCheckpoint(
    save_top_k=2,
    monitor="val/acc",
    mode="max",
    dirpath=f"checkpoints/convnext_v2_tiny/",
    filename="convnext_v2_tiny-drop0.5-{epoch:02d}-{val/acc:.2f}",
)
lr_monitor = LearningRateMonitor(logging_interval="step")
wandb_logger = WandbLogger(project="catdog_classification")
trainer = lightning.Trainer(
    callbacks=[checkpoint_callback, lr_monitor],
    devices="auto",
    accelerator="auto",
    max_epochs=10,
    precision="16-mixed",
    logger=wandb_logger,
)

model = LitModel(lr=0.001, num_classes=2)
trainer.fit(model, dm)
