from cybertron.data_modules.PRISM.HMMDataModule import HMMDataModule
from multiprocessing import freeze_support
from cybertron.pretrained_models.ProtBert import ProtBert
from autobots.lightning_modules.Bert import BertTokenClassification
from autobots.optimizers.preconfigured import get_adafactor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertConfig
from torchmetrics import Accuracy
from torch.nn import ModuleDict
import os
from transformers import DataCollatorForTokenClassification

# Configure Environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

wandb_name = f"transformer_wlrd_test"
gpu_ids = [3, 4]

def train():
    """
    Fine-tune model
    :return:
    """

    # logging
    wandb_logger = WandbLogger(name=wandb_name, project="hmm_reBERT")

    # prepare dataset for fine-tuning
    dm = HMMDataModule(
        batch_size=8,
        label_tokens=False,
        max_length=2048,
        num_workers=16,
        persistent_workers=True,
    )
    dm.collator = DataCollatorForTokenClassification(
        tokenizer=dm.tokenizer,
        label_pad_token_id=(
            dm.label_tokenizer.pad_token_id if not None else -100
        ),
        padding="longest",
    )

    dm.setup(stage="fit")

    # prep model config
    config = BertConfig(
        vocab_size=len(dm.tokenizer.get_vocab()),
        max_position_embeddings=2048,
        pad_token_id=dm.tokenizer.pad_token_id,
        eos_token_id=dm.tokenizer.eos_token_id,
        bos_token_id=dm.tokenizer.bos_token_id,
        sep_token_id=dm.tokenizer.sep_token_id,
        classifier_dropout=None,
        num_labels=dm.tokenizer.vocab_size,
        learning_rate=10e-3,
    )
    print("model configured")

    # setup metrics
    num_classes = dm.label_tokenizer.vocab_size
    print(num_classes)
    accuracy = Accuracy(
        threshold=0.8,
        num_classes=num_classes,
        average="weighted",
        mdmc_average="samplewise",
        subset_accuracy=False,
    )
    train_metrics = ModuleDict({"accuracy": accuracy})

    # instantiate a pytorch lightning module
    model = BertTokenClassification(
        config=config,
        ignore_index=-100,
        optimizer_fn=get_adafactor,
        train_metrics=train_metrics,
    )

    # replace with pretrained protbert base and clean up
    model.model.model = ProtBert()


    # setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="/mnt/storage/grid/home/eric/hmm2bert/work/transformer_wrld/checkpoints",
        filename=f"{wandb_name}_best_val_loss",
        save_top_k=3,
        mode="min",
    )

    # setup trainer
    trainer = Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        gpus=gpu_ids,
        accelerator="ddp",
        precision=16,
        auto_lr_find=True,
        logger=wandb_logger,
    )

    trainer.fit(model, dm)
    model.model.save_pretrained("/mnt/storage/grid/home/eric/hmm2bert/work/transformer_wrld/models")
    print("Finished Training")
    return 0

if __name__ == "__main__":
    freeze_support()
    train()