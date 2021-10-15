from cybertron.data_modules.PRISM.HMMDataModule import HMMDataModule
from multiprocessing import freeze_support
from cybertron.pretrained_models.ProtBert import ProtBert
from autobots.lightning_modules.Bert import BertTokenClassification
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertConfig
from torchmetrics import Accuracy
from torch.nn import ModuleDict
import os
from transformers import DataCollatorForTokenClassification
from autobots.optimizers.preconfigured import get_adamw
from autobots.optimizers.preconfigured import get_deepspeed_adamw
from pytorch_lightning.plugins import DeepSpeedPlugin
from autobots.optimizers.preconfigured import get_fused_adam
from autobots.optimizers.preconfigured import get_adafactor
import deepspeed

# Configure Environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
os.environ["DS_BUILD_CPU_ADAM"] = "1"
os.environ["DS_BUILD_UTILS"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

wandb_name = f"transformer_wlrd_test"
gpu_ids = [6, 7]
max_length = 512

def train():
    """
    Fine-tune model
    :return:
    """

    # logging
    wandb_logger = WandbLogger(name=wandb_name, project="hmm_reBERT")

    # prepare dataset for fine-tuning
    dm = HMMDataModule(
        batch_size=4,
        label_tokens=True,
        max_length=max_length,
        num_workers=32,
        persistent_workers=True,
    )
    dm.collator = DataCollatorForTokenClassification(
        tokenizer=dm.tokenizer,
        label_pad_token_id=(
            dm.label_tokenizer.pad_token_id if not None else -100
        ),
        padding="max_length",
        max_length=max_length,
    )

    dm.setup(stage="fit")

    # prep model config
    config = BertConfig(
        vocab_size=len(dm.tokenizer.get_vocab()),
        pad_token_id=dm.tokenizer.pad_token_id,
        eos_token_id=dm.tokenizer.eos_token_id,
        bos_token_id=dm.tokenizer.bos_token_id,
        sep_token_id=dm.tokenizer.sep_token_id,
        classifier_dropout=None,
        num_labels=dm.label_tokenizer.vocab_size,
        learning_rate=10e-3,
        attention_probs_dropout_prob=0.0,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=40000,
        num_attention_heads=16,
        num_hidden_layers=30,
        type_vocab_size=2,
    )

    print("model configured")

    # setup metrics
    accuracy = Accuracy(
        num_classes=dm.label_tokenizer.vocab_size,
        average="weighted",

    )  # subset_accuracy=False, mdmc_average="samplewise", threshold=0.5, average="weighted",

    train_metrics = ModuleDict({"accuracy": accuracy})

    # instantiate a pytorch lightning module
    model = BertTokenClassification(
        config=config,
        ignore_index=-100,
        optimizer_fn=get_adamw,
        train_metrics=train_metrics,
        val_metrics=train_metrics,
    )

    # setup deepspeed plugin
    stage3 = DeepSpeedPlugin(
        stage=3,
        offload_optimizer=True,
        offload_parameters=True,
    )

    # replace with pretrained protbert base
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
        auto_lr_find=True,
        logger=wandb_logger,
        precision=16,

    ) #        plugins=stage3,

    trainer.fit(model, dm)
    model.model.save_pretrained("/mnt/storage/grid/home/eric/hmm2bert/work/transformer_wrld/models")
    print("Finished Training")
    return 0


if __name__ == "__main__":
    freeze_support()
    train()