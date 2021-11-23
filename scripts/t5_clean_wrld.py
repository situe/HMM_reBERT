from cybertron.data_modules.PRISM.HMMDataModule import HMMDataModule
from multiprocessing import freeze_support
from cybertron.pretrained_models.ProtT5 import ProtT5
# from cybertron.pretrained_models.ProtBert import ProtBert
from autobots.lightning_modules.Bert import BertTokenClassification
from autobots.lightning_modules.T5 import T5MultiTaskSingleOptSeqClassification
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import RobertaConfig
from transformers import T5Config
from transformers import BertConfig
from decepticons.heads.RobertaSequenceClassificationHead import RobertaClassificationHead
from decepticons.heads.TokenClassificationCrfHead import TokenClassificationCrfHead
from decepticons.models.T5.T5ForTokenCrfClassification import T5ForTokenCrfClassification
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
# os.environ["DS_BUILD_CPU_ADAM"] = "1"
os.environ["DS_BUILD_UTILS"] = "1"
os.environ["MAX_JOBS"] = "16"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

wandb_name = f"t5-roberta_transformer_wrld_test"
gpu_ids = [0, 1]
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
        batch_size=2,
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

    #prep classification heads
    classification_heads = ModuleDict()


    t5config = T5Config(
        vocab_size=len(dm.tokenizer.get_vocab()),
        d_model=1024,
        d_kv=32, #must be d_model // num_heads (og was 126) <-- not sure if theres something special or what
        d_ff=16384,
        num_layers=24,
        num_heads=32,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-06,
        initializer_factor=1.0,
        eos_token_id=dm.tokenizer.eos_token_id,
        pad_token_id=dm.tokenizer.pad_token_id,
    ) #hidden_dropout_prob=0.8, #why is it asking for this

    #crf head for token classification
    classification_heads["labels"] = TokenClassificationCrfHead(config=t5config)

    print("model configured")

    # setup metrics
    accuracy = Accuracy(
        num_classes=dm.label_tokenizer.vocab_size,
    )

    label_metrics = ModuleDict({"label_accuracy": accuracy})
    train_metrics = ModuleDict({"labels": label_metrics})

    # load pretrained T5
    pretrained = ProtT5()
    pretrained = pretrained.encoder

    # instantiate a pytorch lightning module
    # model = T5MultiTaskSingleOptSeqClassification(
    #     model=pretrained,
    #     classifier_heads=classification_heads,
    #     optimizer_fn=get_adamw,
    #     train_metrics=train_metrics,
    #     val_metrics=train_metrics,
    #     ignore_index=-100
    # )

    #instatiate T5ForTokenCrfClassification model

    model = T5ForTokenCrfClassification(
        config=t5config,
        model=pretrained,
        classifier_heads=classification_heads["labels"],
        optimizer_fn=get_adamw,
        train_metrics=train_metrics,
        val_metrics=train_metrics,
        ignore_index=-100
    )


    # setup deepspeed plugin
    # stage3 = DeepSpeedPlugin(
    #     stage=3,
    #     offload_optimizer=True,
    #     offload_parameters=False,
    # )model = T5ForTokenCrfClassification(
    #         config=pretrained.config,
    #         model=pretrained,
    #         classifier_heads=classification_heads,
    #         optimizer_fn=get_adamw,
    #         train_metrics=train_metrics,
    #         val_metrics=train_metrics,
    #         ignore_index=-100  # why is it -3 and not -100?
    #     )

    # setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="/mnt/storage/grid/home/eric/hmm2bert/work/transformer_wrld/checkpoints_t5",
        filename=f"{wandb_name}_best_val_loss",
        save_top_k=3,
        mode="min",
    )

    # setup trainer
    trainer = Trainer(
        max_epochs=1,
        callbacks=[checkpoint_callback],
        gpus=gpu_ids,
        strategy="ddp",
        auto_lr_find=True,
        logger=wandb_logger,
        precision=16,
    ) #        plugins=stage3,

    trainer.fit(model, dm)
    # model.model.save_pretrained("/mnt/storage/grid/home/eric/hmm2bert/work/transformer_wrld_models/models")
    print("Finished Training")
    return 0


if __name__ == "__main__":
    freeze_support()
    train()