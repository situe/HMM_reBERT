from transformers import DataCollatorWithPadding
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import os
import warnings
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizer
from reBERTer import *


def main():
    #######################################
    num_cpu = 16
    lr = '6.4937e-05'
    wandb_name = f"pullin>1000_whiteSpace-{lr}-mxepch10"
    num_labels = 60
    max_epch = 10
    batch_size = 4
    gpus = '0, 1'


    #load data file paths
    data_folder = "pullin_parsed_data"
    strat_train_name = "embedding_pullin_noDupes_whiteSpace_train>1000_stratified_domainPiece.pt"
    strat_val_name = "embedding_pullin_noDupes_whiteSpace_val>1000_stratified_domainPiece.pt"

    #checkpoint save folder
    model_folder = "pullin"
    #######################################

    ###-dont need to touch-###
    strat_train_path = f"/mnt/storage/grid/home/eric/hmm2bert/{data_folder}/{strat_train_name}"
    strat_val_path = f"/mnt/storage/grid/home/eric/hmm2bert/{data_folder}/{strat_val_name}"
    ###

    #######################################


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    #load tokenizer and wandb logger

    wandb_logger = WandbLogger(name=wandb_name, project="hmm_reBERT")
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

    #load train and test tensors and instantiate pytorch lightning wrapper for the huggingface model with the base pretrained protbert model

    encoded_train = torch.load(strat_train_path)
    encoded_test = torch.load(strat_val_path)
    bsc = BertTokClassification(pretrained_dir='Rostlab/prot_bert', use_adafactor=True, num_labels=num_labels)

    #setup checkpoint callback

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_epoch',
        dirpath=f'/mnt/storage/grid/home/eric/hmm2bert/models/{model_folder}',
        filename='pullin>1000_whiteSpace_best_loss',
        save_top_k=3,
        mode='min'
    )

    #setup data collator, trainer, and dataloader for train and val dataset

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        max_epochs=max_epch,
        gpus=gpus,
        auto_lr_find=False,
        logger=wandb_logger,
        accelerator="ddp",
        callbacks=[checkpoint_callback]
    ) #
    warnings.filterwarnings("ignore")

    train_dl = DataLoader(encoded_train, batch_size=batch_size, num_workers=num_cpu, collate_fn=data_collator, shuffle=True)
    eval_dl = DataLoader(encoded_test, batch_size=batch_size, num_workers=num_cpu, collate_fn=data_collator, shuffle=False)

    #train and save classifier as checkpoint
    #trainer.tune(bsc, train_dataloader=[train_dl], val_dataloaders=[eval_dl])
    trainer.fit(bsc, train_dataloader=train_dl, val_dataloaders=eval_dl)
    #trainer.save_checkpoint(save_checkpoint_path)


if __name__ == '__main__':
    main()