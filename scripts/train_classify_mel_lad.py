def main():
    from bioformers.datasets.ArrowDataset import ArrowDataset
    from bioformers.datasets import dataset_utils
    from bioformers.utilize.Bert import BertSeqClassification
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import ModelCheckpoint
    import os
    import warnings

    from transformers import BertModel, BertTokenizer
    import torch

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"


    num_cpu = 16
    max_length = 512
    lr = '7.272e-05'

    #load tokenizer and wandb logger

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    wandb_logger = WandbLogger(name=f"mel_lad-{lr}.2", project="hmm_reBERT")

    #load train and test tensors and instantiate pytorch lightning wrapper for the huggingface model with the base pretrained protbert model

    encoded_train = torch.load("/mnt/storage/grid/home/eric/hmm2bert/data_prep/test_train.pt")
    encoded_test = torch.load("/mnt/storage/grid/home/eric/hmm2bert/data_prep/test_val.pt")
    bsc = BertSeqClassification(pretrained_dir="Rostlab/prot_bert", use_adafactor=True, num_labels=2)

    #setup data collator, trainer, and dataloader for train and val dataset

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        resume_from_checkpoint="/mnt/storage/grid/home/eric/hmm2bert/models/mel_lad_epch_20.ckpt",
        max_epochs=21,
        gpus='0',
        auto_lr_find=False,
        logger=wandb_logger
    )
    warnings.filterwarnings("ignore")

    train_dl = DataLoader(encoded_train, batch_size=4, num_workers=num_cpu, collate_fn=data_collator, shuffle=True)
    eval_dl = DataLoader(encoded_test, batch_size=4, num_workers=num_cpu, collate_fn=data_collator, shuffle=False)

    #train and save classifier as checkpoint

    trainer.fit(bsc, train_dataloader=train_dl, val_dataloaders=eval_dl)
    #trainer.save_checkpoint("/mnt/storage/grid/home/eric/hmm2bert/models/mel_lad_epch_21.ckpt")
    #torch.save(bsc.state_dict(), "/mnt/storage/grid/home/eric/hmm2bert/models/saved_model.pt")
    torch.save(bsc, "/mnt/storage/grid/home/eric/hmm2bert/models/saved_model.pt")

if __name__ == '__main__':
    main()




