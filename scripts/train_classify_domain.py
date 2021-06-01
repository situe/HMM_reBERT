def main():
    from bioformers.utilize.Bert import BertSeqClassification
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    import os
    import warnings
    from pytorch_lightning.callbacks import ModelCheckpoint

    from transformers import BertModel, BertTokenizer
    import torch

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    #######################################
    num_cpu = 16
    lr = '6.918e-12'
    wandb_name = f"puffinCaller_classify_domain_1893_with8Functional-{lr}-mxepch24"
    num_labels = 1893
    max_epch = 24
    gpus = '0, 1, 2, 3'

    data_folder = "tensor_datasets"
    strat_train_name = "puffinCaller_with8Functional_balanced_strat_train.pt"
    strat_val_name = "puffinCaller_with8Functional_balanced_strat_val.pt"

    model_folder = "puffinCaller/ckpt"
    save_checkpoint_name = "puffinCaller_with8Functional_balanced_labels_maxepch24.ckpt"

    ###-dont need to touch-###
    save_checkpoint_path = f"/mnt/storage/grid/home/eric/hmm2bert/models/{model_folder}/{save_checkpoint_name}"
    strat_train_path = f"/mnt/storage/grid/home/eric/hmm2bert/data_prep/{data_folder}/{strat_train_name}"
    strat_val_path = f"/mnt/storage/grid/home/eric/hmm2bert/data_prep/{data_folder}/{strat_val_name}"
    ###

    #######################################

    #load tokenizer and wandb logger

    wandb_logger = WandbLogger(name=wandb_name, project="hmm_reBERT")
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

    #load train and test tensors and instantiate pytorch lightning wrapper for the huggingface model with the base pretrained protbert model

    encoded_train = torch.load(strat_train_path)
    encoded_test = torch.load(strat_val_path)
    bsc = BertSeqClassification(pretrained_dir="Rostlab/prot_bert", use_adafactor=True, num_labels=num_labels)

    #setup checkpoint callback

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_epoch',
        dirpath=f'/mnt/storage/grid/home/eric/hmm2bert/models/{model_folder}',
        filename='puffinCaller_with8Functional_balanced_best_loss',
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
    )
    warnings.filterwarnings("ignore")

    train_dl = DataLoader(encoded_train, batch_size=4, num_workers=num_cpu, collate_fn=data_collator, shuffle=True)
    eval_dl = DataLoader(encoded_test, batch_size=4, num_workers=num_cpu, collate_fn=data_collator, shuffle=False)

    #train and save classifier as checkpoint

    trainer.fit(bsc, train_dataloader=train_dl, val_dataloaders=eval_dl)
    #trainer.save_checkpoint(save_checkpoint_path)


if __name__ == '__main__':
    main()