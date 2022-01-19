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
import torch
from transformers import pipeline
from datasets import load_dataset

# Configure Environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
# os.environ["DS_BUILD_CPU_ADAM"] = "1"
# os.environ["DS_BUILD_UTILS"] = "1"
# os.environ["MAX_JOBS"] = "16"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



gpu_ids = [0, 1]
d_set = "/mnt/storage/grid/home/eric/hmm2bert/work/4p_NCBI_hmm_hits.csv"
max_length = 512
wandb_name = f"4p_trained_test"
ckpt_path = "/mnt/storage/grid/home/eric/hmm2bert/work/transformer_wrld_4pDS/checkpoints/train_4pDS_2epch_best_val_loss-v2.ckpt"
model_path = f"/mnt/storage/grid/home/eric/hmm2bert/work/transformer_wrld_4pDS/models/"

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
        csv_fh=d_set
    )
    dm.collator = DataCollatorForTokenClassification(
        tokenizer=dm.tokenizer,
        label_pad_token_id=(
            dm.label_tokenizer.pad_token_id if not None else -100
        ),
        padding="max_length",
        max_length=max_length,
    )

    dm.setup(stage="test")

    dl = dm.test_dataloader()

    ds = dl.dataset

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


    )

    train_metrics = ModuleDict({"accuracy": accuracy})

    # load Pretrained Token Classification Head w/incomplete base
    model = BertTokenClassification.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        config=config,
        ignore_index=-100,
        strict=False
    )

    # load pretrained ProtBert base
    module = ProtBert()
    module = module.from_pretrained(model_path)

    #replace incomplete base with pretrained
    model.model.model = module

    meatpipe = pipeline(
        task="ner",
        model=model.model.eval(),
        tokenizer=dm.tokenizer,
        config=config,
    )

    #========================== get referenceLabels List ===============
    def func_z(row, labelRef, labelList):

        if row not in labelRef.keys():
            labelRef[row] = len(labelRef.keys())
            labelList.append(row)

    dsb = load_dataset('csv', data_files="/mnt/storage/grid/var/datasets/corpus/prism/NCBI_hmm_hits.csv")

    labelRef = {}
    labelList = []
    dsb.map(lambda row: func_z(row, labelRef, labelList), input_columns=["full_name"])

    from sklearn import preprocessing

    le = preprocessing.LabelEncoder()
    le.fit(labelList)

    printList = []
    t = list(le.classes_)
    n = 0

    referenceLabels = {}
    for i in t:
        print(f"{n}) {i}")
        printList.append(i)
        referenceLabels[i] = n
        n += 1

    #====================================== prep sequences into easy iteration format for pipeline =============
    def getSequences(row, sequenceList):
        prelim_seq = list(row)
        seq = " ".join(prelim_seq)
        sequenceList.append(seq)

    # iterates through ds to get true labels of sequences
    def getLabels(row, labelList):
        labelList.append(row)

    sequenceList = []
    labelList = []

    ds.map(lambda row: getSequences(row, sequenceList), input_columns=["sequence"])
    ds.map(lambda row: getLabels(row, labelList), input_columns=["full_name"])

    #========================== iterates through list for pipeline ============================
    predDict = {}
    missingLabels = []
    listIndexOutOfRange = []
    inv_referenceLabels = {v: k for k, v in referenceLabels.items()}
    for i in range(len(sequenceList)):
        currentSeq = sequenceList[i]
        currentLabel = labelList[i]
        pred = meatpipe(currentSeq)

        tempPredList = []
        for row in pred:
            tempPredList.append(row["entity"])

        labelCounter = {}
        for label in tempPredList:
            if label not in labelCounter.keys():
                labelCounter[label] = 1
            else:
                labelCounter[label] += 1

        # decode predicted label from number to label by indexing
        decoded_labelList = []
        notfoundList = []
        for encoded_label in labelCounter:
            try:
                decoded_labelList.append(inv_referenceLabels[int(encoded_label[6:])])
            except:
                # if there are more labels predicted than there are in the reference label (from LabelEncoder)
                print(f"{encoded_label} NOT FOUND")
                decoded_labelList.append(encoded_label)
                notfoundList.append(encoded_label)
                if encoded_label not in missingLabels:
                    missingLabels.append(encoded_label)

        print(f"decoded_labelList: {decoded_labelList}")
        # make new labelCounter with decoded labels
        print(f"labelCounter: {labelCounter}")
        print(f"notfoundlist: {notfoundList}")

        decoded_labelCounter = {}
        for label_idx in range(len(labelCounter)):
            keys_list = list(labelCounter)
            encoded_labelList = list(labelCounter)
            if encoded_labelList[label_idx] not in notfoundList:
                try:
                    decoded_labelCounter[str(decoded_labelList[label_idx])] = labelCounter[keys_list[label_idx]]
                except:
                    listIndexOutOfRange.append(decoded)
            else:
                continue
        try:
            predLabelFinal = max(decoded_labelCounter, key=decoded_labelCounter.get)
        except:
            print("NO PRED???")
        print(f"decoded_labelCounter: {decoded_labelCounter}")
        print(f"Predicted Label: {predLabelFinal}")
        print(f"True label:      {labelList[i]}")
        print(i)
        print(" ")

        correct = 0
        if predLabelFinal == labelList[i]:
            correct = 1
        # append 0 or 1 (depending on correct or incorrect) in a dict (keys are true labels)
        # append the predicted labels in a dict (keys are true labels)
        if labelList[i] not in predDict.keys():
            predDict[labelList[i]] = [correct]
        else:
            predDict[labelList[i]].append(correct)

    json = json.dumps(predDict)
    f = open("/mnt/storage/grid/home/eric/hmm2bert/work/transformer_wrld_new/model_predTest1.json", "w")
    f.write(json)
    f.close()

    json = json.dumps(missingLabels)
    f = open("/mnt/storage/grid/home/eric/hmm2bert/work/transformer_wrld_new/missingLabels_test1.json", "w")
    f.write(json)
    f.close()


    print("Finished Training")
    return 0


if __name__ == "__main__":
    freeze_support()
    train()