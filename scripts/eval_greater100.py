from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import os
import warnings
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import BertModel, BertTokenizer
from abc import ABC
import pytorch_lightning as pl
import torch
from transformers import BertForTokenClassification, BertConfig, Adafactor, AdamW
from transformers import pipeline
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torch.nn.functional as F
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import label_ranking_average_precision_score, average_precision_score, f1_score
from datasets import load_dataset
from datasets import Dataset
from datasets import load_from_disk
import pandas as pd
from sklearn.model_selection import train_test_split
import json

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in
                self.encodings.items()}  # keys are input_ids, token_type_ids, attention_mask, labels, values are stored as a list of lists
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return (len(self.labels))


class newDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.encodings = {"input_ids": dataframe["input_ids"], "token_type_ids": dataframe["token_type_ids"],
                          "attention_mask": dataframe["attention_mask"]}
        self.labels = {"labels": dataframe["labels"]}

    def __getitem__(self, idx):
        item = {"input_ids": self.encodings["input_ids"][idx], "token_type_ids": self.encodings["token_type_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx], "labels": self.labels["labels"][idx]}
        return item

    def __len__(self):
        return (len(self.labels["labels"]))

class BertTokClassification(pl.LightningModule, ABC):
    def __init__(
            self,
            config: BertConfig = None,
            pretrained_dir: str = None,
            use_adafactor: bool = False,
            learning_rate=3e-5,
            **kwargs
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.use_adafactor = use_adafactor
        if pretrained_dir is None:
            self.bert = BertForTokenClassification(config, **kwargs)
        else:
            self.bert = BertForTokenClassification.from_pretrained(pretrained_dir, **kwargs)

    def forward(self, input_ids, attention_mask, labels):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device), labels=labels.to(self.device, dtype=torch.int64))
        loss = outputs.loss


        def get_acc(labels, logits):
            sumList = []
            for i in range(len(labels)):
                y_pred = torch.max(logits[i], 1).indices
                score = accuracy_score(labels[i], y_pred)
                sumList.append(score)
            avg = sum(sumList) / len(labels)
            return avg


        accuracy1 = get_acc(labels.cpu(), outputs.logits.cpu())

        # accuracy = balanced_accuracy_score(master[0], master[1])
        self.log(
            "train_batch_accuracy",
            accuracy1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device), labels=labels.to(self.device, dtype=torch.int64))
        loss = outputs.loss
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # def get_balanced_accuracy(labels, logits):
        #     y_pred = torch.max(logits, 1).indices
        #     score = balanced_accuracy_score(labels, y_pred)
        #     return score
        #
        # def label_average_precision(labels, logits):
        #     y_pred = torch.max(prob, 1).indices
        #     score = label_ranking_average_precision_score(labels, y_pred)
        #     return score
        # def f1_calc(labels, logits):
        #     sumList = []
        #     for i in range(len(labels)):
        #         y_pred = torch.max(logits[i], 1).indices
        #         score = f1_score(labels[i], y_pred, average='macro')
        #         sumList.append(score)
        #     avg = sum(sumList) / len(labels)
        #     return avg




        # """
        # 1. Iterate over the batch:
        #     for each_label in labels.cpu():
        #         shorten the length of the list to its true length using attention list and the function [true_length]
        #
        #     for each_logit in outputs.logits.cpu():
        #         use torch.max(outputs.logits.cpu()[0], 1) to get the indices for each logit (best label prediction)
        #         shorten the indices to its proper label length
        #         compare the indices to the labels
        # """
        def true_length(y_attention_mask):  # finds the start and stop of the actual sequence
            switch = False
            start = 0
            stop = 0
            counter = 0
            attention_mask = list(y_attention_mask)
            for i in attention_mask:
                if int(i) == 1 and switch == False:
                    switch = True
                    start = counter
                elif int(i) == 0 and switch == True:
                    stop = counter
                    break
                elif counter == 511:
                    stop = 512
                counter += 1
            return (start, stop)


        def short_clean(attention_mask, labels, logits): #attention_mask, labels.cpu(), outputs.logits.cpu()

            def true_length(y_attention_mask):  # finds the start and stop of the actual sequence
                switch = False
                start = 0
                stop = 0
                counter = 0
                attention_mask = list(y_attention_mask)
                for i in attention_mask:
                    if int(i) == 1 and switch == False:
                        switch = True
                        start = counter
                    elif int(i) == 0 and switch == True:
                        stop = counter
                        break
                    counter += 1
                return (start, stop)

            masterPred = []
            masterTrue = []
            for batch_index in range(len(labels)):
                real_len = true_length(attention_mask[batch_index])
                predIndecies = torch.max(outputs.logits.cpu()[batch_index], 1).indices
                start = real_len[0]
                stop = real_len[1]
                currentTrue = torch.LongTensor(labels[batch_index][start:stop])
                currentPred = torch.LongTensor(predIndecies[start:stop])
                if len(currentTrue) == 0:
                    masterTrue.append(currentTrue.tolist())
                    masterPred.append(currentPred.tolist())
                    print(f"CURRENT-PRED LEN: {len(currentPred)}")
                    print(f"CURRENT-TRUE LEN:{len(currentTrue)}")

            return (masterTrue, masterPred)

        master = short_clean(attention_mask, labels.cpu(), outputs.logits.cpu())
        print("###################################")
        print(f"MASTER-TRUE: {master[0]}")
        print("###################################")
        print(f"MASTER-PRED: {master[1]}")
        print("###################################")
        print("=======")
        print(f"LABEL LEN: {len(labels.cpu())}")
        for i in range(len(labels.cpu())):
            print(f"SINGLE LABEL LEN: {len(labels.cpu()[i])}")
            print(f"ATTENTION LEN: {len(attention_mask[i])}")
            #print(attention_mask[i])
            print(f"TRUE LEN: {true_length(attention_mask[i])}")
        print("=======")
        print(f"LABELS: {labels.cpu()}")
        print("||||||||||||||||||||||||||||")
        print("=======")
        print(f"LOGITS LEN: {len(outputs.logits.cpu())}")
        for i in outputs.logits.cpu():
            print(f"SINGLE LOGIT LEN: {len(i)}")
        print(f"LOGIT SINGLE LIST LEN: {len(outputs.logits.cpu()[0][0])}")
        b_logit = torch.max(outputs.logits.cpu()[0], 1)
        b_logit_indices = torch.max(outputs.logits.cpu()[0], 1).indices
        print(f"LOGIT BEST: {b_logit}")
        print(f"LOGIT BEST INDICES: {b_logit_indices}")
        print("=======")
        print(f"LOGITS: {outputs.logits.cpu()}")

        # accuracy = label_average_precision(labels.cpu(), logits=outputs.logits.cpu()) #replaced get_balanced_accuracy(labels.cpu(), logits=outputs.logits.cpu()) with label ranking average precision

        # def balanced_accuracy_score(labels, logits):
        #     sumList = []
        #     for i in range(len(labels)):
        #         y_predList = []
        #         trueList = []
        #         y_pred = logits[i]
        #         previous = 0
        #         for lab in labels[i]:
        #             if lab == -100:
        #                 y_predList.append(y_pred[previous])
        #                 trueList.append(previous)
        #             else:
        #                 previous = lab
        #                 y_predList.append(y_pred[previous])
        #                 trueList.append(previous)
        #
        #         num = average_precision_score(trueList, y_predList)
        #         sumList.append(num)
        #     big = sum(sumList) / len(labels)
        #     return big

        def get_bal_acc(labels, logits):
            sumList = []
            for i in range(len(labels)):
                y_pred = torch.max(logits[i], 1).indices
                score = balanced_accuracy_score(labels[i], y_pred)
                sumList.append(score)
            avg = sum(sumList) / len(labels)
            return avg

        accuracy = get_bal_acc(labels.cpu(), outputs.logits.cpu())

        self.log(
            "val_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": loss}

    def configure_optimizers(self):
        if self.use_adafactor:
            return Adafactor(
                self.parameters(),
                lr=self.learning_rate,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False)
        else:
            return AdamW(self.parameters(), lr=self.learning_rate)

    def save_pretrained(self, pretrained_dir):
        self.bert.save_pretrained(self, prtrained_dir)

    def predict_classes(self, input_ids, attention_mask, return_logits=False):
        output = self.bert(input_ids=input_ids.to(self.device), attention_mask=attention_mask)
        if return_logits:
            return output.logits
        else:
            probabilities = F.sigmoid(output.logits)
            predictions = torch.argmax(probabilities)
            return {"probabilities": probabilities, "predictions": predictions}

    def get_attention(self, input_ids, attention_mask, specific_attention_head: int = None):
        output = self.bert(inputs_ids=input_ids.to(self.device), attention_mask=attention_mask)
        if specific_attention_head is not None:
            last_layer = output.attentions[-1]  # grabs the last layer
            all_last_attention_heads = [torch.max(this_input[specific_attention_head], axis=0)[0].indices for this_input in last_layer]
            return all_last_attention_heads
        return output.attentions


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

gpu_idx = 0
num_labels = 285
model_path = "/mnt/storage/grid/home/eric/hmm2bert/models/pullin/pullin_best_loss-v5{1GPU}.pt"
data_folder = "pullin_parsed_data"
strat_val_name = "embedding_pullin_noDupes_val>100_stratified_domainPiece2.pt"
strat_val_path = f"/mnt/storage/grid/home/eric/hmm2bert/{data_folder}/{strat_val_name}"
encoded_label_filename = "encoded_parsed_pullin_noDupes>100_withAA_not_domain.csv"
encoded_csv = f"/mnt/storage/grid/home/eric/hmm2bert/{data_folder}/{encoded_label_filename}"

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

encoded_test = torch.load(strat_val_path)

model = torch.load(model_path)
model.eval()

meatpipe = pipeline(task='ner', model=model.bert, tokenizer=tokenizer, device=gpu_idx, ignore_labels=["-100"])

# load csv to huggingface dataset AND pandas dataframe
dataset = load_dataset('csv', data_files=encoded_csv)
df = pd.read_csv(encoded_csv)
dataset = dataset['train']
print(dataset.column_names)

#split the dataset into train and test, this produces a list with the row positions
num_rows_list = list(range(len(df)))
strat_train, strat_test = train_test_split(df, test_size=.2, stratify=df['labels'], random_state=420)

masterList = []
loopCounter = 0

goByNum = 2

for dataNum in range(0, len(encoded_test), goByNum):

    torch.cuda.device(gpu_idx)
    in_IDS = encoded_test[dataNum:dataNum + goByNum]["input_ids"]
    in_IDS = in_IDS.cuda(gpu_idx)
    att_mask = encoded_test[dataNum:dataNum + goByNum]["attention_mask"]
    att_mask = att_mask.cuda(gpu_idx)

    model = model.cuda(gpu_idx)

    preds = model.predict_classes(in_IDS, att_mask, return_logits=True)
    # preds = torch.max(preds, 1).indices
    # print(preds)
    # parses through each validation sample and gets logits and compiles into a masterList of metrics

    for singleSample in range(len(preds)):

        #   ======================
        def labelLabel(labels):
            labels = labels.tolist()
            trueList = []
            print(labels)
            previous = 0
            for lab in labels:
                if lab == -100:
                    trueList.append(previous)
                else:
                    previous = lab
                    trueList.append(previous)
            return trueList


        def domainAcc(true_label,
                      predict):  # finds the start and stop of the actual sequence (range style where stop is actually stop - 1)
            switch = False
            start = 0
            stop = len(true_label)
            counter = 1
            true_label = true_label
            domainLabel = None
            numCorrectLabels = 0
            numNotLabels = 0

            for i in true_label:
                if int(i) != 0 and switch == False:
                    switch = True
                    start = counter
                elif int(i) == 0 and switch == True:
                    stop = counter - 1
                    break
                counter += 1

            numInDomain = stop - (start - 1)

            for i in true_label:
                if i != 0:
                    domainLabel = i

            for i in predict[start - 1:stop]:
                if i == domainLabel:
                    numCorrectLabels += 1

            domainLabelAcc = numCorrectLabels / numInDomain

            for i in predict[:start - 1]:
                if i == 0:
                    numNotLabels += 1

            for i in predict[stop - 1:]:
                if i == 0:
                    numNotLabels += 1

            numNotInDomain = len(true_label) - numInDomain
            try:
                notDomainLabelAcc = numNotLabels / numNotInDomain
            except:
                notDomainLabelAcc = numNotLabels / len(true_label)

            return [start, stop, domainLabelAcc, notDomainLabelAcc]


        def evaluate_positions(true, predict):
            switch1 = True
            switch2 = True
            predStart = None
            trueStart = None
            predStop = None
            trueStop = None
            counter = 0

            for i in range(len(true)):

                if true[i] != 0 and switch2:
                    switch2 = False
                    trueStart = i
                #                 print(f"REAL START POSITION {trueStart}")

                if predict[i] != 0 and switch1:
                    switch1 = False
                    predStart = i
                #                 print(f"PRED START POSITION {predStart}")

                if true[i] == 0 and switch2 == False:
                    switch2 = None
                    trueStop = i - 1
                #                 print(f"REAL STOP POSITION {trueStop}")

                if predict[i] == 0 and switch1 == False:
                    switch1 = None
                    predStop = i - 1
                #                 print(f"PRED STOP POSITION {predStop}")

                #             if predict[i] != true[i]:
                #                 print(f"{counter}) {true[i]}|=|{predict[i]} >> NOT")
                #             else:
                #                 print(f"{counter}) {true[i]}|=|{predict[i]}")
                counter += 1

            if switch2 == False and trueStop == None:
                trueStop = len(true)

            if switch1 == False and predStop == None:
                predStop = len(pred)

            if trueStart == None and switch2 == True:
                print("HAAAAAAAAAAAAAAAAAAA NO DOMAIN")

            return (trueStart, trueStop, predStart, predStop)


        def matching_labels(true, predict, domainStart, domainStop):
            trueList = []
            predictListDomain = []
            predictListNotdomain = []
            predict = predict.tolist()
            for label in true:
                trueList.append(label)

            for label in predict[domainStart:domainStop]:
                predictListDomain.append(label)

            for label in predict[:domainStart + 1]:
                predictListNotdomain.append(label)
            for label in predict[domainStop:]:
                predictListNotdomain.append(label)

            trueList = set(trueList)
            trueList = list(trueList)
            predictListDomain = set(predictListDomain)
            predictListDomain = list(predictListDomain)
            predictListNotdomain = set(predictListNotdomain)
            predictListNotdomain = list(predictListNotdomain)

            return (trueList, predictListDomain, predictListNotdomain)


        #   ======================
        fullDataDict = {}
        labelDataDict = {}
        predict = torch.max(preds[singleSample], 1).indices
        true = labelLabel(encoded_test[dataNum + singleSample]["labels"])
        domainAccuracy = domainAcc(true, predict)
        acc_start = domainAccuracy[0]
        acc_stop = domainAccuracy[1]
        domain_accuracy = domainAccuracy[2]
        notDomain_accuracy = domainAccuracy[3]
        positions = evaluate_positions(true, predict)
        trueStart = positions[0]
        trueStop = positions[1]
        predStart = positions[2]
        predStop = positions[3]
        domainStart = acc_start - 1
        domainStop = acc_stop
        match_label = matching_labels(true, predict, domainStart, domainStop)
        labelsInTrue = match_label[0]
        labelsInPredDomain = match_label[1]
        labelsInPredNotdomain = match_label[2]
        start_site = 0
        stop_site = 0
        attentionLabel = None

        for i in labelsInTrue:
            if i != 0:
                attentionLabel = i

        print(true)
        print(predict)

        print(
            f"start: {acc_start}, stop: {acc_stop}, domain accuracy: {domain_accuracy}, not domain accuracy: {notDomain_accuracy}")

        print(f"trueList labels: {labelsInTrue}")
        print(f"predictListDomain: {labelsInPredDomain}")
        print(f"predictListNotdomain: {labelsInPredNotdomain}")

        if trueStart == predStart:
            print(f"START SITE ACCURATE")
            start_site = 1
        else:
            print(f"START SITE INACCURATE")
        print(f"TRUE START: {trueStart} || {trueStop} :TRUE STOP")
        if trueStop == predStop:
            print(f"STOP SITE ACCURATE")
            stop_site = 1
        else:
            print(f"STOP SITE INACCURATE")
        print(f"PRED START: {predStart} || {predStop} :PRED STOP")

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        """
        1) domain accuracy
        2) not domain accuracy
        3) start site accuracy
        4) stop site accuracy
        5) labels in true domain
        6) labels in pred domain
        7) labels in pred not domain
        """
        fullDataDict["domain_accuracy"] = domain_accuracy
        fullDataDict["notDomain_accuracy"] = notDomain_accuracy
        fullDataDict["start_site"] = start_site
        fullDataDict["stop_site"] = stop_site
        fullDataDict["trueLabels"] = labelsInTrue
        fullDataDict["predDomainLabels"] = labelsInPredDomain
        fullDataDict["predNotdomainLabels"] = labelsInPredNotdomain

        labelDataDict[attentionLabel] = fullDataDict
        masterList.append(labelDataDict)

    print(f"{loopCounter}/{len(encoded_test)}")
    loopCounter += goByNum
    torch.cuda.empty_cache()
    """
    1. check to see if start site matches, and also check if it labelled anything else before start site wrong too
    2. check to see if end site matches, also check if it labelled anything else after stop site wrong too
    3. check middle to see if it is all filled, also check if its all same label

    4. evaluate based on % of correct labels inside of domain
    5. also evaluate based on % of correct labels outside of domain

    6. then after getting percentages for each sample, evaluate based on the domains (group metrics by domains and average those)
    7. see what types of labels are predicted to be inside the domain, and also see what types of labels are predicted to be outisde the domain
    """

# parses through masterList to group the metrics of each sample by label for easier evaluation
masterEvalDict = {}
for labelDictLevel in masterList:
    lab = list(labelDictLevel.keys())
    lab = lab[0]
    labelDict = labelDictLevel[lab]
    if lab not in masterEvalDict:
        newDict = {
            "domain_accuracy": [labelDict["domain_accuracy"]],
            "notDomain_accuracy": [labelDict["notDomain_accuracy"]],
            "start_site": [labelDict["start_site"]],
            "stop_site": [labelDict["stop_site"]],
            "trueLabels": list(labelDict["trueLabels"]),
            "predDomainLabels": list(labelDict["predDomainLabels"]),
            "predNotdomainLabels": list(labelDict["predNotdomainLabels"])
        }

        masterEvalDict[lab] = newDict
    else:
        innerLabelDict = masterEvalDict[lab]
        innerLabelDict["domain_accuracy"].append(labelDict["domain_accuracy"])
        innerLabelDict["notDomain_accuracy"].append(labelDict["notDomain_accuracy"])
        innerLabelDict["start_site"].append(labelDict["start_site"])
        innerLabelDict["stop_site"].append(labelDict["stop_site"])

        innerLabelDict["trueLabels"] = list(innerLabelDict["trueLabels"])
        innerLabelDict["trueLabels"] += labelDict["trueLabels"]
        innerLabelDict["trueLabels"] = set(innerLabelDict["trueLabels"])

        innerLabelDict["predDomainLabels"] = list(innerLabelDict["predDomainLabels"])
        innerLabelDict["predDomainLabels"] += labelDict["predDomainLabels"]
        innerLabelDict["predDomainLabels"] = set(innerLabelDict["predDomainLabels"])

        innerLabelDict["predNotdomainLabels"] = list(innerLabelDict["predNotdomainLabels"])
        innerLabelDict["predNotdomainLabels"] += labelDict["predNotdomainLabels"]
        innerLabelDict["predNotdomainLabels"] = set(innerLabelDict["predNotdomainLabels"])

# compiles the metrics for each label into a label average

resultDict = {}
for eachLabel in masterEvalDict:
    innerLabDict = {}

    metricData = masterEvalDict[eachLabel]
    dom_acc = sum(metricData["domain_accuracy"]) / len(metricData["domain_accuracy"])
    notDom_acc = sum(metricData["notDomain_accuracy"]) / len(metricData["notDomain_accuracy"])
    start_site = sum(metricData["start_site"]) / len(metricData["start_site"])
    stop_site = sum(metricData["stop_site"]) / len(metricData["stop_site"])
    true_lab = list(metricData["trueLabels"])
    predDomLab = list(metricData["predDomainLabels"])
    predNotdomLab = list(metricData["predNotdomainLabels"])

    innerLabDict["domain_accuracy"] = dom_acc
    innerLabDict["notDomain_accuracy"] = notDom_acc
    innerLabDict["start_site"] = start_site
    innerLabDict["stop_site"] = stop_site
    innerLabDict["true_labels"] = true_lab
    innerLabDict["pred_domain_labels"] = predDomLab
    innerLabDict["pred_notDomain_labels"] = predNotdomLab

    resultDict[eachLabel] = innerLabDict

finalZ = {}
finalZ[0] = resultDict
with open("/mnt/storage/grid/home/eric/hmm2bert/pullin_parsed_data/pullin>100_results.json", "w") as file:
    json.dump(finalZ, file)

print(resultDict)