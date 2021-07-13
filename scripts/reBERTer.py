from torch.utils.data import DataLoader
from abc import ABC
import pytorch_lightning as pl
import torch
from transformers import BertForTokenClassification, BertConfig, Adafactor, AdamW
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torch.nn.functional as F
import pandas as pd #create dataframes
from sklearn import preprocessing #encode labels


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
        self.bert.save_pretrained(self, pretrained_dir)

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


###############EVALUATE FUNCTIONS BELOW########################

def labelLabel(label, sequence, start, stop):
    sequence = sequence.split()
    sequence = "".join(sequence)
    domainList = list(range(start, stop + 1))
    sequenceIndexCounter = 0
    trueList = []

    for currentNum in range(len(sequence)):  # iterate over each amino acid in a single sequence
        # ====
        shiftedNum = currentNum  # CHANGED FROM currentNum+1 TO currentNum # index that we compare to as sequence index starts at zero but start/stop starts at 1
        # ====
        if shiftedNum == domainList[sequenceIndexCounter]:
            trueList.append(label)
            if domainList[sequenceIndexCounter] == domainList[-1]:
                sequenceIndexCounter = 0

            else:
                sequenceIndexCounter += 1
        else:
            trueList.append(0)
    return trueList


def domainAcc(true_label, predict):  #calculates the fulldomain, domain, and notdomain accuracy
    switch = False
    start = None
    stop = len(true_label)
    counter = 0  # CHANGED FROM 1 TO 0
    domainLabel = None
    numCorrectLabels = 0
    numNotLabels = 0
    fullDomainCounter = 0

    # gets accuracy of prediction of whole sequence
    for i in range(len(true_label)):
        if true_label[i] == predict[i]:
            fullDomainCounter += 1
    fullDomainScore = fullDomainCounter / len(predict)

    # sets the start and stop of domain based on true label list
    for i in true_label:
        if int(i) != 0 and switch == False:
            switch = True
            start = counter
        elif int(i) == 0 and switch == True:
            stop = counter - 1
            break
        counter += 1

    # if domain is out of range and whole sequence is notDomain set start equal to stop + 1
    if start == None:
        start = stop + 1

    # calculates the number of amino acids in domain
    numInDomain = stop - (start)

    for i in true_label:
        if i != 0:
            domainLabel = i

    for i in predict[start - 1:stop]:
        if i == domainLabel:
            numCorrectLabels += 1
    try:
        domainLabelAcc = numCorrectLabels / numInDomain
    except:
        domainLabelAcc = 0

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
        notDomainLabelAcc = 0  # numNotLabels / len(true_label) #maybe change to 0?

    return [domainLabelAcc, notDomainLabelAcc, fullDomainScore]


def matching_labels(true, predict, domainStart, domainStop):
    trueList = []
    predictListDomain = []
    predictListNotDomain = []
    for label in true:
        trueList.append(label)

    for label in predict[domainStart:domainStop + 1]: #+1 to domainStop because range is not inclusive
        predictListDomain.append(label)

    for label in predict[:domainStart]:
        predictListNotDomain.append(label)
    for label in predict[domainStop + 1:]:
        predictListNotDomain.append(label)

    trueList = set(trueList)
    trueList = list(trueList)
    predictListDomain = set(predictListDomain)
    predictListDomain = list(predictListDomain)
    predictListNotDomain = set(predictListNotDomain)
    predictListNotDomain = list(predictListNotDomain)

    for i in range(len(trueList)):
        trueList[i] = int(trueList[i])
    for i in range(len(predictListDomain)):
        predictListDomain[i] = int(predictListDomain[i])
    for i in range(len(predictListNotDomain)):
        predictListNotDomain[i] = int(predictListNotDomain[i])

    return (trueList, predictListDomain, predictListNotDomain)


def evaluate_positions(true, predict, buffer=6):

    def check_ahead(position, predict, buffer):
        ret = True
        for i in range(1, buffer + 1):
            if predict[position + i] != 0:
                ret = False
                break
        return ret

    switch1 = True
    switch2 = True
    predStart = None
    trueStart = None
    predStop = None
    trueStop = None


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
            # switch1 = None
            # predStop = i - 1
# =======================================================
            if i < ((len(true) - 1) - buffer):
                checkAhead = check_ahead(i, predict, buffer)
                if checkAhead:
                    switch1 = None
                    predStop = i - 1
                else:
                    switch1 = False

            else:  # how to make it a decaying buffer near end?
                switch1 = None
                predStop = i - 1


    if switch2 == False and trueStop == None:
        trueStop = len(true) - 1

    if switch1 == False and predStop == None:
        predStop = len(predict) - 1

    if trueStart == None and switch2 == True:
        print("HAAAAAAAAAAAAAAAAAAA NO DOMAIN")
        print(trueStart, trueStop, predStart, predStop)
        return False
    else:
        return [trueStart, trueStop, predStart, predStop]

#default buffer length of 6
def metric_extractor(strat_test_path, pipeline):
    meatpipe = pipeline
    strat_test = strat_test_path
    # parses through each sample and adds metrics to dict which is then appended to a list
    masterList = []
    sampleCounter = 0

    for i in range(len(strat_test)):
        labelDict = {}
        sampleDict = {}
        pred_labels = []
        tempList = meatpipe(strat_test.iloc[i]["sequence"], max_length=512)
        label = strat_test.iloc[i]["labels"]
        sequence = strat_test.iloc[i]["sequence"]
        start = strat_test.iloc[i]["start"]
        stop = strat_test.iloc[i]["stop"]
        true_labels = labelLabel(label, sequence, start, stop)

        for aminoDict in tempList:
            labelNum = aminoDict["entity"]
            labelNum = int(labelNum[6:])
            pred_labels.append(labelNum)

        positions = evaluate_positions(true_labels, pred_labels)
        if type(positions) != list:
            print("DROPPED")
            continue
        else:
            trueStart = positions[0]
            trueStop = positions[1]
            predStart = positions[2]
            predStop = positions[3]

        eval_metrics = domainAcc(true_labels, pred_labels)
        domainLabelAcc = eval_metrics[0]
        notDomainLabelAcc = eval_metrics[1]
        fullDomainAcc = eval_metrics[2]

        items = matching_labels(true_labels, pred_labels, trueStart, trueStop) #change eval_start, eval_stop
        trueItems = items[0]
        predDomItems = items[1]
        predNotDomItems = items[2]

        if notDomainLabelAcc < 0:
            print("NEGATIVE VALUE")
        #         print(f"{notDomainLabelAcc} = {eval_metrics[5]} / {eval_metrics[6]}")
        #         print(start, stop)
        #         print(trueStart, trueStop)
        #         print(label)

        sampleDict["fullDomain_accuracy"] = float(fullDomainAcc)
        sampleDict["domain_accuracy"] = float(domainLabelAcc)
        sampleDict["notDomain_accuracy"] = float(notDomainLabelAcc)
        sampleDict["true_labels"] = list(trueItems)
        sampleDict["predDomain_labels"] = list(predDomItems)
        sampleDict["predNotDomain_labels"] = list(predNotDomItems)
        sampleDict["trueStart"] = trueStart
        sampleDict["trueStop"] = trueStop
        sampleDict["predStart"] = float(predStart)
        sampleDict["predStop"] = float(predStop)

        for i in trueItems:
            if i != 0:
                attentionLabel = str(i)

        labelDict[attentionLabel] = sampleDict

        masterList.append(labelDict)

        sampleCounter += 1
        if sampleCounter % 100 == 0:
            print(f"{sampleCounter} / {len(strat_test)}")
    #     print(sampleDict)
    #     print("=============================")
    return masterList

def metric_extractor_9(strat_test_path, pipeline):
    meatpipe = pipeline
    strat_test = strat_test_path
    # parses through each sample and adds metrics to dict which is then appended to a list
    masterList = []
    sampleCounter = 0

    for i in range(len(strat_test)):
        labelDict = {}
        sampleDict = {}
        pred_labels = []
        tempList = meatpipe(strat_test.iloc[i]["sequence"], max_length=512)
        label = strat_test.iloc[i]["labels"]
        sequence = strat_test.iloc[i]["sequence"]
        start = strat_test.iloc[i]["start"]
        stop = strat_test.iloc[i]["stop"]
        true_labels = labelLabel(label, sequence, start, stop)

        for aminoDict in tempList:
            labelNum = aminoDict["entity"]
            labelNum = int(labelNum[6:])
            pred_labels.append(labelNum)

        positions = evaluate_positions(true_labels, pred_labels, 9)
        if type(positions) != list:
            print("DROPPED")
            continue
        else:
            trueStart = positions[0]
            trueStop = positions[1]
            predStart = positions[2]
            predStop = positions[3]

        eval_metrics = domainAcc(true_labels, pred_labels)
        domainLabelAcc = eval_metrics[0]
        notDomainLabelAcc = eval_metrics[1]
        fullDomainAcc = eval_metrics[2]

        items = matching_labels(true_labels, pred_labels, trueStart, trueStop) #change eval_start, eval_stop
        trueItems = items[0]
        predDomItems = items[1]
        predNotDomItems = items[2]

        if notDomainLabelAcc < 0:
            print("NEGATIVE VALUE")
        #         print(f"{notDomainLabelAcc} = {eval_metrics[5]} / {eval_metrics[6]}")
        #         print(start, stop)
        #         print(trueStart, trueStop)
        #         print(label)

        sampleDict["fullDomain_accuracy"] = float(fullDomainAcc)
        sampleDict["domain_accuracy"] = float(domainLabelAcc)
        sampleDict["notDomain_accuracy"] = float(notDomainLabelAcc)
        sampleDict["true_labels"] = list(trueItems)
        sampleDict["predDomain_labels"] = list(predDomItems)
        sampleDict["predNotDomain_labels"] = list(predNotDomItems)
        sampleDict["trueStart"] = trueStart
        sampleDict["trueStop"] = trueStop
        sampleDict["predStart"] = float(predStart)
        sampleDict["predStop"] = float(predStop)

        for i in trueItems:
            if i != 0:
                attentionLabel = str(i)

        labelDict[attentionLabel] = sampleDict

        masterList.append(labelDict)

        sampleCounter += 1
        if sampleCounter % 100 == 0:
            print(f"{sampleCounter} / {len(strat_test)}")
    #     print(sampleDict)
    #     print("=============================")
    return masterList

def metric_extractor_12(strat_test_path, pipeline):
    meatpipe = pipeline
    strat_test = strat_test_path
    # parses through each sample and adds metrics to dict which is then appended to a list
    masterList = []
    sampleCounter = 0

    for i in range(len(strat_test)):
        labelDict = {}
        sampleDict = {}
        pred_labels = []
        tempList = meatpipe(strat_test.iloc[i]["sequence"], max_length=512)
        label = strat_test.iloc[i]["labels"]
        sequence = strat_test.iloc[i]["sequence"]
        start = strat_test.iloc[i]["start"]
        stop = strat_test.iloc[i]["stop"]
        true_labels = labelLabel(label, sequence, start, stop)

        for aminoDict in tempList:
            labelNum = aminoDict["entity"]
            labelNum = int(labelNum[6:])
            pred_labels.append(labelNum)

        positions = evaluate_positions(true_labels, pred_labels, 12)
        if type(positions) != list:
            print("DROPPED")
            continue
        else:
            trueStart = positions[0]
            trueStop = positions[1]
            predStart = positions[2]
            predStop = positions[3]

        eval_metrics = domainAcc(true_labels, pred_labels)
        domainLabelAcc = eval_metrics[0]
        notDomainLabelAcc = eval_metrics[1]
        fullDomainAcc = eval_metrics[2]

        items = matching_labels(true_labels, pred_labels, trueStart, trueStop)
        trueItems = items[0]
        predDomItems = items[1]
        predNotDomItems = items[2]

        if notDomainLabelAcc < 0:
            print("NEGATIVE VALUE")
        #         print(f"{notDomainLabelAcc} = {eval_metrics[5]} / {eval_metrics[6]}")
        #         print(start, stop)
        #         print(trueStart, trueStop)
        #         print(label)

        sampleDict["fullDomain_accuracy"] = float(fullDomainAcc)
        sampleDict["domain_accuracy"] = float(domainLabelAcc)
        sampleDict["notDomain_accuracy"] = float(notDomainLabelAcc)
        sampleDict["true_labels"] = list(trueItems)
        sampleDict["predDomain_labels"] = list(predDomItems)
        sampleDict["predNotDomain_labels"] = list(predNotDomItems)
        sampleDict["trueStart"] = trueStart
        sampleDict["trueStop"] = trueStop
        sampleDict["predStart"] = float(predStart)
        sampleDict["predStop"] = float(predStop)

        for i in trueItems:
            if i != 0:
                attentionLabel = str(i)

        labelDict[attentionLabel] = sampleDict

        masterList.append(labelDict)

        sampleCounter += 1
        if sampleCounter % 100 == 0:
            print(f"{sampleCounter} / {len(strat_test)}")
    #     print(sampleDict)
    #     print("=============================")
    return masterList

def metric_extractor_18(strat_test_path, pipeline):
    meatpipe = pipeline
    strat_test = strat_test_path
    # parses through each sample and adds metrics to dict which is then appended to a list
    masterList = []
    sampleCounter = 0

    for i in range(len(strat_test)):
        labelDict = {}
        sampleDict = {}
        pred_labels = []
        tempList = meatpipe(strat_test.iloc[i]["sequence"], max_length=512)
        label = strat_test.iloc[i]["labels"]
        sequence = strat_test.iloc[i]["sequence"]
        start = strat_test.iloc[i]["start"]
        stop = strat_test.iloc[i]["stop"]
        true_labels = labelLabel(label, sequence, start, stop)

        for aminoDict in tempList:
            labelNum = aminoDict["entity"]
            labelNum = int(labelNum[6:])
            pred_labels.append(labelNum)

        positions = evaluate_positions(true_labels, pred_labels, 18)
        if type(positions) != list:
            print("DROPPED")
            continue
        else:
            trueStart = positions[0]
            trueStop = positions[1]
            predStart = positions[2]
            predStop = positions[3]

        eval_metrics = domainAcc(true_labels, pred_labels)
        domainLabelAcc = eval_metrics[0]
        notDomainLabelAcc = eval_metrics[1]
        fullDomainAcc = eval_metrics[2]

        items = matching_labels(true_labels, pred_labels, trueStart, trueStop) #change eval_start, eval_stop
        trueItems = items[0]
        predDomItems = items[1]
        predNotDomItems = items[2]

        if notDomainLabelAcc < 0:
            print("NEGATIVE VALUE")
        #         print(f"{notDomainLabelAcc} = {eval_metrics[5]} / {eval_metrics[6]}")
        #         print(start, stop)
        #         print(trueStart, trueStop)
        #         print(label)

        sampleDict["fullDomain_accuracy"] = float(fullDomainAcc)
        sampleDict["domain_accuracy"] = float(domainLabelAcc)
        sampleDict["notDomain_accuracy"] = float(notDomainLabelAcc)
        sampleDict["true_labels"] = list(trueItems)
        sampleDict["predDomain_labels"] = list(predDomItems)
        sampleDict["predNotDomain_labels"] = list(predNotDomItems)
        sampleDict["trueStart"] = trueStart
        sampleDict["trueStop"] = trueStop
        sampleDict["predStart"] = float(predStart)
        sampleDict["predStop"] = float(predStop)

        for i in trueItems:
            if i != 0:
                attentionLabel = str(i)

        labelDict[attentionLabel] = sampleDict

        masterList.append(labelDict)

        sampleCounter += 1
        if sampleCounter % 100 == 0:
            print(f"{sampleCounter} / {len(strat_test)}")
    #     print(sampleDict)
    #     print("=============================")
    return masterList

def metric_extractor_24(strat_test_path, pipeline):
    meatpipe = pipeline
    strat_test = strat_test_path
    # parses through each sample and adds metrics to dict which is then appended to a list
    masterList = []
    sampleCounter = 0

    for i in range(len(strat_test)):
        labelDict = {}
        sampleDict = {}
        pred_labels = []
        tempList = meatpipe(strat_test.iloc[i]["sequence"], max_length=512)
        label = strat_test.iloc[i]["labels"]
        sequence = strat_test.iloc[i]["sequence"]
        start = strat_test.iloc[i]["start"]
        stop = strat_test.iloc[i]["stop"]
        true_labels = labelLabel(label, sequence, start, stop)

        for aminoDict in tempList:
            labelNum = aminoDict["entity"]
            labelNum = int(labelNum[6:])
            pred_labels.append(labelNum)

        positions = evaluate_positions(true_labels, pred_labels, 24)
        if type(positions) != list:
            print("DROPPED")
            continue
        else:
            trueStart = positions[0]
            trueStop = positions[1]
            predStart = positions[2]
            predStop = positions[3]

        eval_metrics = domainAcc(true_labels, pred_labels)
        domainLabelAcc = eval_metrics[0]
        notDomainLabelAcc = eval_metrics[1]
        fullDomainAcc = eval_metrics[2]

        items = matching_labels(true_labels, pred_labels, trueStart, trueStop) #change eval_start, eval_stop
        trueItems = items[0]
        predDomItems = items[1]
        predNotDomItems = items[2]

        if notDomainLabelAcc < 0:
            print("NEGATIVE VALUE")
        #         print(f"{notDomainLabelAcc} = {eval_metrics[5]} / {eval_metrics[6]}")
        #         print(start, stop)
        #         print(trueStart, trueStop)
        #         print(label)

        sampleDict["fullDomain_accuracy"] = float(fullDomainAcc)
        sampleDict["domain_accuracy"] = float(domainLabelAcc)
        sampleDict["notDomain_accuracy"] = float(notDomainLabelAcc)
        sampleDict["true_labels"] = list(trueItems)
        sampleDict["predDomain_labels"] = list(predDomItems)
        sampleDict["predNotDomain_labels"] = list(predNotDomItems)
        sampleDict["trueStart"] = trueStart
        sampleDict["trueStop"] = trueStop
        sampleDict["predStart"] = float(predStart)
        sampleDict["predStop"] = float(predStop)

        for i in trueItems:
            if i != 0:
                attentionLabel = str(i)

        labelDict[attentionLabel] = sampleDict

        masterList.append(labelDict)

        sampleCounter += 1
        if sampleCounter % 100 == 0:
            print(f"{sampleCounter} / {len(strat_test)}")
    #     print(sampleDict)
    #     print("=============================")
    return masterList

def evaluate_positions_1(true, predict, buffer=6):

    def check_ahead(position, predict, buffer):
        ret = True
        for i in range(1, buffer + 1):
            if predict[position + i] != 0:
                ret = False
                break
        return ret

    switch1 = True
    switch2 = True
    predStart = None
    trueStart = None
    predStop = None
    trueStop = None


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
            # switch1 = None
            # predStop = i - 1
# =======================================================
            if i < ((len(true) - 1) - buffer):
                checkAhead = check_ahead(i, predict, buffer)
                if checkAhead:
                    switch1 = None
                    predStop = i - 1
                else:
                    switch1 = False

            else:  # how to make it a decaying buffer near end?
                difference = i - ((len(true) - 1) - buffer)
                diff = buffer - difference
                checkAhead = check_ahead(i, predict, diff)
                if checkAhead:
                    switch1 = None
                    predStop = i - 1
                else:
                    switch1 = False


    if switch2 == False and trueStop == None:
        trueStop = len(true) - 1

    if switch1 == False and predStop == None:
        predStop = len(predict) - 1

    if trueStart == None and switch2 == True:
        print("HAAAAAAAAAAAAAAAAAAA NO DOMAIN")
        print(trueStart, trueStop, predStart, predStop)
        return False
    else:
        return [trueStart, trueStop, predStart, predStop]

def metric_extractor_1(strat_test_path, pipeline):
    meatpipe = pipeline
    strat_test = strat_test_path
    # parses through each sample and adds metrics to dict which is then appended to a list
    masterList = []
    sampleCounter = 0

    for i in range(len(strat_test)):
        labelDict = {}
        sampleDict = {}
        pred_labels = []
        tempList = meatpipe(strat_test.iloc[i]["sequence"], max_length=512)
        label = strat_test.iloc[i]["labels"]
        sequence = strat_test.iloc[i]["sequence"]
        start = strat_test.iloc[i]["start"]
        stop = strat_test.iloc[i]["stop"]
        true_labels = labelLabel(label, sequence, start, stop)

        for aminoDict in tempList:
            labelNum = aminoDict["entity"]
            labelNum = int(labelNum[6:])
            pred_labels.append(labelNum)

        positions = evaluate_positions_1(true_labels, pred_labels)
        if type(positions) != list:
            print("DROPPED")
            continue
        else:
            trueStart = positions[0]
            trueStop = positions[1]
            predStart = positions[2]
            predStop = positions[3]

        eval_metrics = domainAcc(true_labels, pred_labels)
        domainLabelAcc = eval_metrics[0]
        notDomainLabelAcc = eval_metrics[1]
        fullDomainAcc = eval_metrics[2]

        items = matching_labels(true_labels, pred_labels, trueStart, trueStop) #change eval_start, eval_stop
        trueItems = items[0]
        predDomItems = items[1]
        predNotDomItems = items[2]

        if notDomainLabelAcc < 0:
            print("NEGATIVE VALUE")
        #         print(f"{notDomainLabelAcc} = {eval_metrics[5]} / {eval_metrics[6]}")
        #         print(start, stop)
        #         print(trueStart, trueStop)
        #         print(label)

        sampleDict["fullDomain_accuracy"] = float(fullDomainAcc)
        sampleDict["domain_accuracy"] = float(domainLabelAcc)
        sampleDict["notDomain_accuracy"] = float(notDomainLabelAcc)
        sampleDict["true_labels"] = list(trueItems)
        sampleDict["predDomain_labels"] = list(predDomItems)
        sampleDict["predNotDomain_labels"] = list(predNotDomItems)
        sampleDict["trueStart"] = trueStart
        sampleDict["trueStop"] = trueStop
        sampleDict["predStart"] = float(predStart)
        sampleDict["predStop"] = float(predStop)

        for i in trueItems:
            if i != 0:
                attentionLabel = str(i)

        labelDict[attentionLabel] = sampleDict

        masterList.append(labelDict)

        sampleCounter += 1
        if sampleCounter % 100 == 0:
            print(f"{sampleCounter} / {len(strat_test)}")
    #     print(sampleDict)
    #     print("=============================")
    return masterList

def group_labels(masterList):
    # parses through masterList to group the metrics of each sample by label for easier evaluation
    masterEvalDict = {}
    sampleCounter = 0
    for labelDictLevel in masterList:
        lab = list(labelDictLevel.keys())
        lab = lab[0]
        labelDict = labelDictLevel[lab]
        if lab not in masterEvalDict:
            newDict = {
                "fullDomain_accuracy": [labelDict["fullDomain_accuracy"]],
                "domain_accuracy": [labelDict["domain_accuracy"]],
                "notDomain_accuracy": [labelDict["notDomain_accuracy"]],
                "true_labels": list(labelDict["true_labels"]),
                "predDomain_labels": list(labelDict["predDomain_labels"]),
                "predNotDomain_labels": list(labelDict["predNotDomain_labels"])
            }
            if labelDict["trueStart"] == labelDict["predStart"]:
                newDict["start"] = [1]
            else:
                newDict["start"] = [0]

            if labelDict["trueStop"] == labelDict["predStop"]:
                newDict["stop"] = [1]
            else:
                newDict["stop"] = [0]

            masterEvalDict[lab] = newDict
        else:
            innerLabelDict = masterEvalDict[lab]
            innerLabelDict["fullDomain_accuracy"].append(labelDict["fullDomain_accuracy"])
            innerLabelDict["domain_accuracy"].append(labelDict["domain_accuracy"])
            innerLabelDict["notDomain_accuracy"].append(labelDict["notDomain_accuracy"])

            innerLabelDict["true_labels"] = list(innerLabelDict["true_labels"])
            innerLabelDict["true_labels"] += labelDict["true_labels"]
            innerLabelDict["true_labels"] = set(innerLabelDict["true_labels"])
            innerLabelDict["true_labels"] = list(innerLabelDict["true_labels"])

            innerLabelDict["predDomain_labels"] = list(innerLabelDict["predDomain_labels"])
            innerLabelDict["predDomain_labels"] += labelDict["predDomain_labels"]
            innerLabelDict["predDomain_labels"] = set(innerLabelDict["predDomain_labels"])
            innerLabelDict["predDomain_labels"] = list(innerLabelDict["predDomain_labels"])

            innerLabelDict["predNotDomain_labels"] = list(innerLabelDict["predNotDomain_labels"])
            innerLabelDict["predNotDomain_labels"] += labelDict["predNotDomain_labels"]
            innerLabelDict["predNotDomain_labels"] = set(innerLabelDict["predNotDomain_labels"])
            innerLabelDict["predNotDomain_labels"] = list(innerLabelDict["predNotDomain_labels"])

            if labelDict["trueStart"] == labelDict["predStart"]:
                innerLabelDict["start"].append(1)
            else:
                innerLabelDict["start"].append(0)

            if labelDict["trueStop"] == labelDict["predStop"]:
                innerLabelDict["stop"].append(1)
            else:
                innerLabelDict["stop"].append(0)

        sampleCounter += 1
        if sampleCounter % 100 == 0:
            print(f"{sampleCounter} / {len(masterList)}")

    return masterEvalDict

def average_metrics(masterEvalDict):
    #compiles the metrics for each label into a label average
    resultDict = {}
    sampleCounter = 0


    for eachLabel in masterEvalDict:
        innerLabDict = {}

        metricData = masterEvalDict[eachLabel]
        fullDom_acc = sum(metricData["fullDomain_accuracy"]) / len(metricData["fullDomain_accuracy"])
        dom_acc = sum(metricData["domain_accuracy"]) / len(metricData["domain_accuracy"])
        notDom_acc = sum(metricData["notDomain_accuracy"]) / len(metricData["notDomain_accuracy"])

        true_lab = list(metricData["true_labels"])
        predDomLab = list(metricData["predDomain_labels"])
        predNotdomLab = list(metricData["predNotDomain_labels"])

        start_acc = sum(metricData["start"]) / len(metricData["start"])
        stop_acc = sum(metricData["stop"]) / len(metricData["stop"])

        innerLabDict["fullDomain_accuracy"] = fullDom_acc
        innerLabDict["domain_accuracy"] = dom_acc
        innerLabDict["notDomain_accuracy"] = notDom_acc

        innerLabDict["true_labels"] = true_lab
        innerLabDict["pred_domain_labels"] = predDomLab
        innerLabDict["pred_notDomain_labels"] = predNotdomLab

        innerLabDict["start_acc"] = start_acc
        innerLabDict["stop_acc"] = stop_acc

        resultDict[eachLabel] = innerLabDict

        sampleCounter += 1
        print(f"{sampleCounter} / {len(masterEvalDict)}")
    return resultDict


def present_labels(csvPath, sample_num):
    df = pd.read_csv(csvPath)

    print(df.columns)
    df = df.drop(["Unnamed: 0"], axis=1)
    print(df.columns)

    # dropped duplicate rows
    print(len(df))
    df.drop_duplicates(subset=["sequence", "labels", "start", "stop"],
                       inplace=True)  # if we removed "start" we would have 618024 sequences instead, one sequence is exactly the same with just a different start site
    print(len(df))
    df.reset_index(inplace=True)
    df = df.drop(["index"], axis=1)
    df.tail()

    # shows the frequency of each label after removing duplicates
    frequencyDictAfter = {}

    for index, row in df.iterrows():
        if row["labels"] not in frequencyDictAfter:
            frequencyDictAfter[row["labels"]] = 1
        else:
            frequencyDictAfter[row["labels"]] += 1

    # counter = 0
    # for i in frequencyDictAfter:
    #     print(i, frequencyDictAfter[i])
    #     counter += 1
    # print(counter)

    # list out labels with less than 100 samples and remove
    dropList = []  # list of labels to drop as they contain less than 100 samples
    numToDrop = []
    amount = 0
    for i in frequencyDictAfter:
        if frequencyDictAfter[i] < sample_num:
            # print(i, frequencyDict[i])
            dropList.append(i)
            numToDrop.append(frequencyDictAfter[i])
            amount += 1
    print(len(dropList))
    dropList.remove("AA_not_domain")
    print(len(dropList))
    print("=====")
    print(len(numToDrop))
    numToDrop.pop(len(numToDrop) - 1)
    print(len(numToDrop))

    # drops the rows with labels in dropList (removes labels with too few sequences)
    print(len(df))
    df.set_index("labels", inplace=True)
    df.drop(dropList, axis=0, inplace=True)
    df.reset_index(inplace=True)
    print(len(df))

    # shows the frequency of each label after removing sequences with less than 100 samples
    frequencyDictAfter = {}

    for index, row in df.iterrows():
        if row["labels"] not in frequencyDictAfter:
            frequencyDictAfter[row["labels"]] = 1
        else:
            frequencyDictAfter[row["labels"]] += 1

    counter = 0
    for i in frequencyDictAfter:
        #     print(i, frequencyDictAfter[i])
        counter += 1
    print(f"number of labels: {counter}")

    # encode labels and print a list of them with corresponding number label (goes alphabetically)

    le = preprocessing.LabelEncoder()
    le.fit(df["labels"])

    printList = []
    t = list(le.classes_)
    n = 0
    for i in t:
        print(f"{n}) {i}")
        printList.append(i)
        n += 1
    print("")
    print(printList)
    return printList


def fullDomain_acc(resultDict, domainList):
    # fullDomain Accuracy
    i80 = []
    i60 = []
    i40 = []
    i20 = []

    for i in resultDict:
        tempDict = {}
        dic = resultDict[i]
        tempDict[i] = resultDict[i]
        if (dic["fullDomain_accuracy"]) >= .8:
            i80.append(tempDict)
        elif (dic["fullDomain_accuracy"]) >= .6:
            i60.append(tempDict)
        elif (dic["fullDomain_accuracy"]) >= .4:
            i40.append(tempDict)
        elif (dic["fullDomain_accuracy"]) >= .2:
            i20.append(tempDict)

    csvDict = {"label": [],
               "fullDomain_accuracy": []
               }

    print("|||||||||||||||FULLDOMAIN ACCURACY|||||||||||||||")
    print("===i80===")
    for i in i80:
        #     print(i)
        for p in i:
            print(p, i[p]["fullDomain_accuracy"])
            csvDict["label"].append(domainList[int(p)])
            csvDict["fullDomain_accuracy"].append(i[p]["fullDomain_accuracy"])
    print("===i60===")
    for i in i60:
        #     print(i)
        for p in i:
            print(p, i[p]["fullDomain_accuracy"])
            csvDict["label"].append(domainList[int(p)])
            csvDict["fullDomain_accuracy"].append(i[p]["fullDomain_accuracy"])
    print("===i40===")
    for i in i40:
        #     print(i)
        for p in i:
            print(p, i[p]["fullDomain_accuracy"])
            csvDict["label"].append(domainList[int(p)])
            csvDict["fullDomain_accuracy"].append(i[p]["fullDomain_accuracy"])
    print("===i20===")
    for i in i20:
        #     print(i)
        for p in i:
            print(p, i[p]["fullDomain_accuracy"])
            csvDict["label"].append(domainList[int(p)])
            csvDict["fullDomain_accuracy"].append(i[p]["fullDomain_accuracy"])

    return csvDict


def domain_acc(resultDict, domainList):
    # domain Accuracy
    i80 = []
    i60 = []
    i40 = []
    i20 = []

    for i in resultDict:
        tempDict = {}
        dic = resultDict[i]
        tempDict[i] = resultDict[i]

        if dic["domain_accuracy"] >= .8:
            i80.append(tempDict)
        elif dic["domain_accuracy"] >= .6:
            i60.append(tempDict)
        elif dic["domain_accuracy"] >= .4:
            i40.append(tempDict)
        elif dic["domain_accuracy"] >= .2:
            i20.append(tempDict)

    csvDict2 = {"label": [],
                "domain_accuracy": []
                }
    print("|||||||||||||||DOMAIN ACCURACY|||||||||||||||")
    print("===i80===")
    for i in i80:
        #     print(i)
        for p in i:
            print(p, i[p]["domain_accuracy"])
            csvDict2["label"].append(domainList[int(p)])
            csvDict2["domain_accuracy"].append(i[p]["domain_accuracy"])
    print("===i60===")
    for i in i60:
        #     print(i)
        for p in i:
            print(p, i[p]["domain_accuracy"])
            csvDict2["label"].append(domainList[int(p)])
            csvDict2["domain_accuracy"].append(i[p]["domain_accuracy"])
    print("===i40===")
    for i in i40:
        #     print(i)
        for p in i:
            print(p, i[p]["domain_accuracy"])
            csvDict2["label"].append(domainList[int(p)])
            csvDict2["domain_accuracy"].append(i[p]["domain_accuracy"])
    print("===i20===")
    for i in i20:
        #     print(i)
        for p in i:
            print(p, i[p]["domain_accuracy"])
            csvDict2["label"].append(domainList[int(p)])
            csvDict2["domain_accuracy"].append(i[p]["domain_accuracy"])

    return csvDict2

def notDomain_acc(resultDict, domainList):
    #notdomain Accuracy
    i80 = []
    i60 = []
    i40 = []
    i20 = []

    for i in resultDict:
        tempDict = {}
        dic = resultDict[i]
        tempDict[i] = resultDict[i]

        if dic["notDomain_accuracy"] >= .8:
            i80.append(tempDict)
        elif dic["notDomain_accuracy"] >= .6:
            i60.append(tempDict)
        elif dic["notDomain_accuracy"] >= .4:
            i40.append(tempDict)
        elif dic["notDomain_accuracy"] >= .2:
            i20.append(tempDict)


    csvDict3 = {"label": [],
              "notDomain_accuracy": []
              }
    print("|||||||||||||||notDOMAIN ACCURACY|||||||||||||||")
    print("===i80===")
    for i in i80:
    #     print(i)
        for p in i:
            print(p, i[p]["notDomain_accuracy"])
            csvDict3["label"].append(domainList[int(p)])
            csvDict3["notDomain_accuracy"].append(i[p]["notDomain_accuracy"])
    print("===i60===")
    for i in i60:
    #     print(i)
        for p in i:
            print(p, i[p]["notDomain_accuracy"])
            csvDict3["label"].append(domainList[int(p)])
            csvDict3["notDomain_accuracy"].append(i[p]["notDomain_accuracy"])
    print("===i40===")
    for i in i40:
    #     print(i)
        for p in i:
            print(p, i[p]["notDomain_accuracy"])
            csvDict3["label"].append(domainList[int(p)])
            csvDict3["notDomain_accuracy"].append(i[p]["notDomain_accuracy"])
    print("===i20===")
    for i in i20:
    #     print(i)
        for p in i:
            print(p, i[p]["notDomain_accuracy"])
            csvDict3["label"].append(domainList[int(p)])
            csvDict3["notDomain_accuracy"].append(i[p]["notDomain_accuracy"])
    return csvDict3