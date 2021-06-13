from Bio import SeqIO #parse fasta files
import os #change directories
import time #track processing time
import pandas as pd #create dataframes
from datasets import load_dataset #load dataframe into huggingface dataset
from sklearn import preprocessing #encode labels
from sklearn.model_selection import train_test_split

import json

#tokenize
from transformers import BertModel, BertTokenizer
import re

import torch


#load encoded csv
filename = "encoded_parsed_pullin>100_pure.csv"

labelled_df = pd.read_csv(f"/mnt/storage/grid/home/eric/hmm2bert/pullin_parsed_data/{filename}")

print(labelled_df.columns)
labelled_df = labelled_df.drop(["Unnamed: 0"], axis=1)
print(labelled_df.columns)

#stratify split dataframe into training split and validation split
train_split, val_split = train_test_split(labelled_df, stratify=labelled_df["labels"], test_size=.2, random_state=420)#stratify=masterLabel
print(len(train_split))
print(len(val_split))
print("========")

# parses through each row and creates 2 lists with single amino acid and label
# ====
masterAcid = []
masterLabel = []
# ====

for index, row in train_split.iterrows():  # iterates over rows
    # ====
    start = row["start"]
    stop = row["stop"]
    sequence = row["sequence"]
    labels = row["labels"]
    domainList = list(range(start, stop + 1))
    sequenceIndexCounter = 0
    singleAcid = []
    singleLabel = []
    # ====
    # print(start, stop)
    # print(domainList[0], domainList[-1])

    for currentNum in range(len(sequence)):  # iterate over each amino acid in a single sequence
        # ====
        currentAcid = sequence[currentNum]
        shiftedNum = currentNum + 1  # index that we compare to as sequence index starts at zero but start/stop starts at 1
        # ====

        if shiftedNum == domainList[sequenceIndexCounter]:
            singleLabel.append(labels)
            singleAcid.append(currentAcid)
            if domainList[sequenceIndexCounter] == domainList[-1]:
                sequenceIndexCounter = 0
            else:
                sequenceIndexCounter += 1

        else:
            singleLabel.append(-100)  # need to add "not domain" label to encoder
            singleAcid.append(currentAcid)

    # print(len(singleAcid))
    # print(len(singleLabel))

    masterAcid.append(singleAcid)
    masterLabel.append(singleLabel)


print(len(masterAcid))
print(len(masterLabel))
print("========")


for i in range(len(masterLabel)):
    if len(masterLabel[i]) < 512:
        while len(masterLabel[i]) < 512:
            masterLabel[i].append(-100)
    elif len(masterLabel[i]) > 512:
        masterLabel[i] = masterLabel[i][:512]

print(masterAcid[0])
print(len(masterAcid[0]))
print(masterLabel[0])
print(len(masterLabel[0]))
print("========")
print("TOKENIZING")

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
encoded_train = tokenizer(masterAcid, is_split_into_words=True, padding=True, max_length=512, truncation=True)


# create dataset object



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


train_dataset = TokenDataset(encoded_train, masterLabel)


filename1 = "embedding_pullin_train>100_stratified.pt"

torch.save(train_dataset, f"/mnt/storage/grid/home/eric/hmm2bert/pullin_parsed_data/{filename1}")
