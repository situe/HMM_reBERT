import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch


# load encoded csv


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

csvFilePath = "/mnt/storage/grid/home/eric/hmm2bert/pullin_parsed_data/encoded_parsed_pullin_noDupes>1000_withAA_not_domain.csv"
fileNamePath = "/mnt/storage/grid/home/eric/hmm2bert/pullin_parsed_data/embedding_pullin_noDupes_val>1000_stratified_individualPiece.pt"

labelled_df = pd.read_csv(csvFilePath)

print(labelled_df.columns)
labelled_df = labelled_df.drop(["Unnamed: 0"], axis=1)
print(labelled_df.columns)
print(f"length: {len(labelled_df)}")

labelled_df.drop(len(labelled_df) - 1, inplace=True)
labelled_df.drop(len(labelled_df) - 1, inplace=True)

train_split, val_split = train_test_split(labelled_df, test_size=.2, random_state=420)  # stratify=masterLabel
print(len(train_split))
print(len(val_split))

# parses through each row and creates 2 lists with single amino acid and label
# ====


MasterAcid = []
MasterLabel = []
# ====

for index, row in val_split.iterrows():  # iterates over rows
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
            singleLabel.append(0)
            singleAcid.append(currentAcid)

    MasterAcid.append(singleAcid)
    MasterLabel.append(singleLabel)

print(MasterAcid[-1])
print(MasterLabel[-1])
print(len(MasterAcid))
print(len(MasterLabel))
print("=======")

for i in range(len(MasterLabel)):
    if len(MasterLabel[i]) < 512:
        while len(MasterLabel[i]) < 512:
            MasterLabel[i].append(0)
    elif len(MasterLabel[i]) > 512:
        MasterLabel[i] = MasterLabel[i][:512]

print(MasterAcid[0])
print(len(MasterAcid[0]))
print(MasterLabel[0])
print(len(MasterLabel[0]))
print("========")
print("TOKENIZING")

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
encoded_val = tokenizer(MasterAcid, is_split_into_words=True, padding=True, max_length=512, truncation=True)

val_dataset = TokenDataset(encoded_val, MasterLabel)







torch.save(val_dataset, fileNamePath)