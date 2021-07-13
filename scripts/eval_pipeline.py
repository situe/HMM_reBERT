from reBERTer import *
import os
from transformers import BertTokenizer
from transformers import pipeline
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import json

####################################################
gpu_idx = 0
num_labels = 60
sample_num = 1000   #sample number (keep domains with greater than X samples)

#model to evaluate with
model_filename = "pullin>1000_whiteSpace_best_loss-epch9{1GPU}.pt"
model_path = f"/mnt/storage/grid/home/eric/hmm2bert/models/pullin/{model_filename}"

#encoded csv to use
data_folder = "pullin_parsed_data"
encoded_label_filename = "encoded_parsed_pullin_noDupes_whiteSpace>1000_withAA_not_domain.csv"
encoded_csv = f"/mnt/storage/grid/home/eric/hmm2bert/{data_folder}/{encoded_label_filename}"

#path of parsed pullin for ordered list of labels
csvName = "parsed_pullin_sequences>100_whiteSpace_withAA_not_domain_uneven.csv"
csvPath = f"/mnt/storage/grid/home/eric/hmm2bert/pullin_parsed_data/{csvName}"


########################################
#stop buffer number (if there is one, leave as empty string if no buffer num)
buffer_num = ""
#raw metrics .json file path
part1 = model_filename[:-3]
part2 = "eval_results"
part3 = "raw_metrics.json"
part4 = "metric_results.json"
part5 = "fullDomain_accuracy.csv"
part6 = "domain_accuracy.csv"
part7 = "notDomain_accuracy.csv"
if buffer_num:
    part3 = buffer_num + part3
    part4 = buffer_num + part4
    part5 = buffer_num + part5
    part6 = buffer_num + part6
    part7 = buffer_num + part7



#directory path
pot_path = f"/mnt/storage/grid/home/eric/hmm2bert/pullin_parsed_data/{part1 + part2}"

#raw metrics .json file path
raw_json = f"{pot_path}/{part3}"

#final results .json file path
result_json = f"{pot_path}/{part4}"

#fullDomain accuracy csv path
fullDom_path = f"{pot_path}/{part5}"

#domain accuracy csv path
domain_path = f"{pot_path}/{part6}"

#notDomain accuracy csv path
notDomain_path = f"{pot_path}/{part7}"

####################################################

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

#create eval directory if doesnt exist (shouldnt exist)
if os.path.exists(pot_path):
    print("DIRECTORY ALREADY EXISTS")
else:
    os.mkdir(pot_path)
    print("EVAL DIRECTORY CREATED")

#load in tokenizer, model (eval mode) and pipeline
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = torch.load(model_path)
model.eval()
meatpipe = pipeline(task='ner', model=model.bert, tokenizer=tokenizer, device=gpu_idx)

# load csv to pandas dataframe
df = pd.read_csv(encoded_csv)
print(df.columns)
df = df.drop(["Unnamed: 0"], axis=1)
print(df.columns)

#split the dataset into train and test, this produces a list with the row positions
strat_train, strat_test = train_test_split(df, test_size=.2, stratify=df['labels'], random_state=420)

#create the ordered list of labels
labelList = present_labels(csvPath, sample_num)

#extract metrics
masterList = metric_extractor(strat_test, meatpipe)

#dump raw metrics into json file for quick pullup
with open(raw_json, "w") as file:
    json.dump(masterList, file)

#group metrics by label
masterEvalDict = group_labels(masterList)

#average metrics
resultDict = average_metrics(masterEvalDict)

with open(result_json, "w") as file:
    json.dump(resultDict, file)

#get fulldomain accuracy and save results as csv
fullDom_acc = fullDomain_acc(resultDict, labelList)
fullDom_df = pd.DataFrame(fullDom_acc)
fullDom_df.to_csv(fullDom_path)

#get domain accuracy and save results as csv
dom_acc = domain_acc(resultDict, labelList)
dom_df = pd.DataFrame(dom_acc)
dom_df.to_csv(domain_path)

#get notDomain accuracy and save results as csv
notDom_acc = notDomain_acc(resultDict, labelList)
notDom_df = pd.DataFrame(notDom_acc)
notDom_df.to_csv(notDomain_path)

