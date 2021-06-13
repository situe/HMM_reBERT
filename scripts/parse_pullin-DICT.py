import os
import json

#=====
writeFile = "parsed_pullin_dict.json"
listWrapper = []
fileCounter = 0
#=====


for fileName in os.listdir(f"/mnt/storage/grid/var/pullin_data/prism"):

#         print("")
#         print("")
#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(f"{len(listWrapper)}/{fileCounter}", fileName)
#         print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#         print("")
    #=====
    fileDict = {}
    #=====

    with open(f"/mnt/storage/grid/var/pullin_data/prism/{fileName}", "r") as file:
        #=====
        orfDict = {}
        loaded_file = json.load(file)
        resultsDict = loaded_file["prism_results"]
        inputDict = resultsDict["input"]
        clustersDict = resultsDict["clusters"] #returns a list of dicts
        numCounter = 0
        #=====
        orfDict["hash"] = inputDict["hash"]
        for i in range(len(clustersDict)): #iterate through list of dicts
            #=====
            suborfDict = {}
            counter = 0
            currentCluster = clustersDict[i] #returns a dict with info of 1 cluster
            orfList = currentCluster["orfs"] #returns a list of dicts, where each dict contains info for 1 orf
            #=====
            for singleORF in orfList: #iterate through each dict in the list of dicts (each dict in list is 1 orf)
                #=====
                dataDict = {}
                domain = singleORF["domains"]
                #=====
                if len(domain) > 0:
                    #=====
                    isoDomain = domain[0] #returns a dict of the orf that contains a domain

                    #=====
#                         print(f"ORF: {numCounter}", "=======================")
#                         print("subORF: ", counter)
#                         print(isoDomain["start"], isoDomain["stop"])
#                         print(singleORF["sequence"])
#                         print(isoDomain["full_name"])
#                         print("       ====================")
#                         print("")

                    dataDict["full_name"] = isoDomain["full_name"]
                    dataDict["start"] = isoDomain["start"]
                    dataDict["stop"] = isoDomain["stop"]
                    dataDict["sequence"] = singleORF["sequence"]

                    suborfDict[f"subORF-{counter}"] = dataDict
                counter += 1
#                 print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
#                 print("")
            orfDict[f"ORF-{numCounter}"] = suborfDict
            numCounter += 1

    fileDict[f"{fileName}"] = orfDict
    if len(fileDict[fileName].keys()) > 1:
        listWrapper.append(fileDict)
        fileCounter += 1
    else:
        fileCounter += 1
        continue
with open(f"/mnt/storage/grid/home/eric/hmm2bert/pullin_parsed_data/{writeFile}", "r+") as json_file:
    json.dump(listWrapper, json_file)