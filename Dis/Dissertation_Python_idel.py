import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from deepbrain import Extractor
import scikitplot as skplt
Brain_Size = 1000000
CN = 0
EMCI = 1
LMCI = 2

Brain_Data = []
Brain_Data_reduced = [0,0,0,0]
Brains = []
Brain_Types = []
listofLMCI = []
listofEMCI = []
listofCN = []
 
def Reduce_Brain(img):
    print("Brain Reduction Start")
    Brain_Data_Small = []
    prob = ext.run(img) 
    mask = prob > 0.5
    for i in range (len(mask)):
        for j in range(len(mask[i])):
            for k in range(len(mask[i][j])):
                if mask[i][j][k]:
                    Brain_Data_Small.append(img[i][j][k])
    print("Brain size pre processed", len(Brain_Data_Small))
    if(len(Brain_Data_Small) != Brain_Size):
        difrence = len(Brain_Data_Small) - Brain_Size
        if(difrence < 0):
            amount = (difrence*-1)/2
            amount = int(round(amount))
            print("Diffrence needed", amount*2)
            for i in range(amount):
                Brain_Data_Small.insert(0, 0)
                Brain_Data_Small.append(0)
        if(difrence > 0):
            amount = difrence/2
            amount = int(round(amount))
            print("Diffrence needed", amount*2)
            for i in range(amount):
                Brain_Data_Small.pop(0)
                Brain_Data_Small.pop(len(Brain_Data_Small)-1)
    if(len(Brain_Data_Small) > Brain_Size):
        Brain_Data_Small.pop(0)
    if(len(Brain_Data_Small) < Brain_Size):
        Brain_Data_Small.append(0)
    print("Brain size post processed",len(Brain_Data_Small))
    print("Brain Reduction End")
    return Brain_Data_Small

listofLMCI = os.listdir(".\DataSet\LMCI")
listofEMCI = os.listdir(".\DataSet\EMCI")
listofCN = os.listdir(".\DataSet\CN")
print("Start brain extraction model")
ext = Extractor()
print("End brain extraction model")

for i in range(len(listofCN)):
    #CN
    file = os.path.join(".\DataSet\CN",listofCN[i])
    img = nib.load(file).get_fdata()
    #Brain_Data_reduced = Reduce_Brain(img)
    filename = "./CN/CN_Number:.txt"
    print(filename)
    f= open(filename,"w+")
    for j in range(len(Brain_Data_reduced)):
        f.write("%d\r\n" % Brain_Data_reduced[j])
    f.close()
    Brains.append(Brain_Data_reduced)
    Brain_Types.append(CN)
    #EMCI
    file = os.path.join(".\DataSet\EMCI",listofEMCI[i])
    img = nib.load(file).get_fdata()
    Brain_Data_reduced = Reduce_Brain(img)
    f= open("EMCI"+listofEMCI[i]+".txt","w+")
    for j in range(10):
        f.write("This is lineemci %d\r\n" % (j+1))
    f.close()
    Brains.append(Brain_Data_reduced)
    Brain_Types.append(EMCI)
    #LMCI
    file = os.path.join(".\DataSet\LMCI",listofLMCI[i])
    img = nib.load(file).get_fdata()
    Brain_Data_reduced = Reduce_Brain(img)
    f= open("LMCI"+listofLMCI[i]+".txt","w+")
    for j in range(10):
        f.write("This is linelcmi %d\r\n" % (j+1))
    f.close()
    Brains.append(Brain_Data_reduced)
    Brain_Types.append(LMCI)

print("Done")
