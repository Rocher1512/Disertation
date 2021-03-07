import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import nibabel as nb
from deepbrain import Extractor
import scikitplot as skplt
Brain_Size = 1400000
CN = 0
EMCI = 1
LMCI = 2

Brain_Data = []
Brain_Data_reduced = []
Brains = []
Brain_Types = []
listofLMCI = []
listofEMCI = []
listofCN = []
 
def Reduce_Brain(img):
    print("Reducing a brain")
    Brain_Data_Small = []
    prob = ext.run(img) 
    mask = prob > 0.5
    for i in range (len(mask)):
        for j in range(len(mask[i])):
            for k in range(len(mask[i][j])):
                if mask[i][j][k]:
                    Brain_Data_Small.append(img[i][j][k])
    print(type(Brain_Data_Small[0]))
    while(len(Brain_Data_Small) < Brain_Size):
        if(len(Brain_Data_Small) % 2 == 0):
            Brain_Data_Small.append(0)
        else:
            Brain_Data_Small.insert(0,0)
    print("the brain is: ",len(Brain_Data_Small))
    return Brain_Data_Small

listofLMCI = os.listdir(".\DataSet\LMCI")
listofEMCI = os.listdir(".\DataSet\EMCI")
listofCN = os.listdir(".\DataSet\CN")
print("start")
ext = Extractor()
print("End")

for i in range(len(listofCN)):
    print("collecting 3 brains")
    #CN
    print(len(Brains))
    print(len(Brain_Types))
    file = os.path.join(".\DataSet\CN",listofCN[i])
    img = nib.load(file).get_fdata()
    Brain_Data_reduced = Reduce_Brain(img)
    Brains.append(Brain_Data_reduced)
    Brain_Types.append(CN)
    #EMCI
    file = os.path.join(".\DataSet\EMCI",listofEMCI[i])
    img = nib.load(file).get_fdata()
    Brain_Data_reduced = Reduce_Brain(img)
    Brains.append(Brain_Data_reduced)
    Brain_Types.append(EMCI)
    #LMCI
    file = os.path.join(".\DataSet\LMCI",listofLMCI[i])
    img = nib.load(file).get_fdata()
    Brain_Data_reduced = Reduce_Brain(img)
    Brains.append(Brain_Data_reduced)
    Brain_Types.append(LMCI)
print(len(Brains))
print(len(Brain_Types))
print(len(Brains[0]))
print(Brain_Types[0])
X_train, X_test, y_train, y_test = train_test_split(Brains, Brain_Types, test_size=0.33)
model = MLPClassifier(solver='sgd', alpha = 0.0001, hidden_layer_sizes=(70,70,70), activation = 'tanh', learning_rate = 'invscaling')

trained_model = model.fit(X_train, y_train)
predict = trained_model.predict(X_test)
print(predict)
print(y_test)
print(len(predict))
print(len(y_test))
f= open("Data.txt","w+")
for i in range(len(predict)):
    f.write("Prediction: ",predict[i])
    f.write("Acctual: ",y_test[i])
f.close() 
