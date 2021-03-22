import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

Brains = []
Brain_Types = []
Brain_Data = []
listofLMCI = []
listofEMCI = []
listofCN = []

listofLMCI = os.listdir("./LMCI")
listofEMCI = os.listdir("./EMCI")
listofCN = os.listdir("./CN")
CN = 0
EMCI = 1
LMCI = 2

for i in range(len(listofCN)):
    print(i)
    Brain_Data = []
    #cn
    file = os.path.join("./CN",listofCN[i])
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            x = line.strip()
            Brain_Data.append(x)
    Brains.append(Brain_Data)
    Brain_Types.append(CN)

    #emci
    Brain_Data = []
    file = os.path.join("./EMCI",listofEMCI[i])
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            x = line.strip()
            Brain_Data.append(x)
    Brains.append(Brain_Data)
    Brain_Types.append(EMCI)
    #lmci
    Brain_Data = []
    file = os.path.join("./LMCI",listofLMCI[i])
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            x = line.strip()
            Brain_Data.append(x)
    Brains.append(Brain_Data)
    Brain_Types.append(LMCI)
print("ahh damn it")   
X_train, X_test, y_train, y_test = train_test_split(Brains, Brain_Types, test_size=0.33)
print(len(y_train))
print(len(y_test))
model = MLPClassifier(solver='sgd', alpha = 0.0001, hidden_layer_sizes=(70,70,70), activation = 'tanh', learning_rate = 'invscaling')

trained_model = model.fit(X_train, y_train)
predict = trained_model.predict(X_test)
print(predict)
print(y_test)
print(len(predict))
print(len(y_test))
