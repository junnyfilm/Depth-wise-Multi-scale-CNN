from sklearn.model_selection import train_test_split
from custom_dataset import customdataset
from custom_dataset import TripletCustomdataset
from torch.utils.data import DataLoader

def loaders(data,label, data2,label2):
    train_set, valid_set, train_label, valid_label = train_test_split(data, label, train_size=0.8, random_state=1)
    traindataset = customdataset(train_set, train_label)
    validdataset = customdataset(valid_set, valid_label)
    traindataloader1 = DataLoader(traindataset, batch_size=32, shuffle=True, drop_last=True )
    validdataloader1 = DataLoader(validdataset, batch_size=32, shuffle=True, drop_last=True )
    testdataset = customdataset(data2, label2)
    testloader = DataLoader(testdataset, batch_size=1, shuffle=False, drop_last=False )
    return traindataloader1,validdataloader1,testloader
def tripletloader(data,label):
    tripletdataset=TripletCustomdataset(data,label)
    triplet__dataloader = DataLoader(tripletdataset, batch_size=32, shuffle=True, drop_last=True )
    return triplet__dataloader