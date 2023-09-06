import torch
import torch.nn as nn
import numpy
from torch.autograd import Variable
device = "cuda" if torch.cuda.is_available() else "cpu"

def test(extractor1,extractor2,extractor3,classifier, dataloader):
    # setup the network
    predlist=[]
    real=[]    
    extractor1.eval()
    extractor2.eval()
    extractor3.eval()
    classifier.eval()
    correct = 0.0
    
    max_tr_score = 0
    max_val_score = 0
    for batch_idx, (data) in enumerate(dataloader):
    
        signal1,signal2,signal3,label = data
        signal1,signal2,signal3,label = Variable(signal1.to(device)), Variable(signal2.to(device)),Variable(signal3.to(device)),Variable(label.to(device).long())
        
        feature11,feature21,feature31 = extractor1(signal1)
        feature12,feature22,feature32 = extractor2(signal2)
        feature13,feature23,feature33 = extractor3(signal3)
        feature=torch.cat((signal1,signal2,signal3,feature11,feature21,feature31,feature12,feature22,feature32,feature13,feature23,feature33),dim=1)
        out,xview = classifier(feature)   

        pred = out.data.max(1, keepdim= True)[1]
        predlist.append(pred.cpu().detach().numpy().squeeze())
        real.append(label)        
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

        # predlist.append(pred)
    print('\nAccuracy: {}/{} ({:.4f}%)\n'.format(
        correct, len(dataloader.dataset), 100. * float(correct) / len(dataloader.dataset)))
    acc=100. * float(correct) / len(dataloader.dataset)
    return acc,predlist,real