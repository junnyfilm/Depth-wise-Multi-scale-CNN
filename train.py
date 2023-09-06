from loss import CORAL,alpha_weight,TripletLoss
import torch
import torch.nn as nn
import numpy
from torch.autograd import Variable
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(extractor1,extractor2,extractor3,classifier, criterion, dataloader,tdataloader,triplet_loader, optimizer, epoch,step):
    
    # setup models
    extractor1.train()
    extractor2.train()
    extractor3.train()
    classifier.train()
    
    # steps
    start_steps = epoch * len(dataloader)
    total_steps = 10 * len(dataloader)

    for batch_idx, (data,tdata, tripletdata) in enumerate(zip(dataloader,tdataloader, triplet_loader)):        # prepare the data
        signal1,signal2,signal3,label = data
        signal1,signal2,signal3,label = Variable(signal1.to(device)), Variable(signal2.to(device)),Variable(signal3.to(device)),Variable(label.to(device).long())
        tsignal1,tsignal2,tsignal3,tlabel = tdata
        tsignal1,tsignal2,tsignal3,tlabel = Variable(tsignal1.to(device)), Variable(tsignal2.to(device)),Variable(tsignal3.to(device)),Variable(tlabel.to(device))
        anchor_pin_data, anchor_po_data, anchor_pdin_data, positive_pin_data, positive_po_data, positive_pdin_data, negative_pin_data, negative_po_data, negative_pdin_data,anchor_label=tripletdata
        anchor_pin_data, anchor_po_data, anchor_pdin_data, positive_pin_data, positive_po_data, positive_pdin_data, negative_pin_data, negative_po_data, negative_pdin_data,anchor_label = Variable(anchor_pin_data.to(device)), Variable(anchor_po_data.to(device)),Variable(anchor_pdin_data.to(device)),Variable(positive_pin_data.to(device)), Variable(positive_po_data.to(device)),Variable(positive_pdin_data.to(device)),Variable(negative_pin_data.to(device)), Variable(negative_po_data.to(device)),Variable(negative_pdin_data.to(device)),Variable(anchor_label.to(device).long())
        
        optimizer.zero_grad()
        feature11,feature21,feature31 = extractor1(signal1)
        feature12,feature22,feature32 = extractor2(signal2)
        feature13,feature23,feature33 = extractor3(signal3)
        feature=torch.cat((signal1,signal2,signal3,feature11,feature21,feature31,feature12,feature22,feature32,feature13,feature23,feature33),dim=1)
        # feature=torch.cat((feature1, feature2,feature3),dim=1)
        out,xview = classifier(feature)   
        
        tfeature11,tfeature21,tfeature31 = extractor1(tsignal1)
        tfeature12,tfeature22,tfeature32 = extractor2(tsignal2)
        tfeature13,tfeature23,tfeature33 = extractor3(tsignal3)
        tfeature=torch.cat((tsignal1,tsignal2,tsignal3,tfeature11,tfeature21,tfeature31,tfeature12,tfeature22,tfeature32,tfeature13,tfeature23,tfeature33),dim=1)
        tout,txview = classifier(tfeature)   
        
        
        afeature11,afeature21,afeature31 = extractor1(anchor_pin_data)
        afeature12,afeature22,afeature32 = extractor2(anchor_po_data)
        afeature13,afeature23,afeature33 = extractor3(anchor_pdin_data)
        anchorfeature=torch.cat((anchor_pin_data,anchor_po_data,anchor_pdin_data,afeature11,afeature21,afeature31,afeature12,afeature22,afeature32,afeature13,afeature23,afeature33),dim=1)
        anchorout,anchorxview = classifier(anchorfeature) 
        
        pafeature11,pafeature21,pafeature31 = extractor1(positive_pin_data)
        pafeature12,pafeature22,pafeature32 = extractor2(positive_po_data)
        pafeature13,pafeature23,pafeature33 = extractor3(positive_pdin_data)
        panchorfeature=torch.cat((positive_pin_data,positive_po_data,positive_pdin_data,pafeature11,pafeature21,pafeature31,pafeature12,pafeature22,pafeature32,pafeature13,pafeature23,pafeature33),dim=1)
        positiveout,positivexview = classifier(positivefeature) 
        
        nafeature11,nafeature21,nafeature31 = extractor1(negative_pin_data)
        nafeature12,nafeature22,nafeature32 = extractor2(negative_po_data)
        nafeature13,nafeature23,nafeature33 = extractor3(negative_pdin_data)
        nanchorfeature=torch.cat((negative_pin_data,negative_po_data,negative_pdin_data,nafeature11,nafeature21,nafeature31,nafeature12,nafeature22,nafeature32,nafeature13,nafeature23,nafeature33),dim=1)
        negatieout,negatiexview = classifier(negatiefeature) 
        
        triplet_loss=tripletloss(anchorxview,positivexview,negatiexview)
        coral_loss = CORAL(out,tout)
        loss = criterion(out, label)+triplet_loss+coral_loss
        
        loss.backward()
        optimizer.step()

        # print loss
        if (batch_idx + 1) % 100 == 0:
            print('[{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(batch_idx * len(signal1), len(dataloader.dataset),100. * batch_idx / len(dataloader), loss.item()))
            # total_loss.append(loss.item())
