import torch
import random
from numpy import array

device = "cuda" if torch.cuda.is_available() else "cpu"


class customdataset(Dataset):
    def __init__(self, data, label): 
        super().__init__()
        self.pin_data=data[:,0,:].unsqueeze(1)   
        self.po_data=data[:,1,:].unsqueeze(1)
        self.pdin_data=data[:,2,:].unsqueeze(1)
        self.label=label
        
    def __len__(self):
        return len(self.label)
  
    def __getitem__(self, idx):
        pin_data = self.pin_data[idx]
        po_data = self.po_data[idx]
        pdin_data = self.pdin_data[idx]
        label = self.label[idx] 
                
        return  pin_data.to(device).float(),po_data.to(device).float(),pdin_data.to(device).float(), label.to(device)


class customdataset_ts(Dataset):
    def __init__(self, data): 
        super().__init__()
        self.pin_data=data[:,0,:].unsqueeze(1)   
        self.po_data=data[:,1,:].unsqueeze(1)
        self.pdin_data=data[:,2,:].unsqueeze(1)
        
    def __len__(self):
        return len(self.pin_data)
  
    def __getitem__(self, idx):
        pin_data = self.pin_data[idx]
        po_data = self.po_data[idx]
        pdin_data = self.pdin_data[idx]
                
        return  pin_data.to(device).float(),po_data.to(device).float(),pdin_data.to(device).float()

class TripletCustomdataset(Dataset):
    def __init__(self, data, label):
        # self.data=data   
        self.pin_data=data[:,0,:].unsqueeze(1)   
        self.po_data=data[:,1,:].unsqueeze(1)
        self.pdin_data=data[:,2,:].unsqueeze(1)
        
        self.label=label
        self.index = array(range(len(label)))
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, item):      
        
        anchor_pin_data = self.pin_data[item]
        anchor_po_data = self.po_data[item]
        anchor_pdin_data = self.pdin_data[item]
        
        anchor_label = self.label[item]

        positive_list = self.index[self.index!=item][self.label[self.index!=item]==anchor_label]

        positive_item = random.choice(positive_list)
        
        positive_pin_data = self.pin_data[positive_item]
        positive_po_data = self.po_data[positive_item]
        positive_pdin_data = self.pdin_data[positive_item]
        
        
        negative_list = self.index[self.index!=item][self.label[self.index!=item]!=anchor_label]
        negative_item = random.choice(negative_list)
        negative_pin_data = self.pin_data[negative_item]
        negative_po_data = self.po_data[negative_item]
        negative_pdin_data = self.pdin_data[negative_item]

        return anchor_pin_data.to(device).float(),anchor_po_data.to(device).float(),anchor_pdin_data.to(device).float(), positive_pin_data.to(device).float(),positive_po_data.to(device).float(),positive_pdin_data.to(device).float(), negative_pin_data.to(device).float(),negative_po_data.to(device).float(),negative_pdin_data.to(device).float(), anchor_label.to(device)
