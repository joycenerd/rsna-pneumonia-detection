import torch
import torch.nn as nn
import matplotlib.pyplot as plt

                                                                                   
                                                                                   
class NLL_OHEM(torch.nn.NLLLoss):                                                     
    """ Online hard example mining. 
    Needs input from nn.LogSotmax() """                                             
                                                                                   
    def __init__(self, ratio,device,total_ep):      
        super(NLL_OHEM, self).__init__(None, True)                                 
        self.ratio = ratio
        self.device=device
        self.total_ep=total_ep                                                         
                                                                                   
    def forward(self, x, y, epoch,sched_ratio=True):                                                                                              
        num_inst = x.size(0)
        if sched_ratio:
            self.step_ratio_sched(epoch)
        else:
            self.ratio=1
        # print(self.ratio)                                                      
        num_hns = int(self.ratio * num_inst)
        if num_hns>0:                                       
            x_ = x.clone()                                                       
            inst_losses = torch.autograd.Variable(torch.zeros(num_inst).to(self.device))       
            for idx, label in enumerate(y.data):                                       
                inst_losses[idx] = -x_.data[idx, label]                                                                                 
            _, idxs = inst_losses.topk(num_hns)                                        
            x_hn = x.index_select(0, idxs)                                             
            y_hn = y.index_select(0, idxs)
            loss=torch.nn.functional.nll_loss(x_hn, y_hn,reduction='mean')
        else:
            loss=torch.nn.functional.nll_loss(x,y,reduction='mean')                                        
        return loss  

    def cyclic_ratio_sched(self,epoch):
        half=int(self.total_ep/2)
        max_range=int(half*0.2)
        if epoch<half:
            if epoch<max_range:
                self.ratio=1.0
            else:
                self.ratio=(half-epoch)/float(half-max_range)
        
        else:
            if epoch<(half+max_range):
                self.ratio=0.5
            else: 
                self.ratio=0.5*(self.total_ep-epoch)/float(half-max_range)
        
    def step_ratio_sched(self,epoch):
        if epoch<40:
            self.ratio=1
        elif epoch>=40 and epoch<60:
            self.ratio=0.9
        elif epoch>=60 and epoch<90:
            self.ratio=0.8
        elif epoch>=90 and epoch<130:
            self.ratio=0.7
        elif epoch>=130 and epoch<170:
            self.ratio=0.6
        elif epoch>=170:
            self.ratio=0.5

if __name__=='__main__':
    ratio_list=[]
    epoch_list=[]
    for epoch in range(0,200):
        if epoch<40:
            ratio=1
        elif epoch>=40 and epoch<60:
            ratio=0.9
        elif epoch>=60 and epoch<90:
            ratio=0.8
        elif epoch>=90 and epoch<130:
            ratio=0.7
        elif epoch>=130 and epoch<170:
            ratio=0.6
        elif epoch>=170:
            ratio=0.5
        ratio_list.append(ratio)
        epoch_list.append(epoch)

    plt.rcParams["font.family"] = "serif"

    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
    
    plt.plot(epoch_list,ratio_list)
    plt.xlabel('epoch')
    plt.ylabel('hard mining ratio (k)')
    plt.savefig('ohem_step_ratio.png')
