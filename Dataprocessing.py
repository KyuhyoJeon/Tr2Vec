import pandas as pd
import numpy as np
import torch
import pickle

from torch.utils.data import Dataset, DataLoader


class Trajectory3Dnewset(Dataset):
    def __init__(self, path='Data/3D/new/csv/', size=None, task_type='C', slicing = False, stride = 10, flag='train', ratio = [7,1,2]):
        super().__init__()
        if size == None:
            if task_type == 'C':
                self.seq_len = size[-1] if slicing and flag!='train' else 2656
                self.pred_len = 5
            else:
                self.seq_len = 100
                self.pred_len = 100
        else:
            if task_type == 'C':
                self.seq_len = size[-1] if slicing and flag!='train' else size[0]
                self.pred_len = size[1]
            else:
                self.seq_len = size[0]
                self.pred_len = size[1]
        
        self.slicing = False if flag=='train' else slicing 
        self.stride = stride
        self.task_type = task_type
        self.flag = flag
        self.path = path
        self.ratio = ratio
        
        self.__read_data__()

    def __read_data__(self):
        '''
        X = ['Rmi_1', 'Rmi_2', 'Rmi_3', 
            'Vmi_1', 'Vmi_2', 'Vmi_3', 
            'Wmb_1', 'Wmb_2', 'Wmb_3', 
            'Accm_1', 'Accm_2', 'Accm_3', 
            'angEuler_1', 'angEuler_2', 'angEuler_3', 
            'FinOut_1', 'FinOut_2', 'FinOut_3', 'FinOut_4', 
            'FinCmd_1', 'FinCmd_2', 'FinCmd_3', 'FinCmd_4']
        Y = [Class]
        '''
        
        NSamples = 500 # if self.task_type == 'C' else 100
        
        d_type = {'train':0, 'valid':1, 'test':2}
        lefts = [0, 
                 int(10*(sum(self.ratio[:1])/sum(self.ratio))), 
                 int(10*(sum(self.ratio[:2])/sum(self.ratio)))]
        rights = [int(10*(sum(self.ratio[:1])/sum(self.ratio))), 
                  int(10*(sum(self.ratio[:2])/sum(self.ratio))), 
                  10]
        
        data_x_raw = []
        data_y_raw = []
        
        for i, tr_type in enumerate(['Normal', 'Burntime', 'Xcpposition', "ThrustTiltAngle", 'Finbias']):
            left = lefts[d_type[self.flag]]
            right = rights[d_type[self.flag]]
                
            for tr_i in range(NSamples):
                if i == 0: tr = pd.read_csv(self.path + f'Type_{i+1}_1.csv', header=None)
                else:
                    if self.slicing == False and self.task_type == 'C': 
                        if tr_i%10 < left or tr_i%10 >= right: continue
                    tr = pd.read_csv(self.path + f'Type_{i+1}_{tr_i+1}.csv', header=None)
                tr.columns = ['Rmi_1', 'Rmi_2', 'Rmi_3', 
                              'Vmi_1', 'Vmi_2', 'Vmi_3', 
                              'Wmb_1', 'Wmb_2', 'Wmb_3', 
                              'Accm_1', 'Accm_2', 'Accm_3', 
                              'angEuler_1', 'angEuler_2', 'angEuler_3', 
                              'FinOut_1', 'FinOut_2', 'FinOut_3', 'FinOut_4', 
                              'err_FinBias_1', 'err_FinBias_2', 'err_FinBias_3', 'err_FinBias_4', 
                              'FinCmd_1', 'FinCmd_2', 'FinCmd_3', 'FinCmd_4', 
                              'err_BurnTime', 'err_Tilt_1', 'err_Tilt_2', 'err_delXcp']
                tr['idx'] = tr.index
                tr = tr[['idx', 
                         'Rmi_1', 'Rmi_2', 'Rmi_3', 
                         'Vmi_1', 'Vmi_2', 'Vmi_3', 
                         'Wmb_1', 'Wmb_2', 'Wmb_3', 
                         'Accm_1', 'Accm_2', 'Accm_3', 
                         'angEuler_1', 'angEuler_2', 'angEuler_3', 
                         'FinOut_1', 'FinOut_2', 'FinOut_3', 'FinOut_4', 
                         'FinCmd_1', 'FinCmd_2', 'FinCmd_3', 'FinCmd_4']].to_numpy()
                tr[:, 1] = 4000 - tr[:, 1]
                if self.task_type == 'C':
                    if self.slicing == False: tr = np.concatenate((tr, np.tile(tr[-1], (self.seq_len-len(tr), 1))), axis=0)
                    if i == 0:
                        if self.slicing:
                            data_x_raw = [tr[l*self.stride:l*self.stride+self.seq_len] for l in range((len(tr)-self.seq_len+1)//self.stride) if left <= l%10 < right]
                            data_y_raw = [i for l in range((len(tr)-self.seq_len+1)//self.stride) if left <= l%10 < right]
                        else:
                            N = int(NSamples*(right-left)/10)
                            data_x_raw = [tr for _ in range(N)]
                            data_y_raw = [i for _ in range(N)]
                    else:
                        if self.slicing:
                            sub_x_raw = [tr[l*self.stride:l*self.stride+self.seq_len] for l in range((len(tr)-self.seq_len+1)//self.stride) if left <= l%10 < right]
                            sub_y_raw = [i for l in range((len(tr)-self.seq_len+1)//self.stride) if left <= l%10 < right]
                            data_x_raw += sub_x_raw
                            data_y_raw += sub_y_raw
                        else:
                            data_x_raw.append(tr)
                            data_y_raw.append(i)
                else:
                    for idx in range(len(tr)-self.seq_len-self.pred_len+1):
                        if idx%10 < left or idx%10 >= right: continue
                        start = idx
                        middle = start + self.seq_len
                        end = middle + self.pred_len
                        
                        data_x_raw.append(tr[start:middle])
                        data_y_raw.append(tr[middle:end])
                if i == 0: break
            # if self.flag == 'train': break
        
        self.data_x = np.array(data_x_raw)
        self.data_y = np.array(data_y_raw)

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]

        return x, y

    def __len__(self):
        return len(self.data_x)


def data_provider(args, flag):
    shuffle_flag = True
    drop_last = False
    batch_size = args.batch_size
    
    if args.model == 'C': size=[args.seq_len, args.class_num, args.slice_len]
    elif args.model == 'F': size=[args.seq_len, args.pred_len]
    
    path = args.path
    
    Data = Trajectory3Dnewset
    data_set = Data(
        path=path,
        size=size,
        task_type = args.model,
        slicing = args.slicing,
        stride = args.slice_stride,
        flag = flag,
        ratio= args.ratio)
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
        
    return data_set, data_loader