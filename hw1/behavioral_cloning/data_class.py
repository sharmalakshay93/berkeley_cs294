import numpy  as np
import pickle as pkl
from sklearn.model_selection import train_test_split
import torch.utils.data

class Data:
    def __init__(self, data_file, train_ratio=0.8, batch_size=100, only_final_results=False):
        
        assert (train_ratio >= 0 and train_ratio <= 1), "train_ratio must be in the range (0,1)"
        
        self.train_ratio = train_ratio
        self.val_ratio = 1 - train_ratio
        self.batch_size = batch_size
        data = pkl.load(open(data_file, "rb"))
        
        if not only_final_results:
            print('Train/val split: {}/{:.3f}'.format(self.train_ratio, self.val_ratio))
        self.X_train, self.X_val, self.y_train, self.y_val = self.split_data(data, only_final_results)
        
        self.input_dim = self.X_train.shape[1]
        self.output_dim = self.y_train.shape[1]
        
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.X_train).float(), torch.from_numpy(self.y_train).float())
        val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.X_val).float(), torch.from_numpy(self.y_val).float())
        
        self.train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)
        self.val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.X_val.shape[0])
        self.expert_stats = data['returns']

    def split_data(self, data, only_final_results):
        '''Split dataset into training and validation sets'''
        
        obs = data['observations']
        actions = np.squeeze(data['actions'])
        
        X_train, X_val, y_train, y_val = \
         train_test_split(obs, actions, test_size=self.val_ratio, random_state=42)
        if not only_final_results:
            print('splitting and shuffling data:')
            print('X_train.shape', X_train.shape)
            print('X_val.shape', X_val.shape)
            print('y_train.shape', y_train.shape)
            print('y_val.shape', y_val.shape)
        return X_train, X_val, y_train, y_val
    
    def get_train_val(self):
        return self.train_data_loader, self.val_data_loader