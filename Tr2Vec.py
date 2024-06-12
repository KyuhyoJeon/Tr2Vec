import time
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score

from Tools import hierarchical_contrastive_loss, take_per_row
from Networks import TrEncoder
from Evaluations import fit_svm


class Tr2Vec:
    def __init__(self, args, dataset=None):
        super().__init__()
        self.dataset = dataset
        self.args = args
        input_dims=1 if args.individual else dataset.data_x.shape[-1]-1

        self.model = TrEncoder(
            input_dims=input_dims, 
            output_dims=args.repr_dims, 
            hidden_dims=args.hidden_dims, 
            depth=args.layer_num, 
            channels=args.channel_num, 
            individual=args.individual
        ).to(args.device)
        self.net = torch.optim.swa_utils.AveragedModel(self.model)
        self.net.update_parameters(self.model)

        self.n_epochs = 0
        self.n_iters = 0

    def train(self, dataset=None, n_epochs=None, n_iters=None, verbose=True):
        assert self.dataset is not None or dataset is not None
        
        dataset = self.dataset if dataset is None else dataset
        
        if n_iters is None and n_epochs is None:
            n_iters = 600 if dataset.data_x.size <= 1000000 else 3000  # default param for n_iters, 600, 3000

        if self.args.debug:
            n_iters = 1

        train_dataset = TensorDataset(torch.from_numpy(dataset.data_x).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.args.batch_size, len(dataset.data_x)), shuffle=True, drop_last=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)

        loss_log = []

        while True:
            print("epoch:", self.n_epochs)
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            
            for batch in train_loader:
                if self.n_iters%100 == 0:
                    print("iter:", self.n_iters)
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                x = batch[0]
                if self.args.max_train_length is not None and x.size(1) > self.args.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.args.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.args.max_train_length]
                x = x.to(self.args.device)
                
                ts_l = int(min(x[:, -1, 0]))
                crop_l = np.random.randint(low=2 ** (self.args.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                optimizer.zero_grad()
                x = x[:, :, 1:]
                out1 = self.model(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                
                out2 = self.model(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]
                
                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.args.temporal_unit
                )
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self.model)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
            if interrupted:
                break

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            if self.args.debug:
                break

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def encode(self, data_x=None):    
        assert self.dataset is not None or data_x is not None
        
        data_x = self.dataset.data_x if data_x is None else data_x
        batch_size = 8
        device = 'cuda' if next(self.model.parameters()).is_cuda else 'cpu'
        
        org_training = self.net.training
        self.net.eval()

        dataset = TensorDataset(torch.from_numpy(data_x).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                out = self.net(x.to(device, non_blocking=True))
                out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = out.size(1),
                ).transpose(1, 2)
                out = out.cpu()
                out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        return output.numpy()

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        
    def eval(self, trainset, testset, visual=False):
        assert trainset.data_y.ndim == 1 or trainset.data_y.ndim == 2
        
        train_repr = self.encode(trainset.data_x)
        test_repr = self.encode(testset.data_x)
        
        clf = fit_svm(train_repr, trainset.data_y)
        acc = clf.score(test_repr, testset.data_y)
        y_score = clf.decision_function(test_repr)
        test_labels_onehot = label_binarize(testset.data_y, classes=np.arange(trainset.data_y.max()+1))
        auprc = average_precision_score(test_labels_onehot, y_score)
        out, eval_res = y_score, { 'acc': acc, 'auprc': auprc }
        
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        if visual:
            pred = clf.predict(test_repr)
            pca = PCA(n_components=3)
            printcipalComponents = pca.fit_transform(test_repr)

            plt.scatter(printcipalComponents[:, 0], printcipalComponents[:, 1], c=pred)
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(printcipalComponents[:, 0], printcipalComponents[:, 1], printcipalComponents[:, 2], c=pred, cmap='viridis', s=10)
            plt.show()
        
        return out, eval_res