import torch
import numpy as np

class EarlyStopping(object):
    def __init__(self, start_epoch, path, path_=None, patience=30):
        self.start_epoch = start_epoch
        self.patience = patience
        self.best_epoch = 0
        self.counter = 0
        self.min_loss = np.inf
        self.stopping = False
        self.path = path
        self.path_ = path_
        
    def __call__(self, epoch, loss, model, model_=None):
        if epoch > self.start_epoch:
            if self.min_loss > loss:
                self.min_loss = loss
                self.best_epoch = epoch
                self.counter = 0
                torch.save(model.state_dict(), self.path)
                if model_ is not None:
                    torch.save(model_.state_dict(), self.path_)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Best Epoch: ", self.best_epoch)
                    print("Early Stopping activated")
                    self.stopping = True
                
                