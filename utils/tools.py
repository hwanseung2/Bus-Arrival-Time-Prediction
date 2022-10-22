import numpy as np
import torch

import nsml
def adjust_learning_rate(optimizer, epoch, t, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lr_adjust = {epoch: args.lr * (0.9 ** ((epoch-1) // 1))}
    if args.lradj=='type1':
        lr_adjust = {epoch: args.lr * (0.9 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            1: 2e-7, 2: 5e-7, 3: 7e-7, 4: 1e-6, 8: 8e-7, 9: 6e-7
        }
    elif args.lradj == 'type3':
        if t <= 10000:
            lr = t / 10000 * args.lr
        else:
            lr = args.lr
    elif args.lradj == 'type4':
        if t <= 10000:
            lr = t / 10000 * args.lr
        elif t > 10000:
            lr = lr_adjust[epoch]
    if (args.lradj == 'type1' or args.lradj == 'type2') and epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    elif args.lradj == 'type3':
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if t <= 10000:
            print('Updating learning rate to {}'.format(lr))
    elif args.lradj == 'type4':
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        # nsml.save('eureka')
        self.val_loss_min = val_loss
