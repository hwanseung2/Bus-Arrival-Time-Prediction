import os
import numpy as np
import argparse
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pandas.core.frame import DataFrame

from trainer import Trainer
from models.model import Informer
import nsml
from copy import deepcopy


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

    parser.add_argument('--mode', type=str, default='train', help='nsml submit일때 해당값이 test로 설정됩니다.')
    parser.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    parser.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    # parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

    parser.add_argument('--seq_len', type=int, default=36, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=16, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=16, help='prediction sequence length')

    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--distil', action='store_false', default=True, help='whether to use distilling in encoder, using this argument means not using distilling')
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--activation', type=str, default='gelu',help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--mix', action='store_false', default=True, help='use mix attention in generative decoder')
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=5, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--using_lradj', type=str2bool, default = True, help='True, False')
    parser.add_argument('--lr', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type3',help='adjust learning rate: type1, type2, type3')
    parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer: adamw, adafactor')
    parser.add_argument('--random_sampling', type=float, default=0, help='random sampling')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
    parser.add_argument('--pre_trained', type=str2bool, default = False, help='True, False')
    parser.add_argument('--pre_trained_dir', type = str)
    parser.add_argument('--cp', type = str)
    parser.add_argument('--using_aug', type = str2bool)
    parser.add_argument('--using_flag', type = str2bool, default = False)
    
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    return parser.parse_args()


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models

        for idx in range(len(self.models)):
            for param in self.models[idx].parameters():
                param.requires_grad = False
        
        self.model_list = torch.nn.ModuleList(self.models)


def testing(test_data, k, n):
    seq_len = k - 1
    label_len = k - 1
    pred_len = n - k + 1

    test_data = test_data[['route_id', 'station_id', 'direction', 'hour', 'dow',  'next_station_distance',\
                            'prev_duration']]
    
    data = test_data.values

    data[:, 4] = data[:, 4] / 7 - 0.5
    seq_x = torch.tensor(data[np.newaxis, :seq_len, 6:])
    seq_y = torch.tensor(data[np.newaxis, :label_len, 6:])
    seq_x_mark = torch.tensor(data[np.newaxis, :seq_len, :6])
    seq_y_mark = torch.tensor(data[np.newaxis, :, :6])
    seq_x_mark = seq_x_mark.float().cuda()
    seq_y_mark = seq_y_mark.float().cuda()

    seq_x = seq_x.float().cuda()
    seq_y = seq_y.float()


    dec_inp = torch.zeros([1, pred_len, seq_y.shape[-1]]).float()
    dec_inp = torch.cat([seq_y[:, :label_len, :], dec_inp], dim=1).float().cuda()

    return seq_x, seq_x_mark, dec_inp, seq_y_mark

def testing_ensemble(models, test_data, k, n, info, flag, flag_300, flag_1346, train_mean, train_std):
    predictions = []
    model_length = len(models.models)

    for model in models.models:
        model.eval()
    if flag == False:
        output_independent = np.zeros((model_length, (n-k+1)))
        seq_x, seq_x_mark, dec_inp, seq_y_mark = testing(test_data, k, n)
        index = 0
        with torch.no_grad():
            for model in models.models:
                model.pred_len = n - k + 1
                output = model(seq_x, seq_x_mark, dec_inp, seq_y_mark)
                output = output[0,:,0].detach().cpu().numpy()
                output = (output * train_std) + train_mean
                output_independent[index, :] = output
                index +=1
        
        output_independent = output_independent.mean(axis = 0)
        output = output_independent.tolist()

        cur_seq = k
        idx = 0
        while cur_seq <= n:
            out = output[idx]
            predictions.append([info['data_index'], info['route_id'], info['plate_no'], info['operation_id'], cur_seq, out])
            idx +=1
            cur_seq += 1
        return predictions

    else:
        return predictions


def bind_model(trainer: Trainer):
    def save(dirname, *args):
        state = {
            'model': trainer.model.state_dict(),
        }
        torch.save(state, os.path.join(dirname, 'model.pth'))
        with open(os.path.join(dirname, "preprocessor.pckl"), "wb") as f:
            pickle.dump(trainer.preprocessor, f)
        with open(os.path.join(dirname, 'train_mean.pckl'), "wb") as g:
            pickle.dump(trainer.train_mean, g)
        with open(os.path.join(dirname, 'train_std.pckl'), "wb") as h:
            pickle.dump(trainer.train_std, h)
        print('[INFO - NSML]: nsml saved')
        print('[INFO - NSML]: nsml saved')
        print('[INFO - NSML]: nsml saved')


    def load(dirname, *args):
        state = torch.load(os.path.join(dirname, 'model.pth'))
        trainer.model.load_state_dict(state['model'])
        # trainer.optimizer.load_state_dict(state['optimizer'])
        with open(os.path.join(dirname, "preprocessor.pckl"), "rb") as f:
            trainer.preprocessor = pickle.load(f)
        with open(os.path.join(dirname, 'train_mean.pckl'), "rb") as g:
            trainer.train_mean = pickle.load(g)
        with open(os.path.join(dirname, 'train_std.pckl'), "rb") as h:
            trainer.train_std = pickle.load(h)
        print(f"[INFO - NSML]: nsml loaded, dirname: {dirname}")

    def infer(test_data : DataFrame):
        test_data, k, n, info, flag, flag_300, flag_1346  = trainer.preprocessor.preprocess_test_data(test_data)
        return testing_ensemble(trainer.model, test_data, k, n, info, flag, flag_300, flag_1346, trainer.train_mean, trainer.train_std)


    nsml.bind(save = save, load = load, infer = infer)

def main():
    args = parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    model = Informer( enc_in = args.enc_in, dec_in = args.dec_in, c_out = args.c_out, seq_len = args.seq_len,
        label_len = args.label_len, out_len = args.pred_len, factor = args.factor, d_model = args.d_model, 
        n_heads = args.n_heads, e_layers = args.e_layers, d_layers = args.d_layers, d_ff = args.d_ff,
        dropout = args.dropout, attn = args.attn, activation = 'gelu', output_attention = args.output_attention,
        distil = args.distil, mix = args.mix, device = torch.device('cuda:0')
        )

    optimizer = optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.wd)
    criterion = nn.SmoothL1Loss()

    trainer = Trainer(args, model, optimizer, criterion)
    bind_model(trainer)
    print(trainer.train_mean, trainer.train_std)
    

    model_dict = {'788':'3', '789':'3', '790':'3', '791':'3'}

    trainers = []
    for key_ in model_dict.keys():
        nsml.load(checkpoint = model_dict[key_], session = f"KR96359/airush2022-2-6/{key_}")
        print(trainer.train_mean, trainer.train_std) # train_mean, train_std 성공적으로 불러와진다.
        trainers.append(deepcopy(trainer.model))
    model = Ensemble(trainers)
    trainer.model = model
    bind_model(trainer)

    if args.pause:
        nsml.paused(scope=locals())
    
    if args.mode == 'train':
        nsml.save('1')



    


if __name__ == '__main__':
    main()
