import argparse
from torchtext import data
import torch

from models import dataset
# from models import dataset2
from models import train1
from models import train2
from models.LSTM import LSTM
from models.AE_LSTM import AE_LSTM
from models.ATAE_LSTM import ATAE_LSTM
from models.IAN import IAN
from models.RAM import RAM
from models.TD_LSTM import TD_LSTM
from models.TC_LSTM import TC_LSTM


# 解析命令行
parser = argparse.ArgumentParser(description='Aspect level Sentiment Analysis')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=32, help='number of epochs for train [default: 6]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')

parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-hidden_size', type=int, default=64, help='hidden size of LSTM [default: 12]')
parser.add_argument('-num_layers', type=int, default=1, help='num of layers in LSTM')
parser.add_argument('-model', type=str, default='lstm', help='choose the model')
parser.add_argument('-static', action='store_true', default=False, help='keey the word embedding static')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-RAM_hidden_size', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-RAM_LSTM_num_layers', type=int, default=1, help='number of embedding dimension [default: 128]')
parser.add_argument('-GRU_hidden_size', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-GRU_timestep', type=int, default=3, help='number of embedding dimension [default: 128]')

# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


def semeval(text_field, label_field, **kargs):
    train_data1, test_data1, train_data2, test_data2, train_data3, test_data3 = dataset.SemEval2014.split(text_field, label_field)
    text_field.build_vocab(train_data1, test_data1,train_data2, test_data2, train_data3, test_data3)
    label_field.build_vocab(train_data1, test_data1, train_data2, test_data2, train_data3, test_data3)
    train_iter1, dev_iter1, train_iter2, dev_iter2, train_iter3, dev_iter3 = data.Iterator.splits(
                                (train_data1, test_data1, train_data2, test_data2, train_data3, test_data3),
                                batch_sizes=(args.batch_size, len(test_data1), args.batch_size, len(test_data2),
                                             args.batch_size, len(test_data3)),
                                **kargs)
    # for i in train_iter:
    #    print('text:',i.text.size())#,[[text_field.vocab.itos(i.text[x]) for x in i.text]])
    #    print("aspect:", i.aspect.size(), i.aspect.t())
    #    print('polar:', i.label.size())#,label_field.vocab.itos(i.babel))
    dev_iter1.sort_within_batch = False
    dev_iter1.sort = False
    dev_iter2.sort_within_batch = False
    dev_iter2.sort = False
    dev_iter3.sort_within_batch = False
    dev_iter3.sort = False
    # 预设是True, 使用for循环是会报错，显示元素之间不可比较，且train_iter, dev_iter在很多属性上的设置上有区别，原因未知

    return train_iter1, dev_iter1, train_iter2, dev_iter3, train_iter3, dev_iter3


text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
# semeval = SemEval2014(text_field, aspect_field, label_field)
train_iter1, test_iter1, train_iter2, test_iter2, train_iter3, test_iter3 = semeval(text_field, label_field)

# for i in train_iter:
#    print(i.text, i.aspect, i.label)
# print('test_iter')
# for j in test_iter:
#    print(i.text, i.aspect, i.label)

args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab)-1   # 4, positive, negative, neutral, conflict
args.text_field = text_field

args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

models = [TD_LSTM(args), TC_LSTM(args), LSTM(args), AE_LSTM(args), ATAE_LSTM(args), IAN(args), RAM(args)]
models_name = ["TD_LSTM", "TC_LSTM", "LSTM", "AE_LSTM", "ATAE_LSTM", "IAN_LSTM", "RAM"]

if args.cuda:
    torch.cuda.set_device(args.device)
    for i in range(len(models)):
        models[i] = models[i].cuda()

# train and test for all models
for i in range(2):
    train1.train(train_iter1, test_iter1, models[i], args, models_name[i])

for i in range(4)+2:
    train2.train(train_iter2, test_iter2, models[i], args, models_name[i])




