from __future__ import print_function
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from models import *
import time
import torch
import argparse
import torch.nn.functional as F
import logging
from tensorboardX import SummaryWriter
from utils import *
from torch.autograd import Variable
from torch.nn import functional as F
logger = logging.getLogger('logger')
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./cifar10-data', type=str)
    parser.add_argument('--epochs_reset', default=40, type=int)
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--epsilon', default=16, type=int)
    parser.add_argument('--alpha', default=16, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    parser.add_argument('--normal_mean', default=0, type=float, help='normal_mean')
    parser.add_argument('--normal_std', default=1, type=float, help='normal_std')
    parser.add_argument('--out-dir', default='AT_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--factor', default=0.6, type=float, help='Label Smoothing')
    parser.add_argument('--lamda', default=7, type=float, help='Label Smoothing')
    parser.add_argument('--gamma', default=0.03, type=float, help='Label Smoothing')
    parser.add_argument('--ratio', default=1, type=float, help='gamma_max/gamma_min')
    parser.add_argument('--momentum_decay', default=0.3, type=float, help='momentum_decay')
    # parser.add_argument('--w1', default=0., type=float, help='hyper-parameter w_1')
    parser.add_argument('--w2', default=1., type=float, help='hyper-parameter w_2')
    parser.add_argument('--initialize', action="store_true", help="select the classification loss for benign samples or random initialized samples.")
    parser.add_argument('--reg-single', action="store_true", help="reduce the single step modification.")
    parser.add_argument('--reg-multi', action="store_true", help="reduce the multi-step modification.")
    arguments = parser.parse_args()
    return arguments
args = get_args()

def _label_smoothing(label, factor, num_classes):
    one_hot = np.eye(num_classes)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(num_classes - 1))

    return result


def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss

output_path = os.path.join(args.out_dir, 'RS')

summary_log_dir=os.path.join(output_path,"Attack")
if not os.path.exists(summary_log_dir):
    os.makedirs(summary_log_dir)
if not os.path.exists(output_path):
    os.makedirs(output_path)
logfile = os.path.join(output_path, 'output.log')
if os.path.exists(logfile):
    os.remove(logfile)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(output_path, 'output.log'))
logger.info(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

epsilon = (args.epsilon / 255.) / std
alpha = (args.alpha / 255.) / std


if 'cifar100' in args.data_dir:
    data_shape = (50000,3,32,32)
    num_classes = 100
    train_loader, test_loader = get_all_loaders_cifar100(args.data_dir,args.batch_size)
elif 'cifar10' in args.data_dir:
    data_shape = (50000,3,32,32)
    num_classes = 10
    train_loader, test_loader = get_all_loaders(args.data_dir,args.batch_size)
elif 'tiny' in args.data_dir:
    data_shape = (100000,3,64,64)
    num_classes = 200
    train_loader, test_loader = get_all_loaders_tinyImageNet(args.data_dir,args.batch_size)
else:
    print('define the data_shape for this dataset')



print('==> Building model..')
logger.info('==> Building model..')
if args.model == "VGG":
    model = VGG('VGG19')
elif args.model == "ResNet18":
    model = ResNet18(num_classes)
elif args.model == "PreActResNest18":
    model = PreActResNet18(num_classes)
elif args.model == "WideResNet":
    model = WideResNet(num_classes = num_classes)
target_model = model.cuda()


opt=torch.optim.SGD(target_model.parameters(),lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
decay_arr = [0, 100, 105, 110]
lr_steps = decay_arr[3]* len(train_loader)
if args.lr_schedule == 'cyclic':
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                    step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
elif args.lr_schedule == 'multistep':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * decay_arr[1]/decay_arr[3], lr_steps * decay_arr[2] / decay_arr[3]],
                                                        gamma=0.1)



def train(args, model, train_loader, opt,scheduler, epoch, pre_loss, beta_loss):
    epoch_time = 0
    train_loss = 0
    train_loss_normal = 0
    train_acc = 0
    train_n = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start_time=time.time()
        rand_perturb = torch.FloatTensor(data.shape).uniform_(-args.epsilon / 255., args.epsilon / 255.)
        rand_perturb = rand_perturb.cuda()
        data, target = data.cuda(), target.cuda()
        data = data + rand_perturb
        data = clamp(data, lower_limit, upper_limit)
        label_smoothing = Variable(torch.tensor(_label_smoothing(target, args.factor, num_classes)).cuda())
        data.requires_grad_()
        with torch.enable_grad():
            ori_output = model(data)
            ori_loss = F.cross_entropy(ori_output, target)#  LabelSmoothLoss(ori_output, (label_smoothing).float())
        grad = torch.autograd.grad(ori_loss, [data])[0]
        perturbation = alpha* torch.sign(grad)
        # perturbation= clamp( alpha* torch.sign(grad), -epsilon, epsilon)
        data_adv = data + perturbation
        data_adv = clamp(data_adv, lower_limit, upper_limit)


        output=model(data_adv)
        if args.initialize:
            # without time consumption
            loss_normal = ori_loss
        else:
            benign_output = model(data)
            loss_normal =  LabelSmoothLoss(benign_output, (label_smoothing).float())
        loss_reg1 = 0
        loss_reg2 = 0
        if args.reg_single:
            loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
            loss_reg1 = loss_fn(output.float(), ori_output.float())
        if args.reg_multi:
            loss_reg2 = torch.mean(torch.sign(output.float() - ori_output.float()) * torch.sign(ori_output.float() - benign_output.float()))
        loss = LabelSmoothLoss(output, (label_smoothing).float())+args.lamda*(loss_reg1 + loss_reg2)

        gamma_flag = False
        for j, _ in enumerate(decay_arr):
            if j < len(decay_arr) - 1:
                start_decay = decay_arr[j]
                end_decay = decay_arr[j+1]
                if epoch > start_decay + args.stride and epoch < end_decay - args.stride:
                    gamma_flag = True
                    break

        if gamma_flag:
            gamma_max = torch.tensor(args.gamma).cuda()
            gamma_min = gamma_max / args.ratio
            if torch.abs(loss_normal - pre_loss) > min(max(gamma_min, beta_loss), gamma_max):
                if loss_normal - pre_loss > 0:
                    # below fitting
                    loss = loss + args.w2 * loss_normal
                else:
                    # over fitting
                    loss = loss - args.w2 * loss_normal



        opt.zero_grad()
        model.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        # faster with Tensor.detach() 
        train_loss += loss.item() * target.size(0)
        train_loss_normal += loss_normal.item() * target.size(0)
        train_acc += (output.max(1)[1] == target).sum().item()
        train_n += target.size(0)

        batch_end_time=time.time()
        epoch_time += batch_end_time-batch_start_time

    pre_loss2 = torch.tensor(train_loss_normal/train_n).cuda()
    beta_loss = pre_loss2 - pre_loss
    pre_loss = pre_loss2

    lr = scheduler.get_lr()[0]
    logger.info('Epoch \t Seconds \t LR \tTrain normal Loss  \t Train Loss \t Train Acc')
    logger.info('%d \t %.1f \t %.4f\t %.4f \t %.4f \t %.4f',
                epoch, epoch_time, lr, train_loss_normal/ train_n, train_loss / train_n, train_acc / train_n)
    return train_loss/ train_n, train_loss_normal/ train_n, pre_loss, beta_loss
def main():
    best_result = 0
    epoch_clean_list = []
    epoch_pgd_list = []
    epoch_loss_list = []
    epoch_loss_ori_list = []
    pre_loss = 0
    beta_loss = 0
    for epoch in range(args.epochs):
        train_loss, train_loss_nor, pre_loss, beta_loss = train(args, target_model,  train_loader, opt,scheduler, epoch, pre_loss, beta_loss)
        logger.info('==> Evaluating...')
        if args.model == "VGG":
            model_test = VGG('VGG19')
        elif args.model == "ResNet18":
            model_test = ResNet18(num_classes)
        elif args.model == "PreActResNest18":
            model_test = PreActResNet18(num_classes)
        elif args.model == "WideResNet":
            model_test = WideResNet(num_classes = num_classes)
        model_test.cuda()
        model_test.load_state_dict(target_model.state_dict())
        model_test.float()
        model_test.eval()

        adv_loss, adv_acc = evaluate_pgd(test_loader, model_test, 10, 1, epsilon)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(adv_acc)
        epoch_loss_list.append(train_loss)
        epoch_loss_ori_list.append(train_loss_nor)
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, adv_loss, adv_acc)
        if best_result<=adv_acc:
            best_result=adv_acc
            torch.save(target_model.state_dict(), os.path.join(output_path, 'best_model.pth'))
        torch.save(target_model.state_dict(), os.path.join(output_path, 'final_model.pth'))

    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)
    logger.info(epoch_loss_list)
    logger.info(epoch_loss_ori_list)



main()


