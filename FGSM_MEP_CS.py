import argparse
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import numpy as np
import torch
from models import *
# from tiny_models import *
from utils import *
import random
from torch.nn import functional as F
from torch.autograd import Variable

logger = logging.getLogger(__name__)


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
    parser.add_argument('--lamda', default=10, type=float, help='Label Smoothing')
    parser.add_argument('--gamma', default=0.03, type=float, help='Label Smoothing')
    parser.add_argument('--ratio', default=1, type=float, help='gamma_max/gamma_min')
    parser.add_argument('--stride', default=1, type=float, help='convergence stride')
    parser.add_argument('--momentum_decay', default=0.3, type=float, help='momentum_decay')
    # parser.add_argument('--w1', default=0., type=float, help='hyper-parameter w_1')
    parser.add_argument('--w2', default=1., type=float, help='hyper-parameter w_2')
    parser.add_argument('--initialize', action="store_true", help="select the classification loss for benign samples or random initialized samples.")
    parser.add_argument('--reg-single', action="store_true", help="reduce the single step modification.")
    parser.add_argument('--reg-multi', action="store_true", help="reduce the multi-step modification.")
    return parser.parse_args()
def _label_smoothing(label, factor, num_classes):
    one_hot = np.eye(num_classes)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(num_classes - 1))

    return result


def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss
upper_limit_y = 1
lower_limit_y = 0
def main():
    args = get_args()
    output_path = os.path.join(args.out_dir, 'CS-MEP')


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
    model=torch.nn.DataParallel(model)
    model = model.cuda()
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)



    num_of_example = data_shape[0]
    batch_size = args.batch_size

    iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)



    decay_arr = [0, 100, 105, 110]
    lr_steps = decay_arr[3]* iter_num
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * decay_arr[1]/decay_arr[3], lr_steps * decay_arr[2] / decay_arr[3]],
                                                         gamma=0.1)

    logger.info('==> Training..')
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    best_result = 0
    epoch_clean_list = []
    epoch_pgd_list = []
    for i, (X, y) in enumerate(train_loader):
        cifar_x, cifar_y = X.cuda(), y.cuda()
    def atta_aug(input_tensor, rst):
        batch_size = input_tensor.shape[0]
        x = torch.zeros(batch_size)
        y = torch.zeros(batch_size)
        flip = [False] * batch_size

        for i in range(batch_size):
            flip_t = bool(random.getrandbits(1))
            x_t = random.randint(0, 8)
            y_t = random.randint(0, 8)

            rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
            if flip_t:
                rst[i] = torch.flip(rst[i], [2])
            flip[i] = flip_t
            x[i] = x_t
            y[i] = y_t

        return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}


    pre_loss = 0
    beta_loss = 0
    for epoch in range(decay_arr[3]):

        batch_size = args.batch_size
        cur_order = np.random.permutation(num_of_example)
        iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)
        batch_idx = -batch_size
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        train_normal_loss = 0
        if epoch %args.epochs_reset== 0:
            temp=torch.rand(data_shape)
            if args.delta_init != 'previous':
                all_delta = torch.zeros_like(temp).cuda()
                all_momentum=torch.zeros_like(temp).cuda()
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                    all_delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                all_delta.data = clamp(alpha * torch.sign(all_delta), -epsilon, epsilon)

        idx = torch.randperm(cifar_x.shape[0])
        cifar_x =cifar_x[idx, :,:,:].view(cifar_x.size())
        cifar_y = cifar_y[idx].view(cifar_y.size())
        all_delta=all_delta[idx, :, :, :].view(all_delta.size())
        all_momentum=all_momentum[idx, :, :, :].view(all_delta.size())

        for i in range(iter_num):

            batch_idx = (batch_idx + batch_size) if batch_idx + batch_size < num_of_example else 0
            X=cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            y= cifar_y[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            delta =all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            next_delta = all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            momentum=all_momentum[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()

            X=X.cuda()
            y=y.cuda()
            batch_size = X.shape[0]
            rst = torch.zeros(batch_size, data_shape[1], data_shape[2], data_shape[3]).cuda()
            X, transform_info = atta_aug(X, rst)
            label_smoothing = Variable(torch.tensor(_label_smoothing(y, args.factor, num_classes)).cuda()).float()
            delta.requires_grad = True
            ori_output = model(X + delta[:X.size(0)])
            ori_loss =  LabelSmoothLoss(ori_output, label_smoothing.float())# F.cross_entropy(ori_output, y)
            decay=args.momentum_decay
            ori_loss.backward(retain_graph=True)
            x_grad = delta.grad.detach()
            grad_norm = torch.norm(x_grad, p=1)
            momentum = x_grad/grad_norm+momentum * decay

            next_delta.data = clamp(delta + alpha * torch.sign(momentum), -epsilon, epsilon)
            next_delta.data[:X.size(0)] = clamp(next_delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta_adv = delta.detach()

            output = model(X + delta_adv[:X.size(0)])

            if args.initialize:
                # without time consumption
                loss_normal = ori_loss
            else:
                benign_output = model(X)
                loss_normal =    LabelSmoothLoss(benign_output, (label_smoothing).float())
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
            loss.backward()
            opt.step()
            # faster with Tensor.detach() 
            train_loss += loss.item() * y.size(0)
            train_normal_loss += loss_normal.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()

            all_momentum[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]] = momentum
            all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]=next_delta
        pre_loss2 = torch.tensor(train_normal_loss/train_n).cuda()
        beta_loss = pre_loss2 - pre_loss
        pre_loss = pre_loss2
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f\t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_normal_loss / train_n,train_acc / train_n)

        logger.info('==> Evaluating..')
        if args.model == "VGG":
            model_test = VGG('VGG19')
        elif args.model == "ResNet18":
            model_test = ResNet18(num_classes)
        elif args.model == "PreActResNest18":
            model_test = PreActResNet18(num_classes)
        elif args.model == "WideResNet":
            model_test = WideResNet(num_classes = num_classes)
        model_test.cuda()
        model_test = torch.nn.DataParallel(model_test)
        model_test.load_state_dict(model.state_dict())
        model_test.float()
        model_test.eval()

        adv_loss, adv_acc = evaluate_pgd(test_loader, model_test, 10, 1,epsilon)
        # adv_loss, adv_acc = evaluate_pgd_cw(test_loader, model_test, 20, 1, args.epsilon)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(adv_acc)
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, adv_loss, adv_acc)
        if best_result <= adv_acc:
            best_result = adv_acc
            torch.save(model.state_dict(), os.path.join(output_path, 'best_model.pth'))

    torch.save(model.state_dict(), os.path.join(output_path, 'final_model.pth'))
    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)

if __name__ == "__main__":
    main()
