"""
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""


import os
import time
import argparse
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import onnxruntime as ort

import amct_onnx
from amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT
from amct_pytorch.nn.module.quantization.conv_transpose_2d import ConvTranspose2dQAT
from amct_pytorch.nn.module.quantization.conv3d import Conv3dQAT
from amct_pytorch.nn.module.quantization.linear import LinearQAT
from resnet import resnet101


PATH = os.path.realpath('./')
OUTPUTS = os.path.join(PATH, 'outputs/retrain')
TMP = os.path.join(OUTPUTS, 'tmp')

SIZE = 224
NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--config_file', dest='config_file', default=None, type=str, help='Config file for qat op.')
parser.add_argument('--batch_num', dest='batch_num', default=2, type=int, help='number of total batch to run')
parser.add_argument(
    '--train_set', dest='train_set', default=None, type=str,
    help='The path of ILSVRC-2012-CLS image classification dataset for training.')
parser.add_argument(
    '--eval_set', dest='eval_set', default=None, type=str,
    help='The path of ILSVRC-2012-CLS image classification dataset for evaluation.')
parser.add_argument(
    '--num_parallel_reads', dest='num_parallel_reads', default=4, type=int,
    help='The number of files to read in parallel.')
parser.add_argument('--batch_size', dest='batch_size',
                    default=16, type=int, help='batch size (default: 16)')
parser.add_argument('--learning_rate', dest='learning_rate',
                    default=1e-5, type=float, help='initial learning rate')
parser.add_argument(
    '--train_iter', dest='train_iter', default=2000, type=int, help='number of total iterations to run')
parser.add_argument('--print_freq', dest='print_freq',
                    default=10, type=int, help='print frequency (default: 10)')
parser.add_argument(
    '--dist_url', dest='dist_url', default='tcp://127.0.0.1:50011', type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--distributed', dest='distributed', action='store_true', help='Use multi-processing distributed training')


def args_check(args):
    """Verify the validity of input parameters"""
    if args.train_set is None:
        raise RuntimeError('Must specify a training dataset path!')
    args.train_set = os.path.realpath(args.train_set)
    if not os.access(args.train_set, os.F_OK):
        raise RuntimeError('Must specify a valid training dataset path!')

    if args.eval_set is None:
        raise RuntimeError('Must specify a evaluation dataset path!')
    args.eval_set = os.path.realpath(args.eval_set)
    if not os.access(args.eval_set, os.F_OK):
        raise RuntimeError('Must specify a valid evaluation dataset path!')


class AverageCounter(object):
    """Compute and store the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count_num = 0
        self.reset_param()

    def __str__(self):
        fmtstr = ''.join(
            ['{name} {value', self.fmt, '} ({average', self.fmt, '})'])
        return fmtstr.format(**self.__dict__)

    def reset_param(self):
        """reset_param"""
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count_num = 0

    def update_param(self, value, size=1):
        """update_param"""
        self.value = value
        self.sum += value * size
        self.count_num += size
        self.average = self.sum / self.count_num


class ProgressCounter(object):
    """Manage and show the information of training and validation"""

    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        """display"""
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = ''.join(['{:', str(num_digits), 'd}'])
        return ''.join(['[', fmt, '/', fmt.format(num_batches), ']'])


def get_input_data(shape_list, model):
    """Get input data to generate onnx graph for amct_pytorch tools"""
    device = next(model.parameters()).device
    input_data = tuple([torch.randn(shape).to(device) for shape in shape_list])
    return input_data


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Set the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def accuracy(output, target, top_k=(1,)):
    """Compute the accuracy over the k top predictions"""
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, prediction = output.topk(max_k, 1, True, True)
        prediction = prediction.t()
        correct = prediction.eq(target.view(1, -1).expand_as(prediction))

        result = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result


def train(train_loader, model, optimizer, train_params, gpu_index):
    """Train the model"""
    iteration = train_params.get('iteration')
    print_freq = train_params.get('print_freq')
    batch_time = AverageCounter('Time', ':6.3f')
    data_time = AverageCounter('Data', ':6.3f')
    losses = AverageCounter('Loss', ':.4e')
    top1 = AverageCounter('Acc@1', ':6.2f')
    top5 = AverageCounter('Acc@5', ':6.2f')
    progress = ProgressCounter(
        iteration, [batch_time, data_time, losses, top1, top5], prefix='Train: ')

    # switch to train mode.
    model.train()

    criterion = nn.CrossEntropyLoss()
    if gpu_index >= 0:
        criterion = nn.CrossEntropyLoss().cuda(gpu_index)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time.
        data_time.update_param(time.time() - end)

        if gpu_index >= 0:
            images = images.cuda(gpu_index, non_blocking=True)
            target = target.cuda(gpu_index, non_blocking=True)

        # compute output.
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss.
        acc1, acc5 = accuracy(output, target, top_k=(1, 5))
        losses.update_param(loss.item(), images.size(0))
        top1.update_param(acc1[0], images.size(0))
        top5.update_param(acc5[0], images.size(0))

        # compute gradient and do SGD step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time.
        batch_time.update_param(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            progress.display(i + 1)
        if (i + 1) >= iteration:
            break


def validate(val_loader, model, validate_params, gpu_index):
    """Validate the model"""
    iteration = validate_params.get('iteration')
    print_freq = validate_params.get('print_freq')
    search_n = validate_params.get('search_n', False)
    batch_time = AverageCounter('Time', ':6.3f')
    losses = AverageCounter('Loss', ':.4e')
    top1 = AverageCounter('Acc@1', ':6.2f')
    top5 = AverageCounter('Acc@5', ':6.2f')
    progress = ProgressCounter(
        iteration, [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode.
    model.eval()

    criterion = nn.CrossEntropyLoss()
    if gpu_index >= 0:
        criterion = nn.CrossEntropyLoss().cuda(gpu_index)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpu_index >= 0:
                images = images.cuda(gpu_index, non_blocking=True)
                target = target.cuda(gpu_index, non_blocking=True)

            # compute output.
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss.
            acc1, acc5 = accuracy(output, target, top_k=(1, 5))
            losses.update_param(loss.item(), images.size(0))
            top1.update_param(acc1[0], images.size(0))
            top5.update_param(acc5[0], images.size(0))

            # measure elapsed time.
            batch_time.update_param(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0 and not search_n:
                progress.display(i + 1)
            if (i + 1) >= iteration and search_n:
                break

    return top1.average, top5.average


def validate_onnx(val_loader, model, print_freq):
    """Validate the onnx model"""
    batch_time = AverageCounter('Time', ':6.3f')
    losses = AverageCounter('Loss', ':.4e')
    top1 = AverageCounter('Acc@1', ':6.2f')
    top5 = AverageCounter('Acc@5', ':6.2f')
    progress = ProgressCounter(
        len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

    # switch to evaluate mode.
    ort_session = ort.InferenceSession(model, amct_onnx.AMCT_SO)

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute output.
            output = ort_session.run(None, {'input': images.numpy()})
            output = torch.from_numpy(output[0])
            loss = criterion(output, target)

            # measure accuracy and record loss.
            acc1, acc5 = accuracy(output, target, top_k=(1, 5))
            losses.update_param(loss.item(), images.size(0))
            top1.update_param(acc1[0], images.size(0))
            top5.update_param(acc5[0], images.size(0))

            # measure elapsed time.
            batch_time.update_param(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                progress.display(i + 1)

    return top1.average, top5.average


def create_data_loader(train_set_dir, test_set_dir, args):
    """Generate dataset loader."""
    traindir = os.path.realpath(train_set_dir)
    train_dataset = datasets.ImageFolder(
        traindir, transforms.Compose(
            [transforms.RandomResizedCrop(SIZE), transforms.RandomHorizontalFlip(), transforms.ToTensor(), NORM]))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.num_parallel_reads, pin_memory=True, sampler=train_sampler)

    valdir = os.path.realpath(test_set_dir)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(SIZE), transforms.ToTensor(), NORM])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_parallel_reads, pin_memory=True)

    return train_loader, train_sampler, val_loader


def cal_original_model_accuracy(model, gpu_index, val_loader, args):
    """Infer the accuracy of the original model."""
    if gpu_index >= 0:
        torch.cuda.set_device(gpu_index)
        model = model.cuda(gpu_index)

    print("=> Validate pre-trained model 'resnet101'")
    validate_params = {
            'iteration': len(val_loader),
            'print_freq': args.print_freq,
            'search_n': False
        }
    ori_top1, ori_top5 = validate(val_loader, model, validate_params, gpu_index)
    print('The origin model top 1 accuracy = {:.2f}%.'.format(ori_top1))
    print('The origin model top 5 accuracy = {:.2f}%.'.format(ori_top5))

    return ori_top1, ori_top5


def train_and_val(model, gpu_index, data_loader, train_sampler, args):
    """train_and_val"""
    train_loader = data_loader[0]
    val_loader = data_loader[1]
    # Allocating a model to a specified device.
    if gpu_index >= 0:
        if args.distributed:
            torch.cuda.set_device(gpu_index)
            model.cuda(gpu_index)
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[gpu_index], find_unused_parameters=True)
        else:
            torch.cuda.set_device(gpu_index)
            model = model.cuda(gpu_index)

    # Define optimizer.
    optimizer = torch.optim.SGD(
        model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)

    # Retrain the model.
    for epoch in range(0, 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.learning_rate)

        # train for train_iter.
        print("=> training quantized model")
        train_params = {
            'iteration': args.train_iter,
            'print_freq': args.print_freq
        }
        train(train_loader, model, optimizer,
              train_params, gpu_index)

        # evaluate on validation set.
        validate_params = {
            'iteration': args.batch_num,
            'print_freq': args.print_freq,
            'search_n': True
        }
        validate(val_loader, model, validate_params, gpu_index)


def cal_quant_model_accuracy(model, val_loader, args):
    """Save the quantized model and infer the accuracy of the quantized model."""
    torch.save({'state_dict': model.state_dict()},
               os.path.join(TMP, 'model_best.pth.tar'))
    print('==> AMCT step3: save onnx model..')
    quantized_pb_path = os.path.join(OUTPUTS, 'ResNet101')
    inter_model_path = os.path.join(TMP, 'ResNet101_inter_model.onnx')
    torch.onnx.export(model,
                      get_input_data([(1, 3, SIZE, SIZE)], model),
                      inter_model_path,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}},
                      opset_version=13)
    amct_onnx.convert_model.convert_qat_model(inter_model_path,
                                              quantized_pb_path)
    print("=> validating fake quant model")
    fake_quant_model_path = os.path.join(OUTPUTS, 'ResNet101_fake_quant_model.onnx')
    quant_top1, quant_top5 = validate_onnx(val_loader,
                                           fake_quant_model_path,
                                           args.print_freq)
    return quant_top1, quant_top5


def from_float(model, config):
    """replace torch original operators with customized qat opeartors."""
    def _set_module(model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)
    for name, module in model.named_modules():
        if name in config.get('skip_layers'):
            continue
        if isinstance(module, torch.nn.Conv2d):
            qat_module = Conv2dQAT.from_float(
                module, config=config.get('Conv2d'))
            _set_module(model, name, qat_module)
        elif isinstance(module, torch.nn.ConvTranspose2d):
            qat_module = ConvTranspose2dQAT.from_float(
                module, config=config.get('ConvTranspose2d'))
            _set_module(model, name, qat_module)
        elif isinstance(module, torch.nn.Conv3d):
            qat_module = Conv3dQAT.from_float(
                module, config=config.get('Conv3d'))
            _set_module(model, name, qat_module)
        elif isinstance(module, torch.nn.Linear):
            qat_module = LinearQAT.from_float(
                module, config=config.get('Linear'))
            _set_module(model, name, qat_module)
    return model


def main():
    """retrain"""
    args = parser.parse_args()
    args_check(args)

    cudnn.benchmark = True
    if torch.cuda.is_available():
        gpu_num = torch.cuda.device_count()
    else:
        gpu_num = 0

    if args.distributed:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function.
        if gpu_num == 0:
            raise RuntimeError(
                'Must has at least one available GPU in distributed training mode!')
        print('Using multi GPUs: DistributedDataParallel mode.')
        mp.spawn(main_worker, nprocs=gpu_num, args=(gpu_num, args))
    else:
        # Simply call main_worker function.
        if gpu_num > 0:
            gpu_index = 0
            print('Using single GPU.')
        else:
            gpu_index = -1
            print('Using CPU, this will be slow')
        main_worker(gpu_index, gpu_num, args)


def main_worker(gpu_index, gpu_num, args):
    """main_worker"""
    # Phase initialization.
    # If multi-card distributed training is used, initialize the training
    # process.
    if not os.path.exists(TMP):
        os.makedirs(TMP)
    if args.distributed:
        dist.init_process_group(
            backend='nccl', init_method=args.dist_url, world_size=gpu_num, rank=gpu_index)

    # Generate training dataset and validation dataset loader.
    train_loader, train_sampler, val_loader = create_data_loader(
        args.train_set, args.eval_set, args)

    # Phase origin model accuracy.
    # Step 1: Create model.
    print("=> Create pre-trained model 'resnet101'")
    # Choose whether to use the model downloaded online or
    # the model modified locally.
    # 1. use the resnet101 model downloaded online.
    model = models.__dict__['resnet101'](pretrained=True)

    # 2. use the resnet101 model modified locally.
    model = resnet101(pretrained=True)

    # Step 2: Calculate origin model's accuracy.
    ori_top1, ori_top5 = cal_original_model_accuracy(
        model, gpu_index, val_loader, args)
    # Phase retrain the model.
    # Step 1: Convert original torch opeartor to customized qat operator.
    print('==> AMCT step1: convert original torch op to qat op..')
    with open(args.config_file, 'r') as f:
        config_info = json.load(f)
    from_float(model, config_info)

    # Step 2: Retraining quantitative model and inferencing.
    data_loader = (train_loader, val_loader)
    train_and_val(model, gpu_index, data_loader,
                  train_sampler, args)

    # Step 4: Save the quantized model and infer the accuracy of the
    # quantized model.
    if not args.distributed or (args.distributed and gpu_index == 0):
        quant_top1, quant_top5 = cal_quant_model_accuracy(model, val_loader, args)

        print(
            '[INFO] ResNet-101 before retrain top1:{:.2f}% top5:{:.2f}%'.format(ori_top1, ori_top5))
        print(
            '[INFO] ResNet-101 after retrain top1:{:.2f}% top5:{:.2f}%'.format(quant_top1, quant_top5))


if __name__ == '__main__':
    main()
