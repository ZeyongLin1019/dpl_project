import os
import shutil
# os.environ['CUDA_VISIBLE_DEVICES']='3'
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models,transforms
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss,resize,spectral_reconstruction_loss
from utils.evaluation import Evaluator
from utils.HSICommonUtils import normlize3D, ImageStretching

# import matplotlib.pyplot as plt
# from visual.visualize_map import DrawResult
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict
from PIL import Image
from model.MambaHSI_Plus import MambaHSI_Plus

from calflops import calculate_flops
from datetime import datetime
torch.autograd.set_detect_anomaly(False)

time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CUBE_NPY = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data', 'label', 'label1_cube.npy'))
DEFAULT_GT_NPY = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data', 'label', 'label1_gt.npy'))


def vis_a_image(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=False):
    visualize_predict(gt_vis,pred_vis,save_single_predict_path,save_single_gt_path,only_vis_label=only_vis_label)
    visualize_predict(gt_vis,pred_vis,save_single_predict_path.replace('.png','_mask.png'),save_single_gt_path,only_vis_label=True)


# random seed setting
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_chunk_ranges(height, num_chunks):
    chunk_ranges = []
    for i in range(num_chunks):
        start = (i * height) // num_chunks
        end = ((i + 1) * height) // num_chunks
        if end > start:
            chunk_ranges.append((start, end))
    return chunk_ranges


def build_class_weights(class_counts, mode='inverse', beta=0.999, clip_min=0.2, clip_max=5.0):
    if mode == 'none':
        return None

    counts = class_counts.float().clamp_min(1.0)
    total = counts.sum().clamp_min(1.0)

    if mode == 'effective':
        beta = float(min(max(beta, 0.0), 0.999999))
        effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=counts.dtype), counts)
        effective_num = effective_num.clamp_min(1e-8)
        class_weights = (1.0 - beta) / effective_num
    else:
        class_weights = total / counts

    class_weights = class_weights / class_weights.mean().clamp_min(1e-6)
    class_weights = torch.clamp(class_weights, min=float(clip_min), max=float(clip_max))
    return class_weights


class SegLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=-1, use_focal=False, focal_gamma=1.5):
        super().__init__()
        self.ignore_index = ignore_index
        self.use_focal = bool(use_focal)
        self.focal_gamma = float(max(0.0, focal_gamma))
        if class_weights is None:
            self.register_buffer('class_weights', None)
        else:
            self.register_buffer('class_weights', class_weights.float())

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target.long(),
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        valid = (target != self.ignore_index)
        if valid.sum().item() == 0:
            return ce.sum() * 0.0

        ce_valid = ce[valid]
        if self.use_focal and self.focal_gamma > 0.0:
            pt = torch.exp(-ce_valid)
            focal_factor = torch.pow(1.0 - pt, self.focal_gamma)
            return (focal_factor * ce_valid).mean()
        return ce_valid.mean()


def forward_in_chunks(net, x_cpu, device, num_chunks):
    chunk_outputs = []
    for start, end in get_chunk_ranges(x_cpu.shape[2], num_chunks):
        x_part = x_cpu[:, :, start:end, :].to(device, non_blocking=True)
        y_pred_part = net(x_part)
        chunk_outputs.append(y_pred_part)
        del x_part
    return torch.cat(chunk_outputs, dim=2)


def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1', 'y'):
            return True
        if v.lower() in ('no', 'false', 'f', '0', 'n'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=int,default=0)
    parser.add_argument('--data_set_path',type=str,default='./data')
    parser.add_argument('--work_dir',type=str,default='./')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--train_samples', type=int, default=30)
    parser.add_argument('--val_samples', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='RUNS')
    parser.add_argument('--record_computecost', type=str2bool, default=True)
    parser.add_argument('--seed_start', type=int, default=0)
    parser.add_argument('--cube_npy_path', type=str, default='')
    parser.add_argument('--gt_npy_path', type=str, default='')
    parser.add_argument('--custom_npy_chunks', type=int, default=32)
    parser.add_argument('--use_amp', type=str2bool, default=True)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--early_stop_min_delta', type=float, default=0.002)
    parser.add_argument('--class_weight_mode', type=str, default='inverse', choices=['none', 'inverse', 'effective'])
    parser.add_argument('--class_weight_beta', type=float, default=0.999)
    parser.add_argument('--class_weight_clip_min', type=float, default=0.2)
    parser.add_argument('--class_weight_clip_max', type=float, default=5.0)
    parser.add_argument('--use_focal', type=str2bool, default=False)
    parser.add_argument('--focal_gamma', type=float, default=1.5)
    parser.add_argument('--use_spectral_recon', type=str2bool, default=False)
    parser.add_argument('--alpha_recon', type=float, default=0.1)
    parser.add_argument('--lambda_sam', type=float, default=0.5)
    parser.add_argument('--lambda_l2', type=float, default=0.03)

    args = parser.parse_args()
    return args


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = get_parser()
record_computecost = args.record_computecost
exp_name = args.exp_name
custom_npy_chunks = max(1, args.custom_npy_chunks)
use_amp = bool(args.use_amp and torch.cuda.is_available())
early_stop_patience = max(0, int(args.early_stop_patience))
early_stop_min_delta = max(0.0, float(args.early_stop_min_delta))
class_weight_mode = args.class_weight_mode
class_weight_beta = float(args.class_weight_beta)
class_weight_clip_min = float(args.class_weight_clip_min)
class_weight_clip_max = float(args.class_weight_clip_max)
use_focal = bool(args.use_focal)
focal_gamma = float(args.focal_gamma)
use_spectral_recon = bool(args.use_spectral_recon)
alpha_recon = float(args.alpha_recon)
lambda_sam = float(args.lambda_sam)
lambda_l2 = float(args.lambda_l2)
all_seed_list = [0,1,2,3,4,5,6,7,8,9]
seed_start = min(max(0, int(args.seed_start)), len(all_seed_list) - 1)
seed_list = all_seed_list[seed_start:]
# seed_list = [9]  #
# seed_list = [5,6,7,8,9]  #

num_list = [args.train_samples, args.val_samples]

dataset_index = args.dataset_index

max_epoch = args.max_epoch
learning_rate = args.lr

net_name = 'MambaHSI_Plus'

paras_dict = {'net_name':net_name,'dataset_index':dataset_index,'num_list':num_list,
              'lr':learning_rate,'seed_list':seed_list,
              'class_weight_mode': class_weight_mode,
              'class_weight_beta': class_weight_beta,
              'use_focal': use_focal,
              'focal_gamma': focal_gamma,
              'use_spectral_recon': use_spectral_recon,
              'alpha_recon': alpha_recon,
              'lambda_sam': lambda_sam,
              'lambda_l2': lambda_l2}


                      # 0        1         2         3        4
data_set_name_list = ['UP', 'HanChuan', 'HongHu', 'Houston']
cube_npy_path = args.cube_npy_path
gt_npy_path = args.gt_npy_path
if not cube_npy_path and not gt_npy_path and os.path.exists(DEFAULT_CUBE_NPY) and os.path.exists(DEFAULT_GT_NPY):
    cube_npy_path = DEFAULT_CUBE_NPY
    gt_npy_path = DEFAULT_GT_NPY

if cube_npy_path and gt_npy_path:
    data_set_name = 'CustomNPY'
else:
    data_set_name = data_set_name_list[dataset_index]

if data_set_name in ['HanChuan', 'HongHu', 'Houston', 'CustomNPY']:
    split_image = True
else:
    split_image = False

transform = transforms.Compose([
    # transforms.Resize((2048, 1024)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=[123.6750, 116.2800, 103.5300], std=[58.395, 57.120, 57.3750]),
])


if __name__ == '__main__':
    data_set_path = args.data_set_path
    work_dir = args.work_dir
    setting_name = 'tr{}val{}'.format(str(args.train_samples),str(args.val_samples)) + '_lr{}'.format(str(learning_rate))

    dataset_name = data_set_name

    exp_name = args.exp_name

    save_folder = os.path.join(work_dir, exp_name, net_name, dataset_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("makedirs {}".format(save_folder))

    current_time = datetime.now().strftime("%m%d%H%M%S")
    new_filename = f"{current_time}_MambaHSI_Plus.py"
    shutil.copy("model/MambaHSI_Plus.py", os.path.join(save_folder, new_filename))

    save_log_path = os.path.join(save_folder,'train_tr{}_val{}.log'.format(num_list[0],num_list[1]))
    logger = setup_logger(name='{}'.format(dataset_name),logfile=save_log_path)
    torch.cuda.empty_cache()

    logger.info(save_folder)

    data, gt = data_load_operate.load_data(
        data_set_name,
        data_set_path,
        cube_npy_path=cube_npy_path,
        gt_npy_path=gt_npy_path,
    )
    is_custom_npy = bool(cube_npy_path and gt_npy_path)

    height, width, channels = data.shape

    gt_reshape = gt.reshape(-1)
    height, width, channels = data.shape
    if is_custom_npy:
        img = None
    else:
        img = ImageStretching(data)

    class_count = max(np.unique(gt))

    flag_list = [1, 0]  # ratio or num
    ratio_list = [0.005, 0.001]  # [train_ratio,val_ratio]

    loss_func = SegLoss(ignore_index=-1, use_focal=use_focal, focal_gamma=focal_gamma)

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(seed_list), class_count])
    evaluator = Evaluator(num_class=class_count)

    for exp_idx, curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)
        single_experiment_name = 'run{}_seed{}'.format(str(exp_idx), str(curr_seed))
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis')
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)
            print("makedirs {}".format(save_vis_folder))

        # shutil.copy("model/MambaHSI_Plus.py", save_folder)
        save_weight_path = os.path.join(save_single_experiment_folder, "best_tr{}_val{}.pth".format(num_list[0], num_list[1]))
        results_save_path = os.path.join(save_single_experiment_folder, 'result_tr{}_val{}.txt'.format(num_list[0], num_list[1]))
        predict_save_path = os.path.join(save_single_experiment_folder, 'pred_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        gt_save_path = os.path.join(save_single_experiment_folder, 'gt_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))

        train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(ratio_list,
                                                                                                       num_list,
                                                                                                       gt_reshape,
                                                                                                       class_count,
                                                                                                       flag_list[0])
        index = (train_data_index, val_data_index, test_data_index)
        train_label, val_label, test_label = data_load_operate.generate_image_iter(data, height, width, gt_reshape, index)

        
        if is_custom_npy:
            x = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).float()
            if data.dtype == np.uint8:
                x = x / 255.0
        else:
            x = transform(np.array(img))
            x = x.unsqueeze(0)
            x = x.float()
        print(x.shape)
        if not is_custom_npy:
            x = x.to(device)
        # build Model
        net = MambaHSI_Plus(in_channels=channels, num_classes=class_count)
        logger.info(paras_dict)
        logger.info(net)
        
        train_label = train_label.to(device)
        test_label = test_label.to(device)
        val_label = val_label.to(device)

        if is_custom_npy:
            train_valid = train_label[train_label >= 0].long().view(-1).cpu()
            if train_valid.numel() > 0:
                class_counts = torch.bincount(train_valid, minlength=class_count).float()
                class_weights = build_class_weights(
                    class_counts,
                    mode=class_weight_mode,
                    beta=class_weight_beta,
                    clip_min=class_weight_clip_min,
                    clip_max=class_weight_clip_max,
                )
                loss_func = SegLoss(
                    class_weights=class_weights.to(device) if class_weights is not None else None,
                    ignore_index=-1,
                    use_focal=use_focal,
                    focal_gamma=focal_gamma,
                )
                if class_weights is not None:
                    logger.info('class_counts:{}|class_weights:{}|mode:{}|beta:{}'.format(
                        class_counts.tolist(), class_weights.tolist(), class_weight_mode, class_weight_beta
                    ))
                else:
                    logger.info('class_counts:{}|class_weights:none|mode:{}'.format(
                        class_counts.tolist(), class_weight_mode
                    ))
            else:
                loss_func = SegLoss(ignore_index=-1, use_focal=use_focal, focal_gamma=focal_gamma)
        else:
            loss_func = SegLoss(ignore_index=-1, use_focal=use_focal, focal_gamma=focal_gamma)

        logger.info('loss_cfg|weight_mode:{}|use_focal:{}|focal_gamma:{}'.format(
            class_weight_mode, use_focal, focal_gamma
        ))

        # ############################################
        # val_label = test_label
        # ############################################

        net.to(device)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        train_loss_list = [100]
        train_acc_list = [0]
        val_loss_list = [100]
        val_acc_list = [0]

        optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

        # logger.info(optimizer)
        best_loss = 99999
        if is_custom_npy and record_computecost:
            logger.info("CustomNPY detected, skip FLOPs calculation to avoid long startup time.")
            record_computecost = False
        if record_computecost:
            net.eval()
            flops, macs1, para = calculate_flops(model=net,
                                                 input_shape=(1, x.shape[1], x.shape[2], x.shape[3]), )
            logger.info("para:{}\n,flops:{}".format(para, flops))

        tic1 = time.perf_counter()
        best_val_acc = 0
        no_improve_epochs = 0


        for epoch in range(max_epoch):
            y_train = train_label.unsqueeze(0)
            train_acc_sum, trained_samples_counter = 0.0, 0
            batch_counter, train_loss_sum = 0, 0
            time_epoch = time.time()
            loss_dict = {}

            net.train()

            if split_image:
                if is_custom_npy:
                    chunk_losses = []
                    chunk_ce_losses = []
                    chunk_recon_losses = []
                    skipped_empty_chunks = 0
                    skipped_nan_chunks = 0
                    chunk_ranges = get_chunk_ranges(x.shape[2], custom_npy_chunks)
                    grad_scale_denom = float(max(1, len(chunk_ranges)))
                    optimizer.zero_grad(set_to_none=True)
                    has_any_backward = False
                    for start, end in chunk_ranges:
                        y_part = y_train[:, start:end, :]
                        if (y_part != -1).sum().item() == 0:
                            skipped_empty_chunks += 1
                            continue
                        x_part = x[:, :, start:end, :].to(device, non_blocking=True)
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            if use_spectral_recon:
                                y_pred_part, recon_part = net(x_part, return_recon=True)
                            else:
                                y_pred_part = net(x_part)
                            if not torch.isfinite(y_pred_part).all().item():
                                skipped_nan_chunks += 1
                                if use_spectral_recon:
                                    del x_part, y_part, y_pred_part, recon_part
                                else:
                                    del x_part, y_part, y_pred_part
                                torch.cuda.empty_cache()
                                continue
                            ce_part = head_loss(loss_func, y_pred_part, y_part.long())
                            if use_spectral_recon:
                                recon_resized = resize(
                                    input=recon_part,
                                    size=x_part.shape[2:],
                                    mode='bilinear',
                                    align_corners=False,
                                    warning=False,
                                )
                                recon_part_loss = spectral_reconstruction_loss(
                                    recon_resized,
                                    x_part,
                                    y_part,
                                    lambda_sam=lambda_sam,
                                    lambda_l2=lambda_l2,
                                )
                                ls_part = ce_part + alpha_recon * recon_part_loss
                            else:
                                recon_part_loss = ce_part.new_tensor(0.0)
                                ls_part = ce_part
                        if not torch.isfinite(ls_part).item():
                            skipped_nan_chunks += 1
                            if use_spectral_recon:
                                del x_part, y_part, y_pred_part, recon_part, recon_resized
                            else:
                                del x_part, y_part, y_pred_part
                            torch.cuda.empty_cache()
                            continue
                        try:
                            scaler.scale(ls_part / grad_scale_denom).backward()
                            has_any_backward = True
                        except RuntimeError as e:
                            if 'returned nan values' in str(e).lower():
                                skipped_nan_chunks += 1
                                if use_spectral_recon:
                                    del x_part, y_part, y_pred_part, recon_part, recon_resized
                                else:
                                    del x_part, y_part, y_pred_part
                                torch.cuda.empty_cache()
                                continue
                            raise
                        chunk_losses.append(ls_part.detach().cpu())
                        chunk_ce_losses.append(ce_part.detach().cpu())
                        if use_spectral_recon:
                            chunk_recon_losses.append(recon_part_loss.detach().cpu())
                            del x_part, y_part, y_pred_part, recon_part, recon_resized
                        else:
                            del x_part, y_part, y_pred_part
                        torch.cuda.empty_cache()
                    if len(chunk_losses) == 0 or not has_any_backward:
                        logger.info('Name:{}|Seed:{}|Iter:{}|all chunks skipped|empty:{}|nan:{}'.format(
                            data_set_name, curr_seed, epoch, skipped_empty_chunks, skipped_nan_chunks
                        ))
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    has_bad_grad = False
                    for p in net.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all().item():
                            has_bad_grad = True
                            break
                    if has_bad_grad:
                        logger.info('Iter:{}|bad_grad_detected|skip_optimizer_step'.format(epoch))
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    total_loss = torch.stack(chunk_losses).sum().numpy()
                    if use_spectral_recon:
                        ce_loss_value = torch.stack(chunk_ce_losses).sum().numpy()
                        recon_loss_value = torch.stack(chunk_recon_losses).sum().numpy() if len(chunk_recon_losses) > 0 else 0.0
                        logger.info('Name:{}|Seed:{}|Iter:{}|ce_loss:{}|recon_loss:{}|total_loss:{}|chunks:{}'.format(
                            data_set_name, curr_seed, epoch, ce_loss_value, recon_loss_value, total_loss, custom_npy_chunks
                        ))
                    else:
                        logger.info('Name:{}|Seed:{}|Iter:{}|loss:{}|chunks:{}'.format(
                            data_set_name, curr_seed, epoch, total_loss, custom_npy_chunks
                        ))
                    if skipped_empty_chunks > 0 or skipped_nan_chunks > 0:
                        logger.info('Iter:{}|skipped_empty_chunks:{}|skipped_nan_chunks:{}'.format(
                            epoch, skipped_empty_chunks, skipped_nan_chunks
                        ))
                else:
                    x_part1 = x[:, :, :x.shape[2] // 2+5, :]
                    y_part1 = y_train[:,:x.shape[2] // 2+5,:]
                    x_part2 = x[:, :, x.shape[2] // 2 - 5: , :]
                    y_part2 = y_train[:,x.shape[2] // 2 - 5:,:]

                    if use_spectral_recon:
                        y_pred_part1, recon_part1 = net(x_part1, return_recon=True)
                        ce1 = head_loss(loss_func,y_pred_part1, y_part1.long())
                        recon_part1 = resize(input=recon_part1, size=x_part1.shape[2:], mode='bilinear', align_corners=False, warning=False)
                        recon1 = spectral_reconstruction_loss(recon_part1, x_part1, y_part1, lambda_sam=lambda_sam, lambda_l2=lambda_l2)
                        ls1 = ce1 + alpha_recon * recon1
                    else:
                        y_pred_part1 = net(x_part1)
                        ce1 = head_loss(loss_func,y_pred_part1, y_part1.long())
                        recon1 = ce1.new_tensor(0.0)
                        ls1 = ce1
                    optimizer.zero_grad()
                    ls1.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()

                    if use_spectral_recon:
                        y_pred_part2, recon_part2 = net(x_part2, return_recon=True)
                        ce2 = head_loss(loss_func,y_pred_part2, y_part2.long())
                        recon_part2 = resize(input=recon_part2, size=x_part2.shape[2:], mode='bilinear', align_corners=False, warning=False)
                        recon2 = spectral_reconstruction_loss(recon_part2, x_part2, y_part2, lambda_sam=lambda_sam, lambda_l2=lambda_l2)
                        ls2 = ce2 + alpha_recon * recon2
                    else:
                        y_pred_part2 = net(x_part2)
                        ce2 = head_loss(loss_func,y_pred_part2, y_part2.long())
                        recon2 = ce2.new_tensor(0.0)
                        ls2 = ce2
                    optimizer.zero_grad()
                    ls2.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()
                    if use_spectral_recon:
                        logger.info('Name:{}|Seed:{}|Iter:{}|ce_loss:{}|recon_loss:{}|total_loss:{}'.format(
                            data_set_name,
                            curr_seed,
                            epoch,
                            (ce1 + ce2).detach().cpu().numpy(),
                            (recon1 + recon2).detach().cpu().numpy(),
                            (ls1 + ls2).detach().cpu().numpy(),
                        ))
                    else:
                        logger.info('Name:{}|Seed:{}|Iter:{}|loss:{}'.format(data_set_name, curr_seed, epoch, (ls1 + ls2).detach().cpu().numpy()))
            else:
                try:
                    if use_spectral_recon:
                        y_pred, recon = net(x, return_recon=True)
                        ce_loss = head_loss(loss_func,y_pred, y_train.long())
                        recon = resize(input=recon, size=x.shape[2:], mode='bilinear', align_corners=False, warning=False)
                        recon_loss = spectral_reconstruction_loss(recon, x, y_train, lambda_sam=lambda_sam, lambda_l2=lambda_l2)
                        ls = ce_loss + alpha_recon * recon_loss
                    else:
                        y_pred = net(x)
                        ce_loss = head_loss(loss_func,y_pred, y_train.long())
                        recon_loss = ce_loss.new_tensor(0.0)
                        ls = ce_loss
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()
                    if use_spectral_recon:
                        logger.info('Name:{}|Seed:{}|Iter:{}|ce_loss:{}|recon_loss:{}|total_loss:{}|seed:{}'.format(
                            data_set_name,
                            curr_seed,
                            epoch,
                            ce_loss.detach().cpu().numpy(),
                            recon_loss.detach().cpu().numpy(),
                            ls.detach().cpu().numpy(),
                            curr_seed,
                        ))
                    else:
                        logger.info('Name:{}|Seed:{}|Iter:{}|loss:{}|seed:{}'.format(data_set_name, curr_seed, epoch, ls.detach().cpu().numpy(), curr_seed))
                except:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    split_image=True
                    x_part1 = x[:, :, :x.shape[2] // 2 + 5, :]
                    y_part1 = y_train[:, :x.shape[2] // 2 + 5, :]
                    x_part2 = x[:, :, x.shape[2] // 2 - 5:, :]
                    y_part2 = y_train[:, x.shape[2] // 2 - 5:, :]

                    if use_spectral_recon:
                        y_pred_part1, recon_part1 = net(x_part1, return_recon=True)
                        ce1 = head_loss(loss_func, y_pred_part1, y_part1.long())
                        recon_part1 = resize(input=recon_part1, size=x_part1.shape[2:], mode='bilinear', align_corners=False, warning=False)
                        recon1 = spectral_reconstruction_loss(recon_part1, x_part1, y_part1, lambda_sam=lambda_sam, lambda_l2=lambda_l2)
                        ls1 = ce1 + alpha_recon * recon1
                    else:
                        y_pred_part1 = net(x_part1)
                        ce1 = head_loss(loss_func, y_pred_part1, y_part1.long())
                        recon1 = ce1.new_tensor(0.0)
                        ls1 = ce1
                    optimizer.zero_grad()
                    ls1.backward()
                    optimizer.step()

                    if use_spectral_recon:
                        y_pred_part2, recon_part2 = net(x_part2, return_recon=True)
                        ce2 = head_loss(loss_func, y_pred_part2, y_part2.long())
                        recon_part2 = resize(input=recon_part2, size=x_part2.shape[2:], mode='bilinear', align_corners=False, warning=False)
                        recon2 = spectral_reconstruction_loss(recon_part2, x_part2, y_part2, lambda_sam=lambda_sam, lambda_l2=lambda_l2)
                        ls2 = ce2 + alpha_recon * recon2
                    else:
                        y_pred_part2 = net(x_part2)
                        ce2 = head_loss(loss_func, y_pred_part2, y_part2.long())
                        recon2 = ce2.new_tensor(0.0)
                        ls2 = ce2
                    optimizer.zero_grad()
                    ls2.backward()
                    optimizer.step()

                    if use_spectral_recon:
                        logger.info('Name:{}|Seed:{}|Iter:{}|ce_loss:{}|recon_loss:{}|total_loss:{}'.format(
                            data_set_name,
                            curr_seed,
                            epoch,
                            (ce1 + ce2).detach().cpu().numpy(),
                            (recon1 + recon2).detach().cpu().numpy(),
                            (ls1 + ls2).detach().cpu().numpy(),
                        ))
                    else:
                        logger.info(
                            'Name:{}|Seed:{}|Iter:{}|loss:{}'.format(data_set_name, curr_seed, epoch, (ls1 + ls2).detach().cpu().numpy()))

            torch.cuda.empty_cache()
            # evaluate stage
            net.eval()
            with torch.no_grad():
                evaluator.reset()
                if is_custom_npy:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output_val = forward_in_chunks(net, x, device, custom_npy_chunks)
                else:
                    output_val = net(x)
                y_val = val_label.unsqueeze(0)
                seg_logits = resize(input=output_val,
                                    size=y_val.shape[1:],
                                    mode='bilinear',
                                    align_corners=True)
                predict = torch.argmax(seg_logits,dim=1).cpu().numpy()
                Y_val_np = val_label.cpu().numpy()
                Y_val_255 = np.where(Y_val_np==-1,255,Y_val_np)
                evaluator.add_batch(np.expand_dims(Y_val_255,axis=0),predict)
                OA = evaluator.Pixel_Accuracy()
                mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                Kappa = evaluator.Kappa()
                logger.info('Evaluate {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA,mAcc,Kappa,mIOU,IOU,Acc))
                # save weight
                if OA >= (best_val_acc + early_stop_min_delta):
                    best_epoch = epoch + 1
                    best_val_acc = OA
                    no_improve_epochs = 0
                    # torch.save(net,save_weight_path)
                    torch.save(net.state_dict(), save_weight_path)
                    # save_epoch_weight_path = os.path.join(save_folder,'{}.pth'.format(str(epoch+1)))
                    # torch.save(net.state_dict(), save_epoch_weight_path)
                else:
                    no_improve_epochs += 1
                if (epoch+1)%50==0:
                    save_single_predict_path = os.path.join(save_vis_folder,'predict_{}.png'.format(str(epoch+1)))
                    save_single_gt_path = os.path.join(save_vis_folder,'gt.png')
                    vis_a_image(gt,predict,save_single_predict_path, save_single_gt_path)

                if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
                    logger.info('Early stopping at epoch {}: no OA improvement >= {} for {} epochs'.format(
                        epoch + 1, early_stop_min_delta, early_stop_patience
                    ))
                    break

                # net.train()
            torch.cuda.empty_cache()


        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        pred_test = []

        load_weight_path = save_weight_path
        net.update_params = None
        # best_net = copy.deepcopy(net)
        best_net = MambaHSI_Plus(in_channels=channels, num_classes=class_count, hidden_dim=128)

        best_net.to(device)
        best_net.load_state_dict(torch.load(load_weight_path))
        best_net.eval()
        test_evaluator = Evaluator(num_class=class_count)
        with torch.no_grad():
            test_evaluator.reset()
            if is_custom_npy:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    output_test = forward_in_chunks(best_net, x, device, custom_npy_chunks)
            else:
                output_test = best_net(x)

            y_test = test_label.unsqueeze(0)
            seg_logits_test = resize(input=output_test,
                                size=y_test.shape[1:],
                                mode='bilinear',
                                align_corners=True)
            predict_test = torch.argmax(seg_logits_test, dim=1).cpu().numpy()
            Y_test_np = test_label.cpu().numpy()
            Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)
            test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), predict_test)
            OA_test = test_evaluator.Pixel_Accuracy()
            mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
            mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
            Kappa_test = evaluator.Kappa()
            logger.info('Test {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA_test, mAcc_test, Kappa_test, mIOU_test, IOU_test,
                                                                                    Acc_test))
            vis_a_image(gt, predict_test, predict_save_path, gt_save_path)
        # Output infors
        f = open(results_save_path, 'a+')
        str_results = '\n======================' \
                      + " exp_idx=" + str(exp_idx) \
                      + " seed=" + str(curr_seed) \
                      + " learning rate=" + str(learning_rate) \
                      + " epochs=" + str(max_epoch) \
                      + " train ratio=" + str(ratio_list[0]) \
                      + " val ratio=" + str(ratio_list[1]) \
                      + " ======================" \
                      + "\nOA=" + str(OA_test) \
                      + "\nAA=" + str(mAcc_test) \
                      + '\nkpp=' + str(Kappa_test) \
                      + '\nmIOU_test:' + str(mIOU_test) \
                      + "\nIOU_test:" + str(IOU_test) \
                      + "\nAcc_test:" + str(Acc_test) + "\n"
        logger.info(str_results)
        f.write(str_results)
        f.close()

        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)

        torch.cuda.empty_cache()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    logger.info("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    logger.info('List of OA: %s', list(OA_ALL))
    logger.info('List of AA: %s', list(AA_ALL))
    logger.info('List of KPP: %s', list(KPP_ALL))
    logger.info('OA= %s +- %s', round(np.mean(OA_ALL) * 100, 2), round(np.std(OA_ALL) * 100, 2))
    logger.info('AA= %s +- %s', round(np.mean(AA_ALL) * 100, 2), round(np.std(AA_ALL) * 100, 2))
    logger.info('Kpp= %s +- %s', round(np.mean(KPP_ALL) * 100, 2), round(np.std(KPP_ALL) * 100, 2))
    # logger.info('Acc per class=', np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2), '+-', np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2))
    mean_acc = np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2)
    std_acc = np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2)
    logger.info('Acc per class= %s +- %s', mean_acc, std_acc)

    # logger.info("Average training time=", round(np.mean(Train_Time_ALL), 2), '+-', round(np.std(Train_Time_ALL), 3))
    # logger.info("Average testing time=", round(np.mean(Test_Time_ALL) * 1000, 2), '+-', round(np.std(Test_Time_ALL) * 1000, 3))
    # logger.info("Average training time= %s +- %s", round(np.mean(Train_Time_ALL), 2), round(np.std(Train_Time_ALL), 3))
    # logger.info("Average testing time= %s +- %s", round(np.mean(Test_Time_ALL) * 1000, 2), round(np.std(Test_Time_ALL) * 1000, 3))

    # Output infors
    mean_result_path = os.path.join(save_folder,'mean_result.txt')
    f = open(mean_result_path, 'w')
    str_results = '\n\n***************Mean result of ' + str(len(seed_list)) + 'times runs ********************' \
                  + '\nList of OA:' + str(list(OA_ALL)) \
                  + '\nList of AA:' + str(list(AA_ALL)) \
                  + '\nList of KPP:' + str(list(KPP_ALL)) \
                  + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                  + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                  + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(
        round(np.std(KPP_ALL) * 100, 2)) \
                  + '\nAcc per class=\n' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)) + '+-' + str(
        np.round(np.std(EACH_ACC_ALL, 0) * 100, 2))
        #  \
        #           + "\nAverage training time=" + str(
        # np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' + str(
        # np.round(np.std(Train_Time_ALL), decimals=3)) \
        #           + "\nAverage testing time=" + str(
        # np.round(np.mean(Test_Time_ALL) * 1000, decimals=2)) + '+-' + str(
        # np.round(np.std(Test_Time_ALL) * 100, decimals=3))
    f.write(str_results)
    f.close()

    del net, x, img
