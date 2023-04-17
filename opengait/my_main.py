import os
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr


parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str,
                    default='config/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--device', default='1', type=str, help='device id, 0 or 1 or 0,1')
opt = parser.parse_args()

MASTER_ADDR = "localhost"
MASTER_PORT = "12346"

# opt.cfgs = "configs/gaitgl/gaitgl_GREW_BNNeck.yaml"

# opt.cfgs = "configs/baseline/baseline_hid.yaml"
# opt.cfgs = "configs/gaitgl/gaitgl_HID.yaml"
# opt.cfgs = "configs/gaitgl/gaitgl_HID_OutdoorGait.yaml"

# opt.cfgs = "configs/baseline/baseline.yaml"
# opt.cfgs = "configs/gaitpart/gaitpart.yaml"
# opt.cfgs = "configs/gln/gln_phase1.yaml"
# opt.cfgs = "configs/gln/gln_phase1.yaml"
# opt.cfgs = "configs/gaitset/gaitset.yaml"
# opt.cfgs = "configs/gaitgl/gaitgl.yaml"

# opt.cfgs = "configs/gaitgl/gaitgl_OutdoorGait.yaml"

# opt.cfgs = "configs/baseline/baseline_OUMVLP.yaml"
# opt.cfgs = "configs/gaitpart/gaitpart_OUMVLP.yaml"
# opt.cfgs = "configs/gaitset/gaitset_OUMVLP.yaml"
# opt.cfgs = "configs/gaitgl/gaitgl_OUMVLP.yaml"
# opt.cfgs = "configs/gaitgl/gaitgl_OUMVLP_mixed.yaml"
# opt.cfgs = "configs/gaitgl/gaitgl_OUMVLP_mixed2.yaml"

opt.cfgs = "configs/gaitgl/gaitgl_HID_OutdoorGait_CASIA-B_OUMVLP.yaml"

opt.phase = 'test'
# opt.phase = 'train'

opt.log_to_file = True


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    model = get_ddp_module(model)
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        Model.run_train(model)
    else:
        Model.run_test(model)


def main(rank, args):
    """
    rank表示进程序号，用于进程间通讯，每一个进程对应了一个rank,单机多卡中可以理解为第几个GPU。
    args为函数传入的参数
    """

    # Environment settings
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    print('Distributed init (rank {}): {}'.format(rank, 'env://'), flush=True)

    # Windows does not support nccl backend, use gloo instead of nccl, it is recommended to use nccl on Linux.
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    run_model(cfgs, training)


if __name__ == '__main__':
    WORK_PATH = "."
    os.chdir(WORK_PATH)
    print("WORK_PATH:", os.getcwd())
    mp.spawn(main,
             args=(opt,),
             nprocs=opt.world_size,
             join=True)

