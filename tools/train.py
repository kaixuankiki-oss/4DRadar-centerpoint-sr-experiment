import _init_path
import argparse
import datetime
import glob
import json
import logging
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.structured_log import emit_metric
from train_utils.training_output import resolve_training_output_dir
from train_utils.train_utils import train_model


BEST_CHECKPOINTS_FILE = 'best_checkpoints.json'


def create_file_logger(name, log_file, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(log_level if rank == 0 else logging.ERROR)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setLevel(log_level if rank == 0 else logging.ERROR)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


class MainTrainingLogFilter(logging.Filter):
    """Keep iterative training progress out of the main experiment log."""

    TRAIN_ONLY_PREFIXES = (
        'TRAIN_METRIC ', 'EPOCH_METRIC ', 'EVAL_METRIC ',
        'CHECKPOINT_METRIC ', 'BEST_CHECKPOINTS ', 'FINAL_BEST_CHECKPOINT_EVAL ',
        'Train:',
    )

    def filter(self, record):
        message = record.getMessage()
        return not message.startswith(self.TRAIN_ONLY_PREFIXES)


def filter_main_training_log(logger):
    training_filter = MainTrainingLogFilter()
    for handler in logger.handlers:
        handler.addFilter(training_filter)


def load_best_checkpoint_record(ckpt_dir, logger=None):
    best_file = ckpt_dir / BEST_CHECKPOINTS_FILE
    if not best_file.exists():
        if logger is not None:
            logger.warning('Best checkpoint record does not exist: %s', best_file)
        return None

    try:
        with open(best_file, 'r') as f:
            records = json.load(f)
    except (OSError, json.JSONDecodeError) as err:
        if logger is not None:
            logger.warning('Failed to read best checkpoint record %s: %s', best_file, err)
        return None

    valid_records = []
    for record in records:
        if 'path' not in record or 'score' not in record:
            continue
        ckpt_path = ckpt_dir / record['path']
        if ckpt_path.exists():
            valid_records.append(record)
        elif logger is not None:
            logger.warning('Best checkpoint candidate is missing: %s', ckpt_path)
    if not valid_records:
        if logger is not None:
            logger.warning('No valid best checkpoint is available in %s', best_file)
        return None

    return sorted(
        valid_records,
        key=lambda item: (float(item.get('score', float('-inf'))), int(item.get('epoch', 0))),
        reverse=True,
    )[0]


def remove_eval_with_train_dir(eval_output_dir, logger=None):
    if not eval_output_dir.exists():
        return

    try:
        shutil.rmtree(eval_output_dir)
        if logger is not None:
            logger.info('Removed per-epoch evaluation artifacts: %s', eval_output_dir)
    except OSError as err:
        if logger is not None:
            logger.warning('Failed to remove per-epoch evaluation artifacts %s: %s', eval_output_dir, err)


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument(
        '--local_rank', '--local-rank', dest='local_rank', type=int, default=None,
        help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=3, help='max number of best-mAP checkpoints to keep')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument(
        '--structured_log_iter_interval', type=int, default=10,
        help='record TRAIN_METRIC every N iterations; epoch boundaries are always recorded'
    )
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    

    args = parser.parse_args()

    if args.structured_log_iter_interval <= 0:
        parser.error('--structured_log_iter_interval must be greater than zero')
    if args.max_ckpt_save_num <= 0:
        parser.error('--max_ckpt_save_num must be greater than zero')
    args.max_ckpt_save_num = min(args.max_ckpt_save_num, 3)

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir, training_source = resolve_training_output_dir(
        cfg.get('OUTPUT_ROOT', 'output'), cfg.ROOT_DIR,
        cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag,
    )
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = output_dir / ('train_%s.log' % log_time)
    train_metrics_log_file = output_dir / ('train_metrics_%s.log' % log_time)
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    filter_main_training_log(logger)
    train_logger = create_file_logger(
        'hr4d_train_metrics.%s.%s' % (cfg.TAG, args.extra_tag),
        train_metrics_log_file,
        rank=cfg.LOCAL_RANK
    )

    # log to file
    logger.info('**********************Start logging**********************')
    logger.info('Training metrics are saved to %s', train_metrics_log_file)
    train_logger.info('**********************Start training metrics logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')
        
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    emit_metric(logger, 'RUN_META', {
        'run_name': args.extra_tag,
        'config': str(args.cfg_file),
        'output_dir': str(output_dir),
        'model': cfg.MODEL.NAME,
        'epochs': args.epochs,
        'batch_size_per_gpu': args.batch_size,
        'world_size': total_gpus,
        'amp': args.use_amp,
        'structured_log_iter_interval': args.structured_log_iter_interval,
        'branch': os.environ.get('GIT_BRANCH', 'training_log_qiyunlong'),
        **training_source,
    })

    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
              
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    logger.info('**********************Prepare per-epoch evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    def eval_after_epoch(epoch_id):
        logger.info('**********************Start evaluation epoch %s %s/%s(%s)**********************' %
                    (epoch_id, cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
        eval_model = model.module if dist_train else model
        cur_epoch_dir = eval_output_dir / ('epoch_%s' % epoch_id)
        cur_result_dir = cur_epoch_dir / cfg.DATA_CONFIG.DATA_SPLIT['test']
        metrics = eval_utils.eval_one_epoch(
            cfg, args, eval_model, test_loader, epoch_id, logger, dist_test=dist_train,
            result_dir=cur_result_dir, metric_logger=train_logger
        )
        eval_model.train()
        if cfg.LOCAL_RANK == 0:
            remove_eval_with_train_dir(cur_epoch_dir, logger=logger)
        logger.info('**********************End evaluation epoch %s %s/%s(%s)**********************' %
                    (epoch_id, cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
        return metrics

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=None,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, 
        logger=logger, 
        logger_iter_interval=args.logger_iter_interval,
        structured_log_iter_interval=args.structured_log_iter_interval,
        ckpt_save_time_interval=None,
        use_logger_to_record=not args.use_tqdm_to_record, 
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg,
        per_epoch_eval_func=eval_after_epoch,
        train_logger=train_logger
    )

    if dist_train:
        torch.distributed.barrier()

    if cfg.LOCAL_RANK == 0:
        logger.info('**********************Per-epoch evaluation completed %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    best_record = load_best_checkpoint_record(ckpt_dir, logger=logger if cfg.LOCAL_RANK == 0 else None)
    if best_record is not None:
        best_ckpt_path = ckpt_dir / best_record['path']
        best_epoch = best_record.get('epoch', 'unknown')
        best_epoch_id = int(best_epoch)
        best_score_name = best_record.get('score_name', 'unknown')
        best_score = float(best_record.get('score', float('nan')))
        if cfg.LOCAL_RANK == 0:
            logger.info(
                '**********************Start final best-checkpoint evaluation epoch %s (%s=%.6f): %s**********************',
                best_epoch, best_score_name, best_score, best_ckpt_path
            )

        eval_model = model.module if dist_train else model
        eval_model.load_params_from_file(
            filename=str(best_ckpt_path),
            to_cpu=dist_train,
            logger=logger,
        )
        final_result_dir = output_dir / 'eval' / 'best_checkpoint' / ('epoch_%s' % best_epoch) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        final_metrics = eval_utils.eval_one_epoch(
            cfg, args, eval_model, test_loader, best_epoch_id, logger, dist_test=dist_train,
            result_dir=final_result_dir, metric_logger=train_logger
        )
        eval_model.train()
        if cfg.LOCAL_RANK == 0:
            emit_metric(train_logger or logger, 'FINAL_BEST_CHECKPOINT_EVAL', {
                'epoch': best_epoch,
                'checkpoint': str(best_ckpt_path),
                'score_name': best_score_name,
                'score': best_score,
                'metrics': final_metrics,
            })
            logger.info('**********************End final best-checkpoint evaluation epoch %s**********************', best_epoch)
            remove_eval_with_train_dir(eval_output_dir, logger=logger)
    elif cfg.LOCAL_RANK == 0:
        logger.warning('Skip final best-checkpoint evaluation because no best checkpoint is available.')

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
