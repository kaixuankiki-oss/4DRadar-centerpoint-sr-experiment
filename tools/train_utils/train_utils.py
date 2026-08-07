import json
import math
import os
import re

import torch
import tqdm
import time
import glob
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from train_utils.structured_log import emit_metric, gpu_snapshot, should_emit_train_metric


BEST_CHECKPOINTS_FILE = 'best_checkpoints.json'


def _extract_map_score(metrics):
    if not metrics:
        return None, None

    preferred_keys = ('hr4d/mean_ap', 'mAP', 'mean_ap', 'map')
    for key in preferred_keys:
        if key in metrics:
            try:
                score = float(metrics[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                return key, score

    for key, value in metrics.items():
        if 'map' not in str(key).lower():
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            return key, score
    return None, None


def _load_best_checkpoints(ckpt_save_dir):
    best_file = ckpt_save_dir / BEST_CHECKPOINTS_FILE
    if best_file.exists():
        try:
            with open(best_file, 'r') as f:
                records = json.load(f)
            return [record for record in records if 'path' in record and 'score' in record]
        except (OSError, json.JSONDecodeError):
            pass

    records = []
    for ckpt_path in glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth')):
        match = re.search(r'checkpoint_epoch_(\d+)\.pth$', os.path.basename(ckpt_path))
        if match is None:
            continue
        records.append({
            'epoch': int(match.group(1)),
            'score': -1.0,
            'score_name': 'unknown',
            'path': os.path.basename(ckpt_path),
        })
    records.sort(key=lambda item: (float(item['score']), int(item.get('epoch', 0))), reverse=True)
    return records


def _rank_best_checkpoints(records, candidate, top_k):
    by_epoch = {}
    for record in records + [candidate]:
        epoch = int(record.get('epoch', -1))
        previous = by_epoch.get(epoch)
        if previous is None or float(record['score']) >= float(previous['score']):
            by_epoch[epoch] = record
    ranked = sorted(
        by_epoch.values(),
        key=lambda item: (float(item['score']), int(item.get('epoch', 0))),
        reverse=True,
    )
    return ranked[:top_k]


def _write_best_checkpoints(ckpt_save_dir, records):
    best_file = ckpt_save_dir / BEST_CHECKPOINTS_FILE
    with open(best_file, 'w') as f:
        json.dump(records, f, indent=2, sort_keys=True)


def _prune_non_best_checkpoints(ckpt_save_dir, records, logger=None):
    keep_files = {record['path'] for record in records}
    ckpt_paths = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
    ckpt_paths += glob.glob(str(ckpt_save_dir / 'latest_model.pth'))
    for ckpt_path in ckpt_paths:
        if os.path.basename(ckpt_path) in keep_files:
            continue
        try:
            os.remove(ckpt_path)
            if logger is not None:
                logger.info('Remove non-top checkpoint: %s', ckpt_path)
        except OSError as err:
            if logger is not None:
                logger.warning('Failed to remove non-top checkpoint %s: %s', ckpt_path, err)


def _cuda_timer_start():
    if not torch.cuda.is_available():
        return time.perf_counter()
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _cuda_timer_stop(start):
    if not torch.cuda.is_available():
        return (time.perf_counter() - start) * 1000.0
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, 
                    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None, 
                    structured_log_iter_interval=10, total_epochs=None, ckpt_save_dir=None,
                    ckpt_save_time_interval=300, show_gpu_stat=False, use_amp=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0**16))
    
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        backward_time = common_utils.AverageMeter()
        optimizer_time = common_utils.AverageMeter()
        grad_norm_m = common_utils.AverageMeter()
        losses_m = common_utils.AverageMeter()

    end = time.time()
    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        
        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter, cur_epoch)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        forward_timer = _cuda_timer_start()
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, tb_dict, disp_dict = model_func(model, batch)
        cur_forward_time = _cuda_timer_stop(forward_timer) / 1000.0

        backward_timer = _cuda_timer_start()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = float(clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP))
        cur_backward_time = _cuda_timer_stop(backward_timer) / 1000.0

        optimizer_timer = _cuda_timer_start()
        scaler.step(optimizer)
        scaler.update()
        cur_optimizer_time = _cuda_timer_stop(optimizer_timer) / 1000.0

        accumulated_iter += 1
 
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_backward_time = commu_utils.average_reduce_value(cur_backward_time)
        avg_optimizer_time = commu_utils.average_reduce_value(cur_optimizer_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)
        avg_grad_norm = commu_utils.average_reduce_value(grad_norm)

        # Log human-readable progress and structured metrics.
        if rank == 0:
            batch_size = batch.get('batch_size', None)
            
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            backward_time.update(avg_backward_time)
            optimizer_time.update(avg_optimizer_time)
            batch_time.update(avg_batch_time)
            grad_norm_m.update(avg_grad_norm)
            losses_m.update(loss.item() , batch_size)
            
            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})',
                'bw_time': f'{backward_time.val:.2f}({backward_time.avg:.2f})',
                'opt_time': f'{optimizer_time.val:.2f}({optimizer_time.avg:.2f})',
                'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})',
                'grad': f'{grad_norm_m.val:.3f}({grad_norm_m.avg:.3f})'
            })

            if should_emit_train_metric(
                    global_iteration=accumulated_iter,
                    iteration=cur_it + 1,
                    iterations_per_epoch=total_it_each_epoch,
                    interval=structured_log_iter_interval,
                    is_first_iteration=(cur_it == start_it)):
                trained_seconds = tbar.format_dict['elapsed'] + pbar.format_dict['elapsed']
                seconds_per_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1)
                remaining_iters = (
                    (total_epochs - cur_epoch - 1) * total_it_each_epoch
                    + total_it_each_epoch - cur_it - 1
                )
                metric = {
                    'epoch': cur_epoch + 1,
                    'total_epochs': total_epochs,
                    'iteration': cur_it + 1,
                    'iterations_per_epoch': total_it_each_epoch,
                    'global_iteration': accumulated_iter,
                    'progress': accumulated_iter / max(total_epochs * total_it_each_epoch, 1),
                    'loss': float(loss.item()),
                    'loss_avg': float(losses_m.avg),
                    'learning_rate': float(cur_lr),
                    'grad_norm': float(avg_grad_norm),
                    'data_time_ms': float(avg_data_time * 1000),
                    'forward_time_ms': float(avg_forward_time * 1000),
                    'backward_time_ms': float(avg_backward_time * 1000),
                    'optimizer_time_ms': float(avg_optimizer_time * 1000),
                    'batch_time_ms': float(avg_batch_time * 1000),
                    'elapsed_seconds': float(trained_seconds),
                    'eta_seconds': float(seconds_per_iter * remaining_iters),
                    'amp_scale': float(scaler.get_scale()),
                    'losses': {key: float(val) for key, val in tb_dict.items()},
                    **gpu_snapshot(include_utilization=(
                        cur_it == start_it or accumulated_iter % logger_iter_interval == 0
                    )),
                }
                emit_metric(logger, 'TRAIN_METRIC', metric)
            
            if use_logger_to_record:
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)
                    
                    logger.info(
                        'Train: {:>4d}/{} ({:>3.0f}%) [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'LR: {lr:.3e}  '
                        f'Time cost: {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)} ' 
                        f'[{tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}]  '
                        'Acc_iter {acc_iter:<10d}  '
                        'Data time: {data_time.val:.2f}({data_time.avg:.2f})  '
                        'Forward time: {forward_time.val:.2f}({forward_time.avg:.2f})  '
                        'Backward time: {backward_time.val:.2f}({backward_time.avg:.2f})  '
                        'Optimizer time: {optimizer_time.val:.2f}({optimizer_time.avg:.2f})  '
                        'Grad norm: {grad_norm.val:.3f}({grad_norm.avg:.3f})  '
                        'Batch time: {batch_time.val:.2f}({batch_time.avg:.2f})'.format(
                            cur_epoch+1,total_epochs, 100. * (cur_epoch+1) / total_epochs,
                            cur_it,total_it_each_epoch, 100. * cur_it / total_it_each_epoch,
                            loss=losses_m,
                            lr=cur_lr,
                            acc_iter=accumulated_iter,
                            data_time=data_time,
                            forward_time=forward_time,
                            backward_time=backward_time,
                            optimizer_time=optimizer_time,
                            grad_norm=grad_norm_m,
                            batch_time=batch_time
                            )
                    )
                    
                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                        # To show the GPU utilization, please install gpustat through "pip install gpustat"
                        gpu_info = os.popen('gpustat').read()
                        logger.info(gpu_info)
            else:                
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                # tbar.refresh()

            # save intermediate ckpt every {ckpt_save_time_interval} seconds         
            time_past_this_epoch = pbar.format_dict['elapsed']
            if (
                    ckpt_save_time_interval is not None
                    and ckpt_save_time_interval > 0
                    and time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt):
                ckpt_name = ckpt_save_dir / 'latest_model'
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                )
                logger.info(f'Save latest model to {ckpt_name}')
                ckpt_save_cnt += 1
                
    if rank == 0:
        emit_metric(logger, 'EPOCH_METRIC', {
            'epoch': cur_epoch + 1,
            'total_epochs': total_epochs,
            'global_iteration': accumulated_iter,
            'loss_avg': float(losses_m.avg),
            'grad_norm_avg': float(grad_norm_m.avg),
            'data_time_ms_avg': float(data_time.avg * 1000),
            'forward_time_ms_avg': float(forward_time.avg * 1000),
            'backward_time_ms_avg': float(backward_time.avg * 1000),
            'optimizer_time_ms_avg': float(optimizer_time.avg * 1000),
            'batch_time_ms_avg': float(batch_time.avg * 1000),
            **gpu_snapshot(include_utilization=True),
        })
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, use_amp=False,
                use_logger_to_record=False, logger=None, logger_iter_interval=None,
                structured_log_iter_interval=10, ckpt_save_time_interval=None, show_gpu_stat=False, cfg=None,
                per_epoch_eval_func=None, train_logger=None):
    accumulated_iter = start_iter
    best_checkpoints = _load_best_checkpoints(ckpt_save_dir) if rank == 0 else []
    progress_logger = train_logger if train_logger is not None else logger
    metric_logger = train_logger if train_logger is not None else logger

    # use for disable data augmentation hook
    hook_config = cfg.get('HOOK', None) 
    augment_disable_flag = False

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            
            augment_disable_flag = disable_augmentation_hook(
                hook_config, dataloader_iter, total_epochs, cur_epoch, cfg,
                augment_disable_flag, progress_logger
            )
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter, 
                
                cur_epoch=cur_epoch, total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record, 
                logger=progress_logger, logger_iter_interval=logger_iter_interval,
                structured_log_iter_interval=structured_log_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval, 
                show_gpu_stat=show_gpu_stat,
                use_amp=use_amp
            )

            trained_epoch = cur_epoch + 1
            if per_epoch_eval_func is not None:
                eval_metrics = per_epoch_eval_func(trained_epoch)
                if rank == 0:
                    score_name, score = _extract_map_score(eval_metrics)
                    emit_metric(metric_logger, 'CHECKPOINT_METRIC', {
                        'epoch': trained_epoch,
                        'score_name': score_name,
                        'map': score,
                        'metrics': eval_metrics,
                    })
                    if score is None:
                        logger.warning('Epoch %d evaluation has no finite mAP metric; checkpoint is not saved.', trained_epoch)
                        continue

                    ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                    candidate = {
                        'epoch': trained_epoch,
                        'score': float(score),
                        'score_name': score_name,
                        'path': '%s.pth' % ckpt_name.name,
                    }
                    ranked = _rank_best_checkpoints(best_checkpoints, candidate, max_ckpt_save_num)
                    should_save = any(record['epoch'] == trained_epoch for record in ranked)
                    if should_save:
                        save_checkpoint(
                            checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                        )
                        best_checkpoints = ranked
                        _prune_non_best_checkpoints(ckpt_save_dir, best_checkpoints, logger=progress_logger)
                        _write_best_checkpoints(ckpt_save_dir, best_checkpoints)
                        logger.info(
                            'Save top-%d checkpoint for epoch %d (%s=%.6f): %s',
                            max_ckpt_save_num, trained_epoch, score_name, score, ckpt_name
                        )
                    else:
                        logger.info(
                            'Epoch %d checkpoint is not saved because %s=%.6f is outside top-%d.',
                            trained_epoch, score_name, score, max_ckpt_save_num
                        )
                    emit_metric(metric_logger, 'BEST_CHECKPOINTS', {
                        'top_k': max_ckpt_save_num,
                        'checkpoints': best_checkpoints,
                    })
            elif trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )
                if progress_logger is not None:
                    progress_logger.info('Save checkpoint to %s', ckpt_name)


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)


def disable_augmentation_hook(hook_config, dataloader, total_epochs, cur_epoch, cfg, flag, logger):
    """
    This hook turns off the data augmentation during training.
    """
    if hook_config is not None:
        DisableAugmentationHook = hook_config.get('DisableAugmentationHook', None)
        if DisableAugmentationHook is not None:
            num_last_epochs = DisableAugmentationHook.NUM_LAST_EPOCHS
            if (total_epochs - num_last_epochs) <= cur_epoch and not flag:
                DISABLE_AUG_LIST = DisableAugmentationHook.DISABLE_AUG_LIST
                dataset_cfg=cfg.DATA_CONFIG
                logger.info(f'Disable augmentations: {DISABLE_AUG_LIST}')
                dataset_cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = DISABLE_AUG_LIST
                dataloader._dataset.data_augmentor.disable_augmentation(dataset_cfg.DATA_AUGMENTOR)
                flag = True
    return flag
