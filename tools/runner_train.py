import time
import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
# from datasets import data_transforms

# train_transforms = transforms.Compose(
#     [
#         # data_transforms.PointcloudScale(),
#         # data_transforms.PointcloudRotate(),
#         # data_transforms.PointcloudRotatePerturbation(),
#         # data_transforms.PointcloudTranslate(),
#         # data_transforms.PointcloudJitter(),
#         # data_transforms.PointcloudRandomInputDropout(),
#         data_transforms.PointcloudScaleAndTranslate(),
#     ]
# )

class Metric:
    def __init__(self, metric):
        self.type_acc = metric['type_acc']
        self.state_error = metric['state_error']
        self.ori_error = metric['ori_error']
        self.pos_error = metric['pos_error']

    def better_than(self, other):
        if self.state_error < other.state_error:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['type_acc'] = self.type_acc
        _dict['state_error'] = self.state_error
        _dict['ori_error'] = self.ori_error
        _dict['pos_error'] = self.pos_error
        return _dict


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.test)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    start_metric = {'type_acc':0., 'state_error': 99., 'ori_error':99., 'pos_error':99.}
    best_metrics = Metric(start_metric)
    metrics = Metric(start_metric)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss', 'Type_acc', 'State_error', 'Ori_error', 'Pos_error'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (img, points, label) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)

            dataset_name = config.dataset.train._base_.NAME
            img = img.cuda()
            points = points.cuda()

            # points = train_transforms(points)
            loss, type_acc, state_error, ori_error, pos_error = base_model(img, points, label)
            try:
                loss.backward()
                # print("Using one GPU")
            except:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()*1000])
            else:
                type_acc = type_acc.mean()
                state_error = state_error.mean()
                ori_error = ori_error.mean()
                pos_error = pos_error.mean()
                losses.update([loss.item()*1000, type_acc*100, state_error.item(), ori_error.item(), pos_error.item()])


            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)

        if (epoch+1) % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                print_log('[Validation] Best--------------------------------------------------------------------------', logger = logger)
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 25 ==0 and epoch >=250:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    type_acc_list, state_error_list, ori_error_list, pos_error_list = [],[],[],[]
    with torch.no_grad():
        for idx, (img, points, label) in enumerate(test_dataloader):
            img = img.cuda()
            points = points.cuda()
            loss, type_acc, state_error, ori_error, pos_error = base_model(img, points, label)
            type_acc_list.append(type_acc)
            state_error_list.append(state_error)
            ori_error_list.append(ori_error)
            pos_error_list.append(pos_error)

        type_acc_mean = sum(type_acc_list) / len(type_acc_list)
        state_error_mean = sum(state_error_list) / len(state_error_list)
        ori_error_mean = sum(ori_error_list) / len(ori_error_list)
        pos_error_mean = sum(pos_error_list) / len(pos_error_list)
        print_log('[Validation] EPOCH: %d  type_acc = %.2f  state_error = %.2f, ori_error = %.2f, pos_error = %.2f'
                  % (epoch, type_acc_mean*100, state_error_mean, ori_error_mean, pos_error_mean), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', type_acc, epoch)
        val_writer.add_scalar('Metric/STATE', state_error, epoch)
        val_writer.add_scalar('Metric/ORI', ori_error, epoch)
        val_writer.add_scalar('Metric/POS', pos_error, epoch)

    metric = {'type_acc':type_acc_mean, 'state_error': state_error_mean, 'ori_error':ori_error_mean, 'pos_error':pos_error_mean}
    return Metric(metric)

def test_net():
    pass