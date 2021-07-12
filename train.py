import os
import torch
import logging
import importlib
import shutil
from utils import provider
import numpy as np

from tqdm import tqdm
from utils.dataset import PartNormalDataset
import hydra
import yaml
from utils.general import to_categorical
from utils.general import seg_classes, seg_label_to_cat

def create_attr_dict(yaml_config):
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def main():

    with open('config/partseg.yaml', 'r') as f:
        args = AttrDict(yaml.safe_load(f.read()))
    create_attr_dict(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    work_dir = args.work_dir
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    log_file = os.path.join(work_dir, 'log.log')
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    root = hydra.utils.to_absolute_path(
        'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/')

    TRAIN_DATASET = PartNormalDataset(
        root=root, npoints=args.num_point, split='trainval', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(
        root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    args.input_dim = (6 if args.normal else 3) + 16
    args.num_class = 50
    num_category = 16
    num_part = args.num_class
    shutil.copy(hydra.utils.to_absolute_path('models/model.py'), '.')

    model = getattr(importlib.import_module('models.model'),
                    'PointTransformerSeg')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        logger.info('Epoch %d (%d/%s):' %
                    (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay **
                                       (epoch // args.step_size)), LEARNING_RATE_CLIP)
        logger.info('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * \
            (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
        model = model.train()

        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(
                points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)

            points, label, target = points.float().cuda(
            ), label.long().cuda(), target.long().cuda()
            optimizer.zero_grad()

            seg_pred = model(torch.cat([points, to_categorical(
                label, num_category).repeat(1, points.shape[1], 1)], -1))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(
                correct.item() / (args.batch_size * args.num_point))
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train accuracy is: %.5f' % train_instance_acc)

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            model = model.eval()

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(
                ), label.long().cuda(), target.long().cuda()
                seg_pred = model(torch.cat([points, to_categorical(
                    label, num_category).repeat(1, points.shape[1], 1)], -1))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros(
                    (cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(
                        logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (
                        np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            for cat in sorted(shape_ious.keys()):
                logger.info('eval mIoU of %s %f' %
                            (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        logger.info('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = os.path.join(work_dir, 'best_model.pth')
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            logger.info('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        logger.info('Best accuracy is: %.5f' % best_acc)
        logger.info('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        logger.info('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1

    shutil.copy(savepath, './best_model.pth')

if __name__ == '__main__':
    main()
