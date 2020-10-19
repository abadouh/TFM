
"""\
ResNet Module for pyEDDL library
pyEDDL 0.8.0
pyECVL 0.4.2
"""

import math
import time
import eddl_model as resnet
import pyecvl.ecvl as ecvl
import pyeddl._core.eddl as eddl
from pyeddl.tensor import Tensor

# import pyeddl._core.eddlT as eddlT
import random



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def eddl_load_data_cifar(path=None, download=False, load_subset=False, subset_size=4096):
    """
    :param path: Path to CIFAR10 dataset
    :param download: Default-False - download the dataset to current directory.
    :param load_subset: Default-False - use subset of the dataset for testing purpose.
    :param subset_size: Default-4096, the test size is subset_size/4.
    :return: two tuples of (x_train, y_train), (x_test, y_test) .
    """
    (x_train, y_train), (x_test, y_test) = resnet.load_cifar(path, download)

    if load_subset:
        x_subset_train = Tensor([subset_size, 3, 32, 32])
        y_subset_train = Tensor([subset_size, 10])

        eddl.next_batch([x_train], [x_subset_train])
        eddl.next_batch([y_train], [y_subset_train])

        x_subset_test = Tensor([math.ceil(subset_size/4), 3, 32, 32])
        y_subset_test = Tensor([math.ceil(subset_size/4), 10])

        eddl.next_batch([x_test], [x_subset_test])
        eddl.next_batch([y_test], [y_subset_test])
        return (x_subset_train, y_subset_train), (x_subset_test, y_subset_test)

    return (x_train, y_train), (x_test, y_test)


def eddl_load_data_isic(path=None, batch_size = 32, debug=True):
    """

    :param path: Path to YAML file
    :param batch_size: default 32
    :param debug: default True
    :return: ecvl.DLDataset, x, y
    """
    size = [224, 224]

    training_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size)
    ])

    """
    ecvl.AugMirror(.5),
    ecvl.AugFlip(.5),
    ecvl.AugRotate([-180, 180]),
    ecvl.AugAdditivePoissonNoise([0, 10]),
    ecvl.AugGammaContrast([0.5, 1.5]),
    ecvl.AugGaussianBlur([0, 0.8]),
    ecvl.AugCoarseDropout([0, 0.3], [0.02, 0.05], 0.5)
    """

    if debug:
        print("Applying the following augmentation on train set: resizeDim {}".format(size))

    validation_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size)
    ])

    if debug:
        print("Applying the following augmentation on validation set: resizeDim {}".format(size))

    dataset_augs = ecvl.DatasetAugmentations(
        [training_augs, validation_augs, None]
    )

    if debug:
        print("Reading dataset")
    d = ecvl.DLDataset(path, batch_size, dataset_augs)
    #x = eddlT.create([batch_size, d.n_channels_, size[0], size[1]])
    #y = eddlT.create([batch_size, len(d.classes_)])

    if debug:
        num_samples_train = len(d.GetSplit())
        num_batches_train = num_samples_train // batch_size
        d.SetSplit(ecvl.SplitType.validation)
        num_samples_val = len(d.GetSplit())
        num_batches_val = num_samples_val // batch_size

        print("User batch size: {} ".format(batch_size))
        print("Total number of train samples: {} , in {} batches".format(num_samples_train, num_batches_train))
        print("Total number of validation samples: {} , in {} batches".format(num_samples_val, num_batches_val))

    return d #, x, y


def eddl_load_data_imagenet(path=None, batch_size = 32, debug=True):
    """

    :param path: Path to YAML file
    :param batch_size: default 32
    :param debug: default True
    :return: ecvl.DLDataset
    """
    size = [224, 224]

    training_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size)
    ])

    """
    ecvl.AugMirror(.5),
    ecvl.AugFlip(.5),
    ecvl.AugRotate([-180, 180]),
    ecvl.AugAdditivePoissonNoise([0, 10]),
    ecvl.AugGammaContrast([0.5, 1.5]),
    ecvl.AugGaussianBlur([0, 0.8]),
    ecvl.AugCoarseDropout([0, 0.3], [0.02, 0.05], 0.5)
    """

    if debug:
        print("Applying the following augmentation on train set: resizeDim {}".format(size))

    validation_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(size)
    ])

    if debug:
        print("Applying the following augmentation on validation set: resizeDim {}".format(size))

    dataset_augs = ecvl.DatasetAugmentations(
        [training_augs, validation_augs, None]
    )

    if debug:
        print("Reading dataset")
    d = ecvl.DLDataset(path, batch_size, dataset_augs)

    if debug:
        num_samples_train = len(d.GetSplit())
        num_batches_train = num_samples_train // batch_size
        d.SetSplit(ecvl.SplitType.validation)
        num_samples_val = len(d.GetSplit())
        num_batches_val = num_samples_val // batch_size

        print("User batch size: {} ".format(batch_size))
        print("Total number of train samples: {} , in {} batches".format(num_samples_train, num_batches_train))
        print("Total number of validation samples: {} , in {} batches".format(num_samples_val, num_batches_val))

    return d


# Create the resnet50 model
def eddl_resnet50(dataset="cifar10", learning_rate=1e-2, momentum=0.9, gpu=False, debug=False):

    if dataset.casefold() == "cifar10":
        shape = [3, 32, 32]
        number_classes = 10
    elif dataset.casefold() == "isic":
        shape = [3, 224, 224]
        number_classes = 8
    elif dataset.casefold() == "imagenet":
        shape = [3, 224, 224]
        number_classes = 1000
    else:
        print("Currently we support only ImageNet, CIFAR10 and ISIC datasets, ")

    model, out = resnet.Resnet50(shape, number_classes, learning_rate, momentum, gpu, debug)
    return model, out


def eddl_train(model, x_train, y_train, x_test, y_test, learning_rate=1e-2, momentum=0.9,
               epochs=10, batch_size=128, dynamic_lr=False):

    print("start to train model")

    batch_time = AverageMeter('BatchTime', ':6.3f')
    total_time = AverageMeter('TotalTime', ':6.3f')
    end_total = time.time()
    for i in range(epochs):

        if dynamic_lr and (i % 30 == 0) and i > 0:

            # every 30 epochs we want to decrease the learning rate value by 0.1
            print("every 30 epochs we want to decrease the learning rate value by 0.1")
            learning_rate = learning_rate * 0.1
            eddl.setlr(model, [learning_rate, momentum])
        batch_time.reset()
        end = time.time()
        print("fit model epoch: {}".format(i), flush=True)
        eddl.fit(model, [x_train], [y_train], batch_size, 1)

        print("validate model epoch: {}".format(i), flush=True)
        eddl.evaluate(model, [x_test], [y_test])
        batch_time.update(time.time() - end)
        print(batch_time, flush=True)
    total_time.update(time.time() - end_total)
    print(total_time, flush=True)
    return model


def eddl_train_DLDataset(model, out, d, learning_rate=1e-2, momentum=0.9, epochs=10, dynamic_lr=False):

    batch_time = AverageMeter('BatchTime', ':6.3f')
    total_time = AverageMeter('TotalTime', ':6.3f')

    # Use the image resized dims defined by user or default image size for resnet [224,224]
    if hasattr(d, 'resize_dims_'):
        size = d.resize_dims_
    else:
        size = [224, 224]

    x = Tensor([d.batch_size_, d.n_channels_, size[0], size[1]])
    y = Tensor([d.batch_size_, len(d.classes_)])
    d.SetSplit(ecvl.SplitType.training)
    num_samples_train = len(d.GetSplit())
    num_batches_train = num_samples_train // d.batch_size_

    d.SetSplit(ecvl.SplitType.validation)
    num_samples_val = len(d.GetSplit())
    num_batches_val = num_samples_val // d.batch_size_

    indices = list(range(d.batch_size_))
    metric = eddl.getMetric("categorical_accuracy")

    print("Starting training", flush=True)
    end_total = time.time()
    for e in range(epochs):
        if dynamic_lr and (e % 30 == 0) and e > 0:
            # every 30 epochs we want to decrease the learning rate value by 0.1
            print("every 30 epochs we want to decrease the learning rate value by 0.1")
            learning_rate = learning_rate * 0.1
            eddl.setlr(model, [learning_rate, momentum])    

        print("Epoch {:d}/{:d} - Training".format(e + 1, epochs), flush=True)

        d.SetSplit(ecvl.SplitType.training)
        eddl.reset_loss(model)
        total_metric = []
        s = d.GetSplit()
        random.shuffle(s)
        d.split_.training_ = s
        d.ResetAllBatches()
        batch_time.reset()
        end = time.time()
        for b in range(num_batches_train):
            print("Epoch {:d}/{:d} (batch {:d}/{:d}) - ".format(e + 1, epochs, b + 1, num_batches_train),
                  end="", flush=True)
            d.LoadBatch(x, y)
            x.div_(255.0)
            tx, ty = [x], [y]
            eddl.train_batch(model, tx, ty, indices)
            eddl.print_loss(model, b)
            batch_time.update(time.time() - end)
            end = time.time()
            print(batch_time, flush=True)
            print()

        #print("Saving weights")
        #eddl.save(
        #    net, "isic_classification_checkpoint_epoch_%s.bin" % e, "bin"
        #)

        print("Epoch %d/%d - Evaluation (validation set)" % (e + 1, epochs), flush=True)
        d.SetSplit(ecvl.SplitType.validation)
        batch_time.reset()
        end = time.time()
        for b in range(num_batches_val):
            n = 0
            print("Epoch {:d}/{:d} (batch {:d}/{:d}) - ".format(e + 1, epochs, b + 1, num_batches_val),
                  end="", flush=True)
            d.LoadBatch(x, y)
            x.div_(255.0)
            eddl.forward(model, [x])
            output = eddl.getOutput(out)
            sum_ = 0.0
            for k in range(d.batch_size_):
                result = output.select([str(k)])
                target = y.select([str(k)])
                ca = metric.value(target, result)
                total_metric.append(ca)
                sum_ += ca
                """
                if args.out_dir:
                    result_a = np.array(result, copy=False)
                    target_a = np.array(target, copy=False)
                    classe = np.argmax(result_a).item()
                    gt_class = np.argmax(target_a).item()
                    single_image = eddlT.select(x, k)
                    img_t = ecvl.TensorToView(single_image)
                    img_t.colortype_ = ecvl.ColorType.BGR
                    single_image.mult_(255.)
                    filename = d.samples_[d.GetSplit()[n]].location_[0]
                    head, tail = os.path.splitext(os.path.basename(filename))
                    bname = "%s_gt_class_%s.png" % (head, gt_class)
                    cur_path = os.path.join(
                        current_path, d.classes_[classe], bname
                    )
                    ecvl.ImWrite(cur_path, img_t)
                """
                n += 1
            batch_time.update(time.time() - end)
            end = time.time()
            print("batch categorical accuracy:{}".format(sum_ / d.batch_size_), flush=True)
            print(batch_time, flush=True)
        if(num_batches_val > 0):
            total_avg = sum(total_metric) / len(total_metric)
            print("Total categorical accuracy:{}".format(total_avg), flush=True)
        else:
            print("Warning! \n "
                  "Please check your validation set size, it might be smaller than the batch size,\n "
                  "Validation test didn't execute as batch number is 0")

    total_time.update(time.time() - end_total)
    print(total_time, flush=True)
    return model


def eddl_validate(model, x_test, y_test):
    eddl.evaluate(model, [x_test], [y_test])


def eddl_validate_DLDataset(model, out, d):
    batch_time = AverageMeter('BatchTime', ':6.3f')
    total_time = AverageMeter('TotalTime', ':6.3f')

    # Use the image resized dims defined by user or default image size for resnet [224,224]
    if hasattr(d, 'resize_dims_'):
        size = d.resize_dims_
    else:
        size = [224, 224]

    x = Tensor([d.batch_size_, d.n_channels_, size[0], size[1]])
    y = Tensor([d.batch_size_, len(d.classes_)])

    d.SetSplit(ecvl.SplitType.validation)
    num_samples_val = len(d.GetSplit())
    num_batches_val = num_samples_val // d.batch_size_

    indices = list(range(d.batch_size_))
    metric = eddl.getMetric("categorical_accuracy")

    print("Start Evaluation: ",flush=True)
    total_metric = []
    print("Evaluation (validation set)", flush=True)
    d.ResetAllBatches()
    d.SetSplit(ecvl.SplitType.validation)
    end_total = time.time()
    batch_time.reset()
    end = time.time()
    for b in range(num_batches_val):
        n = 0
        print("(batch {:d}/{:d}) - ".format(b + 1, num_batches_val),
              end="", flush=True)
        d.LoadBatch(x, y)
        x.div_(255.0)
        eddl.forward(model, [x])
        output = eddl.getOutput(out)
        sum_ = 0.0
        for k in range(d.batch_size_):
            result = output.select([str(k)])
            target = y.select([str(k)])
            ca = metric.value(target, result)
            total_metric.append(ca)
            sum_ += ca
            """
            if args.out_dir:
                result_a = np.array(result, copy=False)
                target_a = np.array(target, copy=False)
                classe = np.argmax(result_a).item()
                gt_class = np.argmax(target_a).item()
                single_image = eddlT.select(x, k)
                img_t = ecvl.TensorToView(single_image)
                img_t.colortype_ = ecvl.ColorType.BGR
                single_image.mult_(255.)
                filename = d.samples_[d.GetSplit()[n]].location_[0]
                head, tail = os.path.splitext(os.path.basename(filename))
                bname = "%s_gt_class_%s.png" % (head, gt_class)
                cur_path = os.path.join(
                    current_path, d.classes_[classe], bname
                )
                ecvl.ImWrite(cur_path, img_t)
            """
            n += 1
        batch_time.update(time.time() - end)
        end = time.time()
        print("categorical_accuracy:".format( sum_ / d.batch_size_), flush=True)
        print(batch_time)
    if(num_batches_val > 0):
        total_avg = sum(total_metric) / len(total_metric)
        print("Total categorical accuracy:{}".format(total_avg), flush=True)
    else:
         print("Warning! \n "
               "Please check your validation set size, it might be smaller than the batch size,\n "
               "Validation test didn't execute as batch number is 0")

    #total_avg = sum(total_metric) / len(total_metric)
    #print("Total categorical accuracy:{}".format(total_avg), flush=True)

    total_time.update(time.time() - end_total)
    print(total_time, flush=True)
    return model


