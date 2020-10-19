from __future__ import print_function
import numpy as np
import os
import warnings
import argparse
import sys
import random as python_random

python_random.seed(1234)

#from resnet_keras import keras_print_devices, keras_resnet50, keras_train, keras_prepare_dataset, keras_evaluate
#from resnet_pytorch import pytorch_load_data_cifar, pytorch_resnet50, pytorch_train, pytorch_validate, pytorch_env
#from resnet_eddl import  eddl_load_data_cifar, eddl_resnet50, eddl_train, eddl_validate


def main(args):
    """" this application goal is to give a wrapper for resnet50 in three different frameworks:
    1. Keras + tensorflow
    2. pytorch
    3. pyeddl

     """
    if args.framework.casefold() == "tensorflow":

        from resnet_keras import keras_print_devices, keras_resnet50, keras_train, \
            keras_prepare_dataset, keras_evaluate, keras_prepare_isic_dataset, keras_train_flow, keras_evaluate_flow

        if args.debug:
            keras_print_devices()

        # Load the dataset that we want to work on:
        if args.dataset.casefold() == "cifar10":
            (x_train, y_train), (x_test, y_test) = keras_prepare_dataset(path=args.dataset_dir, dataset="cifar10",
                                                                         debug=args.debug, load_subset=args.subset)
            nclasses = 10
        elif args.dataset.casefold() == "isic":
            train, test = keras_prepare_isic_dataset(path=args.dataset_dir, dataset=args.dataset.casefold(),
                                                     batch_size=args.bs)
            nclasses = 8
        else:
            print("Please specify a dataset name, for more info run all_reasnet50.py -h")

        # Create the resnet50 model

        model = keras_resnet50(dataset=args.dataset.casefold(), number_classes=nclasses, learning_rate=args.lr, momentum=args.momentum)

        # Train the model, note that we are running validation test in every epoch as well!
        if args.dataset.casefold() == "cifar10":
            model, history = keras_train(model, x_train, y_train, x_test, y_test, epochs=args.epochs, batch_size=args.bs,
                                         dynamic_lr=True, time2acc=args.time2accuracy)
            # Evaluate the model, note that the accuercy is the same as the last run of
            # the train as it use the same dataset
            keras_evaluate(model, x_test, y_test)
        elif args.dataset.casefold() == "isic":
            model, history = keras_train_flow(model, train, test, epochs=args.epochs, dynamic_lr=True,
                                              time2acc=args.time2accuracy)
            # Evaluate the model, note that the accuercy is the same as the last run of
            # the train as it use the same dataset
            keras_evaluate_flow(model, test)
        else:
            print("error!!")

    elif args.framework.casefold() == "pytorch":

        from resnet_pytorch import pytorch_load_data_cifar, pytorch_resnet50, pytorch_train, pytorch_validate, \
                                    pytorch_env, pytorch_load_data_isic, pytorch_load_data_imagenet

        if args.gpuid is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')
        if args.debug:
            pytorch_env()
        # Loading the dataset that we want to work on, currently we only support CIFAR10.

        if args.dataset.casefold() == "cifar10":
            train_loader, val_loader, classes = pytorch_load_data_cifar(args.dataset_dir,batch_size=args.bs,
                                                                        load_subset=args.subset)
        elif args.dataset.casefold() == "isic":
            train_loader, val_loader, classes = pytorch_load_data_isic(args.dataset_dir, batch_size=args.bs)
        elif args.dataset.casefold() == "imagenet":
            train_loader, val_loader, classes = pytorch_load_data_imagenet(args.dataset_dir, batch_size=args.bs)
        else:
            print("Please specify a dataset name, for more info run all_reasnet50.py -h")

        # Create the resnet50 model
        model, criterion, optimizer = pytorch_resnet50(gpu=args.gpu, gpuid=args.gpuid, learning_rate=args.lr,
                                                       momentum=args.momentum, weight_decay=args.weight_decay,
                                                       debug=args.debug, dataset=args.dataset.casefold())

        pytorch_train(model, criterion, optimizer, train_loader, val_loader, learning_rate=args.lr,
                      gpu=args.gpu, gpuid=args.gpuid, epochs=args.epochs, dynamic_lr=True, time2acc=args.time2accuracy)

        top1_avg = pytorch_validate(val_loader, model, criterion, gpu=args.gpu, gpuid=args.gpuid)

        print("Pytorch-Resnet: {} accuracy of top1 on validation set".format(top1_avg))

    elif args.framework.casefold() == "pyeddl":

        from resnet_eddl import eddl_load_data_cifar, eddl_load_data_isic, eddl_resnet50, \
            eddl_train, eddl_train_DLDataset, eddl_validate, eddl_validate_DLDataset

        if args.dataset.casefold() == "cifar10":
            (x_train, y_train), (x_test, y_test) = eddl_load_data_cifar(args.dataset_dir, load_subset=args.subset)
        elif args.dataset.casefold() == "isic":
            isic_dataset = eddl_load_data_isic(args.dataset_dir, args.bs, args.debug)
        elif args.dataset.casefold() == "imagenet":
            imagenet_dataset = eddl_load_data_imagenet(args.dataset_dir, args.bs, args.debug)
        else:
            print("Please specify a dataset name, for more info run all_reasnet50.py -h")

        # Create the resnet50 model
        model, out = eddl_resnet50(args.dataset, learning_rate=args.lr, momentum=args.momentum,
                                                       gpu=args.gpu, debug=args.debug)

        if args.dataset.casefold() == "cifar10":
            model = eddl_train(model, x_train, y_train, x_test, y_test, learning_rate=args.lr, momentum=args.momentum,
                               epochs=args.epochs, batch_size=args.bs, dynamic_lr=True)
            print("Validation phase:")
            eddl_validate(model, x_test, y_test)
        elif args.dataset.casefold() == "isic":
            model = eddl_train_DLDataset(model, out, isic_dataset, epochs=args.epochs, dynamic_lr=True)
            print("Validation phase:")
            eddl_validate_DLDataset(model, out, isic_dataset)
        elif args.dataset.casefold() == "imagenet":
            model = eddl_train_DLDataset(model, out, imagenet_dataset, epochs=args.epochs, dynamic_lr=True)
            print("Validation phase:")
            eddl_validate_DLDataset(model, out, imagenet_dataset)


        print("done!!!")
    else:
        print("The current support frameworks are: TensorFlow and pytorch, \n pyeddl will be support shortly!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cifer10 with different frameworks and platform using "
                                                 "Resnet50 topology.")
    parser.add_argument("--framework", choices=["tensorflow", "pytorch", "pyeddl"] , required=True)
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--bs", "--batch-size", type=int, metavar="INT", default=32)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--subset", action="store_true", help='Choosing smaller training dataset, default=4k')
    parser.add_argument("--sb_size", "--subset_size", type=int, metavar="INT", default=4096, dest='subset')
    parser.add_argument("--lr", '--learning_rate', default=0.1, type=float, help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum', dest='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--model_dir', type=str, default='/tmp/resnet_model',
                        help='The directory where the model will be stored.')
    parser.add_argument("--time2accuracy", action="store_true", help="stop trainning when top-1 accuracy reach "
                                                                     "to 74.9% and report total run time")
    parser.add_argument("--dataset", choices=["ImageNet", "cifar10", "isic"],
                        help="choose your dateset, not yet implemented", default="cifar10")
    parser.add_argument('--dataset_dir', type=str, default="/home/abadouh/.keras/datasets/cifar-10-batches-py/",
                        help='The directory of the dataset.')
    parser.add_argument('--gpuid', default=None, type=int, help='GPU id to use.')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    main(parser.parse_args(sys.argv[1:]))
