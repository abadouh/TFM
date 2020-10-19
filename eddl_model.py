
"""\
ResNet Module for pyEDDL library
pyEDDL 0.8.0
pyECVL 0.4.2
"""

import pyeddl._core.eddl as eddl
from pyeddl.tensor import Tensor

def BatchNorm2D(layer):
    return eddl.BatchNormalization(layer, True)


def conv1x1(layer, filters, strides=[1, 1]):
    return eddl.GlorotNormal(eddl.Conv(layer, filters, [1, 1], strides, "same", use_bias=False))

def conv3x3(layer, filters, strides=[1,1]):
    return eddl.GlorotNormal(eddl.Conv(layer, filters, [3, 3], strides, "same", use_bias=True))

def Bottleneck(layer, in_filters, out_filters, stride):
    expansion = 4

    identity = layer

    layer = conv1x1(layer, out_filters)
    layer = BatchNorm2D(layer)
    layer = eddl.ReLu(layer)

    layer = conv3x3(layer, out_filters, stride)
    layer = BatchNorm2D(layer)
    layer = eddl.ReLu(layer)

    layer = conv1x1(layer, expansion * out_filters)
    layer = BatchNorm2D(layer)

    if stride != [1, 1] or in_filters != expansion * out_filters:
        identity = BatchNorm2D(conv1x1(identity, 4*out_filters, stride))

    layer = eddl.Sum(layer, identity)
    layer = eddl.ReLu(layer)

    return layer


def make_layer(layer, in_planes, planes, num_blocks, strides):
    list_strides = [strides] + [[1, 1]]*(num_blocks - 1)

    for stride in list_strides:
        layer = Bottleneck(layer, in_planes, planes, stride)
        in_planes = 4 * planes
    return layer


def ResNetBody(layer, filters, in_planes, units):

    for i in range(len(units)):
        stride = [1, 1] if 0 == i else [2, 2] # the first conv layer should have stride [1,1] in conv3x3 and
                                              # in the downsample conv
        layer = make_layer(layer, in_planes[i], filters[i], num_blocks=units[i], strides=stride)
    return layer


def Resnet50(shape, number_classes=10, learning_rate=1e-2,momentum=0.9, gpu=False, debug=False):
    
    in_ = eddl.Input(shape)
    layer = in_

    # Part of the augmentation needed to rescale the image, relevant for bigger input size.
    # layer = eddl.RandomCropScale(layer, [0.8, 1.0])
    # layer = eddl.Flip(layer, 1)
    if shape == [3, 224, 224]:
        # ISIC and ImageNet
        layer = eddl.ReLu(BatchNorm2D(eddl.GlorotNormal(eddl.Conv(layer, 64, [7, 7], [2, 2], "same", use_bias=True))))# 7x7 with stride 2
    else: # == [3, 32, 32]:
        # Cifar
        layer = eddl.ReLu(BatchNorm2D(eddl.GlorotNormal(eddl.Conv(layer, 64, [3, 3], [1, 1], "same", use_bias=True))))

    # v3
    layer = eddl.MaxPool(layer, [3, 3], [2, 2], "same")

    layer = ResNetBody(layer, [64, 128, 256, 512], [64, 256, 512, 1024, 2048], [3, 4, 6, 3])

    # v3
    # if gpu:
    #layer = eddl.MaxPool(layer, [4, 4])
    layer = eddl.GlobalAveragePool(layer, name="avg_pool")

    layer = eddl.Reshape(layer, [-1])

    out = eddl.Activation(eddl.Dense(layer, number_classes), "softmax")
    net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.sgd(learning_rate, momentum),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(g=[1], mem="low_mem") if gpu else eddl.CS_CPU()
        #eddl.CS_GPU(g=[1,1], mem="low_mem") if gpu else eddl.CS_CPU()
    )
    
    if debug:
        eddl.summary(net)
        eddl.plot(net, "model.pdf", "TB")

    return net, out


def load_cifar(path=None, download=False):

    if path is None:
        path = "./"

    if download:
        eddl.download_cifar10()
    print(path)
    try:
        x_train = Tensor.load(path+"/cifar_trX.bin")
        y_train = Tensor.load(path+"/cifar_trY.bin")
    except:
        print("Fail to load the train set, make sure you supply the correct path")
        exit()

    try:
        x_test = Tensor.load(path+"/cifar_tsX.bin")
        y_test = Tensor.load(path+"/cifar_tsY.bin")
    except:
        print("Fail to load the test set, make sure you supply the correct path")
        exit()

    x_train.div_(255.0)
    x_test.div_(255.0)

    return (x_train, y_train), (x_test, y_test)


#def BasicBlock(layer, filters, strides=[1,1], downsample=None):
#    """ Draft! need to check it!!!!
#
#        Basic block for resnet18 and resnet34
#
#    """
#    identity = layer
#
#    layer = conv3x3(layer, filters, strides)
#    layer = BatchNorm2D(layer)
#    layer = eddl.ReLu(layer)
#    layer = conv3x3(layer, filters)
#    layer = BatchNorm2D(layer)
#    if downsample is not None:
#        identity = downsample(identity)
#
#    layer = eddl.Sum(layer, identity)
#
#    return layer
#

""""
resnet_spec = {18: (ResBottleNeckBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: (ResBottleNeckBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: (ResBottleNeckBlock, [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: (ResBottleNeckBlock, [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: (ResBottleNeckBlock, [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

"""
