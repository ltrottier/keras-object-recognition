import numpy as np
from models import load_model
from keras import backend as K
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.datasets import cifar10, cifar100, mnist
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from optparse import OptionParser

# Read options
parser = OptionParser()
parser.add_option('--savepath', default='results')
parser.add_option('--dataset', default="cifar10", help="cifar10, cifar100")
parser.add_option('--net_type', default='resnet')
parser.add_option('--depth', type=int, default=16)
parser.add_option('--widen', type=int, default=1)
parser.add_option('--weight_decay', type=float, default=1e-4)
parser.add_option('--randomcrop', type=int, default=4)
parser.add_option('--randomcrop_type', default="reflect", help="zero, reflect")
parser.add_option('--hflip', action='store_true', default=True)
parser.add_option('--epoch_max', type=int, default=200)
parser.add_option('--epoch_init', type=int, default=0)
parser.add_option('--bs', type=int, default=128)
parser.add_option('--nthreads', type=int, default=2)
parser.add_option('--lr', type=float, default=0.1)
parser.add_option('--lr_decay', type=float, default=0.2)
parser.add_option('--lr_schedule', default=[60,120,160])
parser.add_option('--momentum', type=float, default=0.9)
parser.add_option('--nesterov', action='store_true', default=True)
(opts, args) = parser.parse_args()
print(opts)

# Load data
(xtr, ytr), (xtst, ytst) = eval(opts.dataset).load_data()
xtr = xtr.astype('float32')
ytr = to_categorical(ytr)
xtst = xtst.astype('float32')
ytst = to_categorical(ytst)
trsize, imh, imw, imc = xtr.shape
tstsize = xtst.shape[1]
n_classes = ytr.shape[1]

# Data generator
trdatagen = ImageDataGenerator(featurewise_center=True,
                               featurewise_std_normalization=True,
                               width_shift_range=opts.randomcrop/imw,
                               height_shift_range=opts.randomcrop/imh,
                               fill_mode=opts.randomcrop_type,
                               cval=0,
                               horizontal_flip=opts.hflip)
trdatagen.fit(xtr)
trgenerator = trdatagen.flow(xtr, ytr, batch_size=opts.bs)

tstdatagen = ImageDataGenerator(featurewise_center=True,
                                featurewise_std_normalization=True)
tstdatagen.fit(xtr)
tstgenerator = tstdatagen.flow(xtst, ytst, batch_size=opts.bs)

# Instanciate model
model = load_model(net_type=opts.net_type,
                   input_shape=(imh, imw, imc),
                   n_classes=n_classes,
                   depth=opts.depth,
                   weight_decay=opts.weight_decay,
                   widen=opts.widen)

optimizer = SGD(lr=opts.lr, momentum=opts.momentum, nesterov=opts.nesterov)

model.compile(loss="categorical_crossentropy",
              optimizer=optimizer,
              metrics=['accuracy'])

# Train model
def lrs_callback(epoch):
    return opts.lr * opts.lr_decay**(np.array(opts.lr_schedule) <= epoch).sum()

learning_rate_scheduler = LearningRateScheduler(lrs_callback)
tensorboard = TensorBoard(opts.savepath)
callbacks = [learning_rate_scheduler, tensorboard]

model.fit_generator(generator=trgenerator,
                    samples_per_epoch=trsize,
                    nb_epoch=opts.epoch_max,
                    initial_epoch=opts.epoch_init,
                    nb_worker=opts.nthreads,
                    pickle_safe=True,
                    callbacks=callbacks,
                    validation_data=tstgenerator,
                    nb_val_samples=tstsize)
