import os
import glob
import numpy as np
from collections import Counter
from models import load_model
from keras import backend as K
from keras.callbacks import (
    LearningRateScheduler, TensorBoard, ModelCheckpoint, CSVLogger)
from keras.datasets import cifar10, cifar100, mnist
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from argparse import ArgumentParser

config = K.tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = K.tf.Session(config=config)

# Read options
parser = ArgumentParser()
parser.add_argument('--savepath', default='results')
parser.add_argument('--dataset', default="dataset", help="dataset path")
parser.add_argument('--img_dim', default=224, help="image dimension")
parser.add_argument('--net_type', default='resnet50imagenet')
parser.add_argument('--depth', type=int, default=16)
parser.add_argument('--widen', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--unfreeze_step', type=int, default=5)
parser.add_argument('--generator_n_fit', type=int, default=250)
parser.add_argument('--randomcrop', type=int, default=4)
parser.add_argument('--randomcrop_type', default="reflect", help="zero, reflect")
parser.add_argument('--hflip', action='store_false', default=True)
parser.add_argument('--epoch_max', type=int, default=50)
parser.add_argument('--epoch_init', type=int, default=0)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--nthreads', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_decay', type=float, default=0.2)
parser.add_argument('--lr_schedule', nargs='+', default=[15,30,40], type=int)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', action='store_false', default=True)
opts = parser.parse_args()
print(opts)

# Get data for generator fit
imh, imw, imc = opts.img_dim, opts.img_dim, 3
n_classes = len(os.walk(os.path.join(opts.dataset, 'train')).__next__()[1])
train_img_paths = list(glob.iglob(os.path.join(opts.dataset, 'train', '**', '*.jpg'), recursive=True))
data = []
for i in range(opts.generator_n_fit):
    new_data = load_img(train_img_paths[i], target_size=(imh, imw, imc))
    new_data = np.array(new_data).astype(np.float32)
    new_data = np.expand_dims(new_data, 0)
    data.append(new_data)
data = np.concatenate(data, 0)

# Data generator
trdatagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=opts.randomcrop/imw,
    height_shift_range=opts.randomcrop/imh,
    fill_mode=opts.randomcrop_type,
    cval=0,
    horizontal_flip=opts.hflip)
trdatagen.fit(data)
print('Mean and std of trdatagen:')
print(trdatagen.mean)
print(trdatagen.std)
trgenerator = trdatagen.flow_from_directory(
    os.path.join(opts.dataset, 'train'),
    target_size=(imh, imw),
    batch_size=opts.bs)

tstdatagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)
tstdatagen.fit(data)
tstgenerator = tstdatagen.flow_from_directory(
    os.path.join(opts.dataset, 'test'),
    target_size=(imh, imw),
    batch_size=opts.bs)

# Instanciate model
model = load_model(
    net_type=opts.net_type,
    input_shape=(imh, imw, imc),
    n_classes=n_classes,
    depth=opts.depth,
    weight_decay=opts.weight_decay,
    widen=opts.widen)

optimizer = SGD(lr=opts.lr, momentum=opts.momentum, nesterov=opts.nesterov)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=['accuracy'])

# Compute class weights
def compute_class_weights(y, smooth_factor=0):
    counter = Counter(y)
    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}

class_weight = compute_class_weights(trgenerator.classes)

# Define callbacks
class Unfreeze(Callback):
    def __init__(self, epoch_step=2):
        self.idx = -1
        self.epoch_step = epoch_step

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_step == 0:
            self.idx = self.idx - 1
            if self.idx < -len(self.model.layers):
                self.idx = -len(self.model.layers)
            self.model.layers[self.idx].trainable = True
            self.model.compile(
                loss="categorical_crossentropy",
                optimizer=self.model.optimizer,
                metrics=['accuracy'])

def lrs_callback(epoch):
    return opts.lr * opts.lr_decay**(np.array(opts.lr_schedule) <= epoch).sum()

learning_rate_scheduler = LearningRateScheduler(lrs_callback)
tensorboard = TensorBoard(opts.savepath)
checkpoint = ModelCheckpoint(
    os.path.join(opts.savepath, "model.hdf5"),
    monitor="val_acc",
    save_best_only=True,
    mode="max")
logger = CSVLogger(os.path.join(opts.savepath, "results.log"), append=True)
unfreeze = Unfreeze(opts.unfreeze_step)
callbacks = [learning_rate_scheduler, tensorboard, checkpoint, logger, unfreeze]

# Train model
model.fit_generator(
    generator=trgenerator,
    samples_per_epoch=trgenerator.n,
    nb_epoch=opts.epoch_max,
    initial_epoch=opts.epoch_init,
    class_weight=class_weight,
    nb_worker=opts.nthreads,
    pickle_safe=True,
    callbacks=callbacks,
    validation_data=tstgenerator,
    nb_val_samples=tstgenerator.n
)

model.save(os.path.join(opts.savepath, "model_last.hdf5"))
