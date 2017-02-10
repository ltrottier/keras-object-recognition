# keras-object-recognition
Minimalist Keras implementation for deep learning object recognition.

## Installation

First, install the requirements:
```bash
pip install -r requirements.txt
```

Make sure keras uses tensorflow backend. Edit `~/.keras/keras.json` like this:
```json
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_dim_ordering": "tf"
}
```

## Training

Train a model with:
```bash
python train.py
```

Default options (see `train.py` for the available options):

1. `--savepath results`
2. `--dataset cifar10`
3. `--net_type resnet`
4. `--depth 16`
5. `--widen 1`
6. `--weight_decay 5e-4`
7. `--randomcrop 4`
8. `--randomcrop_type reflect`
9. `--hflip` (pass to remove hflip)
10. `--epoch_max 200`
11. `--epoch_init 0`
12. `--bs 128`
13. `--nthreads 2`
14. `--lr 0.1`
15. `--lr_decay 0.2`
16. `--lr_schedule 60 120 160`
17. `--momentum 0.9`
18. `--nesterov` (pass to remove nesterov)
