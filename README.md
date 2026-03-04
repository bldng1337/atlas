## Atlas
Training and Dataset code for controlling [MESA](https://huggingface.co/NewtNewt/MESA).

Training scripts are at [train_controlnet.py](train_controlnet.py), [train_controlnet_distill.py](train_controlnet_distill.py), and [train_t2i.py](train_t2i.py) respectively although the t2i script is currently outdated in relation to the controlnet scripts.
Dataset scripts are in the dataset folder with the perhaps most interesting being [feature_map.py](dataset/feature_map.py) which is used to create the feature maps that are used as conditioning for the controlnet models. The rest are scripts to download and merge [Major-Tom](https://huggingface.co/Major-TOM) into one dataset for training.
Some tests are at [test.py](test.py) and [overfit_test.py](overfit_test.py).
The model code is at [models.py](models.py).
