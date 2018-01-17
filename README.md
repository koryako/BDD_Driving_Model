

伯克利的深度驾驶模型

项目介绍

在BDD 自动驾驶项目中，我们预测未来汽车的运动动作

To attack the task, we collected [Berkeley DeepDrive Video Dataset](https://goo.gl/forms/7XThUcjpGALkqxFU2) with our partner [Nexar](https://www.getnexar.com/),
proposed a FCN+LSTM model and implement it using tensorflow.


为了完成这个任务我们收集了自己的数据集

## The Berkeley DeepDrive Video Dataset(BDD-V)

BDD-V dataset will be released [here](https://goo.gl/forms/7XThUcjpGALkqxFU2). Sign up for notification when we release the data

数据集目前还没有放出，可以使用百度的数据集

## Using Our Code:
### Installation
First clone the codebase to your local file system at $BDD_ROOT.
```
git clone https://github.com/gy20073/BDD_Driving_Model.git && cd BDD_Driving_Model && export BDD_ROOT=$(pwd)

进入BDD_Driving_Model 使得当前目录赋值到变量中
```
For Ubuntu 14.04 and 16.04, you can install all dependencies using:
```
cd $BDD_ROOT && bash setup.sh
安装Anaconda 环境
```
Or if you don't want to install Anaconda or you're using other versions of Linux, you could manually install those packages:

- Tensorflow 0.11
- ffmpeg and ffprobe
- Python packages:
    * IPython, PIL, opencv-python, scipy, matplotlib, numpy, sklearn

### Model Zoo
We provide some pretrained models that are ready to use. Download the model zoo [here](https://drive.google.com/drive/folders/0B7pFVHKojiewM3A4azZmOV9SYkk?usp=sharing) to $BDD_ROOT"/data" and make sure they are available at locations like $BDD_ROOT"/data/discrete_cnn_lstm/model.ckpt-146001.bestmodel".
预训练模型，估计很大

The `tf.caffenet.bin` is a pretrained Alexnet model generated from the [Caffe-Tensorflow](https://github.com/ethereon/caffe-tensorflow) tool. It's used as the finetuning start point for the driving model.

使用 caffe-转tensorflow alexnet 作为权重初始化的值
### Run a Pretrained Model
With the pretrained model, you could test it on your own dashcam video. `wrapper.py` is a simple wrapper that use the model without the requirement to prepare a TFRecord dataset. It takes in an image at every 1/3 second and output the predicted future egomotion. See `wrapper_test.ipynb` for an example usage.
使用wropper.py 可以预测自己的视频文件，一秒3张的速度预测动作
TODO(Improve Visualization)

### Data Preparation:
Download and unzip the dataset's training and validation set into some directory $DATA_ROOT. There should be directories like $DATA_ROOT/train/videos and $DATA_ROOT/val/info.

Then run the following commands to generate indexes of videos and convert raw videos to TFRecords. For validation set:
```
cd $BDD_ROOT"/data_prepare"
python filter.py $DATA_ROOT/val
python prepare_tfrecords.py --video_index=$DATA_ROOT/val/video_filtered_38_60.txt --output_directory=$DATA_ROOT/tfrecords/validation
```
and on the training set:
```
cd $BDD_ROOT"/data_prepare"
python filter.py $DATA_ROOT/train
python prepare_tfrecords.py --video_index=$DATA_ROOT/train/video_filtered_38_60.txt --output_directory=$DATA_ROOT/tfrecords/train
```

### Model Training, Validation and Monitoring:
To train a driving model, first change some path flags in $BDD_ROOT"/config.py". In particular set FLAGS.pretrained_model_path = "$BDD_ROOT/data/tf.caffenet.bin" and FLAGS.data_dir = "$DATA_ROOT/tfrecords". They are paths to the ImageNet pretrained Alexnet model and the TFRecord files we got from the previous data preparation step.

There are a bunch of different types of models proposed in the paper and implemented in this repo. The configuration of each model is a function in `config.py`, such as `discrete_tcnn1` and `continuous_datadriven_bin`. The `discrete_tcnn1` model is a model with temporal convolution of window size 1 and the model predicts discrete driving actions such as `Go`, `Stop`, `Left` and `Right`. The `continuous_datadriven_bin` model is a CNN-LSTM style model that predicts continuous egomotions, including future angular velocity and future speed. The binning method used in this model is a data-driven approach.
https://github.com/ethereon/caffe-tensorflow
We will use `discrete_tcnn1` as a running example, the training procedures of other models are similar. To train the model, run
```
cd $BDD_ROOT && python config.py train discrete_tcnn1
```
For the `continuous_datadriven_bin` model, we need to get the distribution before do the actually training, to get the distribution, run
```
cd $BDD_ROOT && python config.py stat continuous_datadriven_bin
```
During training, the program will write checkpoints and logs to $BDD_ROOT"/data/discrete_tcnn1". To monitor the validation performance, we could start another process evaluating the model periodically
```
cd $BDD_ROOT && python config.py eval discrete_tcnn1
```
One could also use tensorboard to visually monitor the training progress
```
cd $BDD_ROOT"/data" && tensorboard --logdir=. --port=8888
```
and open it at: `http://localhost:8888`


## Reference:
If you want to cite our [**paper**](https://arxiv.org/pdf/1612.01079.pdf), please use the following bibtex:
```
@article{xu2016end,
  title={End-to-end learning of driving models from large-scale video datasets},
  author={Xu, Huazhe and Gao, Yang and Yu, Fisher and Darrell, Trevor},
  journal={arXiv preprint arXiv:1612.01079},
  year={2016}
}
```
