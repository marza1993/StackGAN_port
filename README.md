# StackGAN-pytorch python >= 3.6, torch >= 1.7, CUDA >= 10.x port

This repository, forked from [StackGAN-pytorch](https://github.com/hanzhanggit/StackGAN-Pytorch.git), is an implementation 
of the author's "StackGAN" text-to-image synthesis method, that is compatible with **python >= 3.6 and newer versions of torch and CUDA**, as the original could be run only with python 2.7 and older libraries.

Also, this code can be run both on Ubuntu 18.04 and on Windows 10 x64 (see the **Dependencies** section below).


- [Tensorflow implementation](https://github.com/hanzhanggit/StackGAN)

- [Inception score evaluation](https://github.com/hanzhanggit/StackGAN-inception-model)

- [StackGAN-v2-pytorch](https://github.com/hanzhanggit/StackGAN-v2)

Pytorch implementation for reproducing COCO results in the paper [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1612.03242v2.pdf) by Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, Dimitris Metaxas. The network structure is slightly different from the tensorflow implementation. 

<img src="examples/framework.jpg" width="850px" height="370px"/>


### Dependencies

**Ubuntu**

- Ubuntu version: >= 16.04, <= 18.04 (tested with Ubuntu 18.04). Note that CUDA 8.0 to 10.2 is supported on Ubuntu 16.04, while only CUDA 10.0 to 10.2 is supported on Ubuntu 18.04
- CUDA: 10.1. Use the following command to check wich version is installed on your system: `nvcc --version`
- Python: >= 3.6 (tested with python 3.6.9)
- Torch: 1.7.0+cu101. Use the following command: `pip install torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html`. If you want to install different versions go to [Pytorch download page](https://pytorch.org/get-started/locally/). \
To check that torch works correctly with CUDA run the "test_gpu_torch.py" script and see the generated output (it should output "True", a string describing the CUDA device found and the name of your GPU card).
- Install packages from "requirements_ubuntu_18-04.txt". Use the command: `pip install <path_to>/requirements_ubuntu_18-04.txt`.

**Windows**

- Windows version: 10, 64 bit (x64).
- CUDA: 10.0. Use the following command to check wich version is installed on your system: `nvcc --version`
- Python: >= 3.6 (tested with python 3.7.8)
- Torch: 1.8.1+cu102. Use the following command: `pip install torch==1.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html`. If you want to install different versions go to [Pytorch download page](https://pytorch.org/get-started/locally/).\
  To check that torch works correctly with CUDA run the "test_gpu_torch.py" script and see the generated output.
- Install packages from "requirements_win_10_x64.txt". Use the command: `pip install <path_to>/requirements_win_10_x64.txt`

**Recommended**: in order not to mix different packages and/or python versions on your system, it is convenient to use a virtual environment, which is a self contained environment with all the dependencies needed by your application. with **virtualenv**:
- Create a new virtual environment: `virtualenv -p <path_to_python_executable> <virtual_env_name>`
- Activate the created environment. 
    * on Windows: `<virtual_env_name>/Scripts/activate`
    * on Linux: `source <virtual_env_name>/Scripts/activate`
- Once the environment is activated, install the required dependencies: `pip install <path_to>/requirements_<win/linux>.txt` (see the two previous sections).
- With the `pip list` command, you will see all the packages available within the environment.
- Execute your application
- Deactivate: `deactivate`



### Data

1. Download our preprocessed char-CNN-RNN text embeddings for [training coco](https://drive.google.com/open?id=0B3y_msrWZaXLQXVzOENCY2E3TlU) and  [evaluating coco](https://drive.google.com/open?id=0B3y_msrWZaXLeEs5MTg0RC1fa0U), save them to `data/coco`.
  - [Optional] Follow the instructions [reedscot/icml2016](https://github.com/reedscot/icml2016) to download the pretrained char-CNN-RNN text encoders and extract text embeddings.
2. Download the [coco](http://cocodataset.org/#download) image data. Extract all the images into `data/coco/images` folder.
3. The data folder structure should look like:
```
data
|-- coco
|   |-- images
|   |    COCO_train2014_000000581921.jpg
|   |    COCO_train2014_000000581909.jpg
|   |    ...
|   |-- test
|   |    filename.txt
|   |    filename.pickle
|   |    val_filename.txt
|   |    val_captions.txt
|   |    val_captions.t7
|   |-- train
|   |    char-CNN-RNN-embeddings.pickle
|   |    filenames.pickle 
```
  




### Training
- The steps to train a StackGAN model on the COCO dataset using our preprocessed embeddings.
  - Step 1: train Stage-I GAN (e.g., for 120 epochs) . From the `./code` folder: `python main.py --cfg cfg/coco_s1.yml --gpu <GPU_ID>` (if you only have one GPU card, GPU_ID = 0).
  - Step 2: train Stage-II GAN (e.g., for another 120 epochs). From the `./code` folder: `python main.py --cfg cfg/coco_s2.yml --gpu <GPU_ID>`
- `*.yml` files are example configuration files for training/evaluating our models.
- If you run in GPU memory occupation errors, try reducing the batch size in the `*.yml` files, e.g. from 128 to 64.
- If you want to try your own datasets, [here](https://github.com/soumith/ganhacks) are some good tips about how to train GAN. Also, we encourage to try different hyper-parameters and architectures, especially for more complex datasets.



### Pretrained Model
- [StackGAN for coco](https://drive.google.com/open?id=0B3y_msrWZaXLYjNra2ZSSmtVQlE). Download and save it to `models/coco`.
- **Our current implementation has a higher inception score(10.62±0.19) than reported in the StackGAN paper**



### Evaluating
- From the `./code` folder, run `python main.py --cfg cfg/coco_eval.yml --gpu <GPU_ID>` to generate samples from captions in COCO validation set.
TODO: aggiungere la parte del visualizzatore

Examples for COCO:
 
![](examples/coco_2.png)
![](examples/coco_3.png)

Save your favorite pictures generated by our models since the randomness from noise z and conditioning augmentation makes them creative enough to generate objects with different poses and viewpoints from the same discription :smiley:



### Citing StackGAN
If you find StackGAN useful in your research, please consider citing:

```
@inproceedings{han2017stackgan,
Author = {Han Zhang and Tao Xu and Hongsheng Li and Shaoting Zhang and Xiaogang Wang and Xiaolei Huang and Dimitris Metaxas},
Title = {StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks},
Year = {2017},
booktitle = {{ICCV}},
}
```


**Our follow-up work**

- [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1710.10916)
- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](https://arxiv.org/abs/1711.10485) [[supplementary]](https://1drv.ms/b/s!Aj4exx_cRA4ghK5-kUG-EqH7hgknUA)[[code]](https://github.com/taoxugit/AttnGAN)


**References**

- Generative Adversarial Text-to-Image Synthesis [Paper](https://arxiv.org/abs/1605.05396) [Code](https://github.com/reedscot/icml2016)
- Learning Deep Representations of Fine-grained Visual Descriptions [Paper](https://arxiv.org/abs/1605.05395) [Code](https://github.com/reedscot/cvpr2016)
