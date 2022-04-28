## About data preprocessing

1. Please follow the PartNet data downloading instructions on [the webpage](https://github.com/daerduoCarey/partnet_seg_exps/blob/master/data/README.md) to download `ins_seg_h5.zip`, and unzip it to the current folder.
2. Downloading the folders `after_merging_label_ids` and `train_val_test_split` from [the repo](https://github.com/daerduoCarey/partnet_seg_exps/tree/master/stats).
3. run `python parse_h5_to_points.py` to get all the points files.
4. run `python run_get_points_filelist.py` to get filelists for each level.
5. run `python run_get_points_filelist_level123.py` to get the concated filelists for each category.
6. run `python run_convert_points_to_tfrecords_level123.py` to get all the tfrecords for each category.

<!-- <img src="comparison.png" alt="comparison" width=800px; height=331px;/>


## Introduction

This work is based on our * paper. We proposed a new method for 3D shape instance segmentation. You can check our [project webpage](https://isunchy.github.io/projects/3d_instance_segmentation.html) for a quick overview.

Recognizing 3D part instances from 3D point cloud is crucial for 3D structure and scene understanding. Many learning-based approaches simply utilize semantic segmentation and instance center prediction as training tasks and fail to further exploit the inherent relationship between shape semantics and part instances. In this paper, we present a new method for 3D part instance segmentation. Our method exploits semantic segmentation for fusing nonlocal instance features for instance center prediction and further enhances the fusion scheme in a multi- and cross-level way. We also propose a semantic region center prediction task for training and leverage the prediction results to improve the clustering of instance points. Our method outperforms existing methods with a large-margin improvement in the PartNet benchmark. We also demonstrate that our feature fusion scheme can be applied to other existing methods to improve their performance in indoor scene instance segmentation tasks.

In this repository, we release the code and data for training the networks for 3d shape instance segmentation.


## Setup


        docker pull tensorflow/tensorflow:1.15.0-gpu-py3
        docker run -it --runtime=nvidia -v /path/to/3d_instance_segmentation/:/workspace tensorflow/tensorflow:1.15.0-gpu-py3
        cd /workspace
        pip install tqdm scipy scikit-learn --user


## Experiments


### Data Preparation

We provide the Google drive link for downloading the training and test datasets:

>[Training data](https://pan.baidu.com/s/1fIy5LvqkqW_Usr5yoDoMyg) [need update]


### Training

To start the training, run

        $ python 3DInsSegNet.py --logdir log/test_chair --train_data data/Chair_level123_train_4489.tfrecords --test_data data/Chair_level123_test_1217.tfrecords --test_data_visual data/Chair_level123_test_1217.tfrecords --train_batch_size 8 --test_batch_size 1 --max_iter 100000 --test_every_iter 5000 --test_iter 1217 --test_iter_visual 0 --cache_folder test_chair --gpu 0 --n_part_1 6 --n_part_2 30 --n_part_3 39 --level_1_weight 1 --level_2_weight 1 --level_3_weight 1 --phase train --seg_loss_weight 1 --offset_weight 1 --sem_offset_weight 1 --learning_rate 0.1 --delete_0 --notest_visual --depth 6 --weight_decay 0.0001 --stop_gradient --category Chair

### Test

To test a trained model, run

        $ python 3DInsSegNet.py --logdir log/test_chair --train_data data/Chair_level123_train_4489.tfrecords --test_data data/Chair_level123_test_1217.tfrecords --test_data_visual data/Chair_level123_test_1217.tfrecords --train_batch_size 8 --test_batch_size 1 --max_iter 100000 --test_every_iter 5000 --test_iter 1217 --test_iter_visual 0 --cache_folder test_chair --gpu 0 --n_part_1 6 --n_part_2 30 --n_part_3 39 --level_1_weight 1 --level_2_weight 1 --level_3_weight 1 --phase test --seg_loss_weight 1 --offset_weight 1 --sem_offset_weight 1 --learning_rate 0.1 --ckpt weight/Chair --delete_0 --notest_visual --depth 6 --weight_decay 0.0001 --stop_gradient --category Chair

We provide the trained weights used in our paper:

>[Weights](https://pan.baidu.com/s/1BlzepfBJzKMV5VnmAbV_mA) [need update]



## License

MIT Licence

## Contact

Please contact us (Chunyu Sun sunchyqd@gmail.com, Yang Liu yangliu@microsoft.com) if you have any problem about our implementation.

 -->