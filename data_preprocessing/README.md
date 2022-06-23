## About data preprocessing

1. Please follow the PartNet data downloading instructions on [the webpage](https://github.com/daerduoCarey/partnet_seg_exps/blob/master/data/README.md) to download `ins_seg_h5.zip`, and unzip it to the current folder.
2. Downloading the folders `after_merging_label_ids` and `train_val_test_split` from [the repo](https://github.com/daerduoCarey/partnet_seg_exps/tree/master/stats).
3. run `python parse_h5_to_points.py` to get all the points files.
4. run `python run_get_points_filelist.py` to get filelists for each level.
5. run `python run_get_points_filelist_level123.py` to get the concated filelists for each category.
6. run `python run_convert_points_to_tfrecords_level123.py` to get all the tfrecords for each category.
