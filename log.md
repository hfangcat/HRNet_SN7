- [x] Add yaml file for spacenet7
- [x] Generate training data and labels (train.lst/val.lst/test.lst)
python tools.py /local_storage/datasets/sn7_winner_split/train train 2>err.log
python tools.py /local_storage/datasets/sn7_winner_split/test_public test
- [x] Modify the dataset class for spacenet7
- [x] Download the pretrained model