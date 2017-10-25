echo "data augmentation"
python run_cnn.py -test test_da.txt -train train_da.txt -wd 0.0001 -da on

echo "to test learning rate"
python run_cnn.py -test test_lr1.txt -train train_lr1.txt -lr 0.001
python run_cnn.py -test test_lr2.txt -train train_lr2.txt -lr 0.01
python run_cnn.py -test test_lr3.txt -train train_lr3.txt -lr 0.1



