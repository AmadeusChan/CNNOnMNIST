echo "to test learning rate"
python run_cnn.py -test test_lr1.txt -train train_lr1.txt -lr 0.001
python run_cnn.py -test test_lr2.txt -train train_lr2.txt -lr 0.01
python run_cnn.py -test test_lr3.txt -train train_lr3.txt -lr 0.1
python run_cnn.py -test test_lr4.txt -train train_lr4.txt -lr 1

echo "to test weight decay"
python run_cnn.py -test test_wd1.txt -train train_wd1.txt -wd 0
python run_cnn.py -test test_wd2.txt -train train_wd2txt -wd 0.0001
python run_cnn.py -test test_wd3.txt -train train_wd3.txt -wd 0.001
python run_cnn.py -test test_wd4.txt -train train_wd4.txt -wd 0.01

echo "to test batch size"
python run_cnn.py -test test_bs1.txt -train train_bs1.txt -bs 50
python run_cnn.py -test test_bs3.txt -train train_bs3.txt -bs 100
python run_cnn.py -test test_bs5.txt -train train_bs5.txt -bs 200

