echo "to test weight decay"
python run_cnn.py -test test_wd1.txt -train train_wd1.txt -wd 0.00001
python run_cnn.py -test test_wd2.txt -train train_wd2txt -wd 0.0001
python run_cnn.py -test test_wd3.txt -train train_wd3.txt -wd 0.001
python run_cnn.py -test test_wd4.txt -train train_wd4.txt -wd 0.01
python run_cnn.py -test test_wd5.txt -train train_wd5.txt -wd 0.1

echo "to test batch size"
python run_cnn.py -test test_bs1.txt -train train_bs1.txt -bs 30
python run_cnn.py -test test_bs2.txt -train train_bs2.txt -bs 60
python run_cnn.py -test test_bs3.txt -train train_bs3.txt -bs 100
python run_cnn.py -test test_bs4.txt -train train_bs4.txt -bs 150
python run_cnn.py -test test_bs5.txt -train train_bs5.txt -bs 200

