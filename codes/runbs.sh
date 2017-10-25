echo "to test batch size"
python run_cnn.py -test test_bs2.txt -train train_bs2.txt -bs 50
python run_cnn.py -test test_bs3.txt -train train_bs3.txt -bs 100
python run_cnn.py -test test_bs5.txt -train train_bs5.txt -bs 200

