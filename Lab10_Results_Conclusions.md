Batchnorm before relu
/home/ibab/Machine_Learning/.venv/bin/python /home/ibab/Deep_Learning_Labs/lab10/DL_Lab10_FFN.py 
Original shape: (10463, 462)
Input data(landmark) shape: (462, 943)
Output data(target) shape: (462, 9520)
462
Epoch 100/1000, Training Loss: 2036.2618
Epoch 200/1000, Training Loss: 865.7452
Epoch 300/1000, Training Loss: 584.7555
Epoch 400/1000, Training Loss: 501.5042
Epoch 500/1000, Training Loss: 439.3285
Epoch 600/1000, Training Loss: 423.8748
Epoch 700/1000, Training Loss: 416.8407
Epoch 800/1000, Training Loss: 399.2975
Epoch 900/1000, Training Loss: 379.6279
Epoch 1000/1000, Training Loss: 379.1832
Validation Loss: 1732.7011
Test Loss: 1296.0217

Process finished with exit code 0



Batchnorm after Relu
/home/ibab/Machine_Learning/.venv/bin/python /home/ibab/Deep_Learning_Labs/lab10/DL_Lab10_FFN.py 
Original shape: (10463, 462)
Input data(landmark) shape: (462, 943)
Output data(target) shape: (462, 9520)
462
Epoch 100/1000, Training Loss: 1267.8615
Epoch 200/1000, Training Loss: 664.4212
Epoch 300/1000, Training Loss: 473.6973
Epoch 400/1000, Training Loss: 437.6077
Epoch 500/1000, Training Loss: 421.4812
Epoch 600/1000, Training Loss: 371.9151
Epoch 700/1000, Training Loss: 356.8926
Epoch 800/1000, Training Loss: 353.3967
Epoch 900/1000, Training Loss: 367.8783
Epoch 1000/1000, Training Loss: 340.7116
Validation Loss: 1732.9345
Test Loss: 1331.7407

Process finished with exit code 0



Hyperparamter Tuning (lr-Learning Rate,dropout_prob-Dropout Probability)
/home/ibab/Machine_Learning/.venv/bin/python /home/ibab/Deep_Learning_Labs/lab10/DL_Lab10_FFN.py 
==================================================
Tuning with lr=0.01, dropout=0.3
Original shape: (10463, 462)
Input data(landmark) shape: (462, 943)
Output data(target) shape: (462, 9520)
Epoch 100/1000, Training Loss: 945.0121
Epoch 200/1000, Training Loss: 561.4217
Epoch 300/1000, Training Loss: 455.2613
Epoch 400/1000, Training Loss: 428.6166
Epoch 500/1000, Training Loss: 388.8217
Epoch 600/1000, Training Loss: 381.2570
Epoch 700/1000, Training Loss: 390.7099
Epoch 800/1000, Training Loss: 384.8721
Epoch 900/1000, Training Loss: 371.3734
Epoch 1000/1000, Training Loss: 379.4243
==================================================
Tuning with lr=0.01, dropout=0.5
Original shape: (10463, 462)
Input data(landmark) shape: (462, 943)
Output data(target) shape: (462, 9520)
Epoch 100/1000, Training Loss: 1224.5480
Epoch 200/1000, Training Loss: 858.9004
Epoch 300/1000, Training Loss: 784.3170
Epoch 400/1000, Training Loss: 668.3578
Epoch 500/1000, Training Loss: 680.1583
Epoch 600/1000, Training Loss: 686.0909
Epoch 700/1000, Training Loss: 664.3783
Epoch 800/1000, Training Loss: 642.1187
Epoch 900/1000, Training Loss: 609.3255
Epoch 1000/1000, Training Loss: 586.5241
==================================================
Tuning with lr=0.005, dropout=0.3
Original shape: (10463, 462)
Input data(landmark) shape: (462, 943)
Output data(target) shape: (462, 9520)
Epoch 100/1000, Training Loss: 1604.9618
Epoch 200/1000, Training Loss: 827.6748
Epoch 300/1000, Training Loss: 574.6489
Epoch 400/1000, Training Loss: 495.0362
Epoch 500/1000, Training Loss: 443.8419
Epoch 600/1000, Training Loss: 446.7831
Epoch 700/1000, Training Loss: 401.4605
Epoch 800/1000, Training Loss: 393.3358
Epoch 900/1000, Training Loss: 367.0266
Epoch 1000/1000, Training Loss: 365.1277
==================================================
Tuning with lr=0.005, dropout=0.5
Original shape: (10463, 462)
Input data(landmark) shape: (462, 943)
Output data(target) shape: (462, 9520)
Epoch 100/1000, Training Loss: 2018.3597
Epoch 200/1000, Training Loss: 1067.4304
Epoch 300/1000, Training Loss: 863.1521
Epoch 400/1000, Training Loss: 743.2219
Epoch 500/1000, Training Loss: 682.4494
Epoch 600/1000, Training Loss: 676.2766
Epoch 700/1000, Training Loss: 661.9792
Epoch 800/1000, Training Loss: 631.7712
Epoch 900/1000, Training Loss: 582.9527
Epoch 1000/1000, Training Loss: 618.4868
==================================================
Tuning with lr=0.001, dropout=0.3
Original shape: (10463, 462)
Input data(landmark) shape: (462, 943)
Output data(target) shape: (462, 9520)
Epoch 100/1000, Training Loss: 15559.4102
Epoch 200/1000, Training Loss: 4970.0831
Epoch 300/1000, Training Loss: 2105.1645
Epoch 400/1000, Training Loss: 1641.1696
Epoch 500/1000, Training Loss: 1271.8796
Epoch 600/1000, Training Loss: 1004.7809
Epoch 700/1000, Training Loss: 785.9089
Epoch 800/1000, Training Loss: 672.6128
Epoch 900/1000, Training Loss: 568.1039
Epoch 1000/1000, Training Loss: 510.1393
==================================================
Tuning with lr=0.001, dropout=0.5
Original shape: (10463, 462)
Input data(landmark) shape: (462, 943)
Output data(target) shape: (462, 9520)
Epoch 100/1000, Training Loss: 16006.2904
Epoch 200/1000, Training Loss: 5176.9900
Epoch 300/1000, Training Loss: 2260.2042
Epoch 400/1000, Training Loss: 1842.5700
Epoch 500/1000, Training Loss: 1523.7811
Epoch 600/1000, Training Loss: 1352.2852
Epoch 700/1000, Training Loss: 1105.7202
Epoch 800/1000, Training Loss: 957.2029
Epoch 900/1000, Training Loss: 890.8816
Epoch 1000/1000, Training Loss: 805.8547

Hyperparameter tuning results:
      lr  dropout     val_loss
0  0.010      0.3  1269.926312
1  0.010      0.5  1373.385965
2  0.005      0.3  2138.958034
3  0.005      0.5  1562.644744
4  0.001      0.3  1946.461144
5  0.001      0.5  1369.253715

Best hyperparameters: lr=0.01, dropout=0.3

Final evaluation on test set:
Original shape: (10463, 462)
Input data(landmark) shape: (462, 943)
Output data(target) shape: (462, 9520)
Epoch 100/1000, Training Loss: 980.6619
Epoch 200/1000, Training Loss: 546.0997
Epoch 300/1000, Training Loss: 473.3755
Epoch 400/1000, Training Loss: 427.5102
Epoch 500/1000, Training Loss: 405.5704
Epoch 600/1000, Training Loss: 426.2359
Epoch 700/1000, Training Loss: 399.2144
Epoch 800/1000, Training Loss: 371.5286
Epoch 900/1000, Training Loss: 390.1351
Epoch 1000/1000, Training Loss: 369.1394
Test Loss: 1771.2796

Process finished with exit code 0

lr
dropout
val_loss
0.01
0.3
1269.93
0.01
0.5
1373.39
0.005
0.3
2138.96
0.005
0.5
1562.64
0.001
0.3
1946.46
0.001
0.5
1369.25
Observations:
    • High learning rate (0.01) performs better than lower learning rates for this dataset.
    • Lower dropout (0.3) generally performs better than 0.5, except at lr=0.001.
Best combination: lr=0.01, dropout=0.3 → val_loss = 1269.93.
 WITHOUT DROPOUT
/home/ibab/Machine_Learning/.venv/bin/python /home/ibab/Deep_Learning_Labs/lab10/DL_Lab10_FFN.py 
Original shape: (10463, 462)
Input data(landmark) shape: (462, 943)
Output data(target) shape: (462, 9520)
Epoch 100/1000, Training Loss: 683.2459
Epoch 200/1000, Training Loss: 229.5728
Epoch 300/1000, Training Loss: 145.8185
Epoch 400/1000, Training Loss: 103.1656
Epoch 500/1000, Training Loss: 114.0832
Epoch 600/1000, Training Loss: 91.0540
Epoch 700/1000, Training Loss: 74.5155
Epoch 800/1000, Training Loss: 77.0471
Epoch 900/1000, Training Loss: 61.2644
Epoch 1000/1000, Training Loss: 68.0009
Test Loss: 1874.6065

Process finished with exit code 0

WITHOUT  BATHCNORM
/home/ibab/Machine_Learning/.venv/bin/python /home/ibab/Deep_Learning_Labs/lab10/DL_Lab10_FFN.py 
Original shape: (10463, 462)
Input data(landmark) shape: (462, 943)
Output data(target) shape: (462, 9520)
Epoch 100/1000, Training Loss: 2992.1720
Epoch 200/1000, Training Loss: 2554.3720
Epoch 300/1000, Training Loss: 2460.5881
Epoch 400/1000, Training Loss: 2328.5961
Epoch 500/1000, Training Loss: 2447.2993
Epoch 600/1000, Training Loss: 2511.4016
Epoch 700/1000, Training Loss: 2322.2081
Epoch 800/1000, Training Loss: 2465.8704
Epoch 900/1000, Training Loss: 2320.1645
Epoch 1000/1000, Training Loss: 2438.9839
Test Loss: 4440.8008

Process finished with exit code 0

CONCLUSIONS:
    • BatchNorm: Batch normalization significantly stabilizes training and reduces training loss compared to no BatchNorm. Training converges smoothly.
    • Dropout: Adding dropout slows down training and increases validation loss slightly; we can increase the number of layers and check.
    • No Dropout: Training loss drops very fast and reaches very low values, so dropout is not explicitly needed in this case
    • No BatchNorm: Without batch normalization, the network struggles to converge; training loss remains high and fluctuates showing unstable training and poor generalization.
    • The performance of the model can be evaluated with evaluation metrics like accuracy,f1-score etc(not included here).
    • The best hyper parameters are lr=0.01 and dropout_prob=0.3(although no dropout works best)
