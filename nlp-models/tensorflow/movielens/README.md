<img src="https://github.com/zhedongzheng/finch/blob/master/assets/movielens.png">

You need Python 2 for this sub-project, because we need to use [PaddlePaddle](http://www.paddlepaddle.org/) (which only supports Python 2 for now) for processed Movielens data

```
pip install paddlepaddle tensorflow pandas tqdm
```

```
cd ./data
python movielens_paddle.py
cd ..
python train.py
```
```
------------
Epoch [48/50] | Batch [0/3516] | Loss: 2.50 | LR: 0.0001
Epoch [48/50] | Batch [500/3516] | Loss: 1.71 | LR: 0.0001
Epoch [48/50] | Batch [1000/3516] | Loss: 1.50 | LR: 0.0001
Epoch [48/50] | Batch [1500/3516] | Loss: 2.43 | LR: 0.0001
Epoch [48/50] | Batch [2000/3516] | Loss: 2.06 | LR: 0.0001
Epoch [48/50] | Batch [2500/3516] | Loss: 5.85 | LR: 0.0001
Epoch [48/50] | Batch [3000/3516] | Loss: 3.06 | LR: 0.0001
Epoch [48/50] | Batch [3500/3516] | Loss: 2.03 | LR: 0.0001
------------------------------
Testing losses: 2.96130696282
Prediction: 2.96, Actual: 3.00
Prediction: 1.92, Actual: 3.00
Prediction: 2.39, Actual: 3.00
Prediction: 2.82, Actual: 3.00
Prediction: 3.37, Actual: 3.00
------------
Epoch [49/50] | Batch [0/3516] | Loss: 2.50 | LR: 0.0001
Epoch [49/50] | Batch [500/3516] | Loss: 1.70 | LR: 0.0001
Epoch [49/50] | Batch [1000/3516] | Loss: 1.51 | LR: 0.0001
Epoch [49/50] | Batch [1500/3516] | Loss: 2.44 | LR: 0.0001
Epoch [49/50] | Batch [2000/3516] | Loss: 2.04 | LR: 0.0001
Epoch [49/50] | Batch [2500/3516] | Loss: 5.82 | LR: 0.0001
Epoch [49/50] | Batch [3000/3516] | Loss: 3.06 | LR: 0.0001
Epoch [49/50] | Batch [3500/3516] | Loss: 2.00 | LR: 0.0001
------------------------------
Testing losses: 2.97281685357
Prediction: 2.92, Actual: 3.00
Prediction: 1.72, Actual: 3.00
Prediction: 2.40, Actual: 3.00
Prediction: 2.69, Actual: 3.00
Prediction: 3.32, Actual: 3.00
------------

```
