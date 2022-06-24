这是一个中文情感分类任务的实验。具体的，先用skip-gram对中文词进行词向量训练，然后用Transformer-encoder或者LSTM对编码过的语句进行情感二分类。

运行方法：

```bash
cd word_emb
python train.py #预训练词向量，存储在gen.pt中
cd ../LSTM
python train.py #用LSTM进行情感二分类
cd ../transformer
python train.py #用Transformer-encode进行情感二分类
```

