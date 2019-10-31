# Reimplement of the sequence model in deeplearning.ai
该仓库包括两个部分，第一部分复现了Deep-Learning-Coursera课程中序列模型的第一周的作业内容，利用numpy实现了rnn和lstm的前向传播和后向传播；
第二部分实现了利用lstm和lstmcell对minst数据集的识别分类任务。

Reference of part 1: https://github.com/enggen/Deep-Learning-Coursera
Reference of part 2: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py


## RNN cell structure:
![image](https://github.com/zhaojiachen1994/rnnLearning/blob/master/figure/rnn_step_forward.png)


## LSTM cell structure:
![image](https://github.com/zhaojiachen1994/rnnLearning/blob/master/figure/LSTM.png)

## Key points:

- LSTM只是在RNN的基础上多了一记忆细胞层，可以把过去的信息和当下的信息隔离开来，过去的信息直接在Cell运行，当下的决策在Hidden State里面运行.
- LSTM的门状态都由a_<t-1>和x_t决定. 门值是一个向量.
- forget gate 和 update gate控制了cell state (memory state); output gate 控制了hidden date.
- LSTM的输出状态有hidden state决定.
