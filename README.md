# Reimplement of the sequence model in deeplearning.ai

Ref: https://github.com/enggen/Deep-Learning-Coursera

## RNN cell structure:
![image](https://github.com/zhaojiachen1994/rnnLearning/blob/master/figure/rnn_step_forward.png)


## LSTM cell structure:
![image](https://github.com/zhaojiachen1994/rnnLearning/blob/master/figure/LSTM.png)

## Key points:

- LSTM只是在RNN的基础上多了一记忆细胞层，可以把过去的信息和当下的信息隔离开来，过去的信息直接在Cell运行，当下的决策在Hidden State里面运行.
- LSTM的门状态都由a_<t-1>和x_t决定. 门值是一个向量.
