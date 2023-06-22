import torch
import torch.nn as nn


class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()

        # https://blog.csdn.net/weixin_36670529/article/details/105910767
        # 它和torch的其他机制结合紧密，继承了nn.Module的网络模型class可以使用nn.ModuleList并识别其中的parameters，当然这只是个list，不会自动实现forward方法。可见，
        # 普通list中的子module并不能被主module所识别，而ModuleList中的子module能够被主module所识别
        # nn.Sequential定义的网络中各层会按照定义的顺序进行级联，需要保证各层的输入和输出之间要衔接
        # nn.Sequential实现了forward()方法，因此可以直接通过x = self.combine(x)的方式实现forward
        # 而nn.ModuleList则没有顺序性要求，也没有forward()方法
        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2),  # input shape = output shape = 2
            ]
        )
        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),  # 一个字典中有100个词，每个词嵌入向量的大小为128 dim
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),  # n_step=T=100步，num_units即hidden layer的parameter
            ]
        )

    def forward(self, x, t):
        # 训练的时候不是按顺序给的，是随机采样的，需要额外的时间信息
        # 也可以不用embedding,直接送T进去训练，但这种信息无法很好学到，embedding层会嵌入时间的信息，变成网络更容易理解的模式。
        # transformer也是这样加时间信息的
        # x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)  # 把t编码为128维的embedding
            x = self.linears[2 * idx](x)  # linear
            x += t_embedding  # embedding是通过加法加进去的
            x = self.linears[2 * idx + 1](x)  # relu

        x = self.linears[-1](x)  # linear，保证x的形状不变

        return x
