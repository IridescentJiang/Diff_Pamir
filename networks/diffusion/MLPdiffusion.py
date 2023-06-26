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
        self.conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels=5312, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=128, out_channels=5312, kernel_size=1),  # input shape = output shape = 2
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, t):
        # 训练的时候不是按顺序给的，是随机采样的，需要额外的时间信息
        # 也可以不用embedding,直接送T进去训练，但这种信息无法很好学到，embedding层会嵌入时间的信息，变成网络更容易理解的模式。
        # transformer也是这样加时间信息的
        # x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)  # 把t编码为128维的embedding
            x = self.conv[2 * idx](x)  # conv
            t_embedding = t_embedding.transpose(1, 2).contiguous()
            x = x + t_embedding  # embedding是通过加法加进去的
            x = self.conv[2 * idx + 1](x)  # relu

        x = self.conv[-1](x)  # conv，保证x的形状不变

        return x

    def p_sample_loop(self, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
        """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
        cur_x = torch.randn(shape).to(self.device)
        x_seq = [cur_x]
        for i in reversed(range(n_steps)):
            # 逆扩散过程是自回归的，即必须按顺序依次推出x[t],x[t-1],x[t-2]...
            # 不能并行inference
            cur_x = self.p_sample(cur_x, i, betas.to(self.device), one_minus_alphas_bar_sqrt.to(self.device))
            x_seq.append(cur_x)
        # 把很多步采样拼起来
        return x_seq

    def p_sample(self, x, t, betas, one_minus_alphas_bar_sqrt):  # 参数重整化的过程
        """从x[t]采样t-1时刻的重构值，即从x[t]采样出x[t-1]"""
        t = torch.tensor([t]).to(self.device)

        coeff = (betas[t] / one_minus_alphas_bar_sqrt[t]).to(self.device)

        eps_theta = self.forward(x, t.unsqueeze(-1))

        mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

        # 得到mean后，再生成一个随机量z，
        z = torch.randn_like(x)
        sigma_t = betas[t].sqrt()

        sample = mean + sigma_t * z
        # 上面就单步采样
        return (sample)

    def diffusion_loss_fn(self, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
        """对任意时刻t进行采样计算loss"""
        batch_size = x_0.shape[0]
        # n_steps是为了算loss的时候，可以在n_steps这个范围内随机地生成一些t

        # 对一个batchsize样本生成随机的时刻t,覆盖到更多不同的t
        t = torch.randint(0, n_steps, size=(batch_size // 2,))  # size=(batch_size//2,)中的,不可少
        t = torch.cat([t, n_steps - 1 - t], dim=0)  # [batchsize]
        t = t.unsqueeze(-1).unsqueeze(-1).to(self.device)  # [batchsize, 1, 1]

        # x0的系数
        a = alphas_bar_sqrt[t].to(self.device)

        # eps的系数
        aml = one_minus_alphas_bar_sqrt[t].to(self.device)

        # 生成随机噪音eps
        e = torch.randn_like(x_0).to(self.device)

        # 构造模型的输入,即x_t可以用x_0和t来表示
        x = x_0 * a + e * aml

        # 送入模型，得到t时刻的随机噪声预测值
        output = self.forward(x, t.squeeze(-1))

        # 与真实噪声一起计算误差，求平均值
        return (e - output).square().mean()
        # 目的：让网络预测的噪声  接近于  真实扩散过程的噪声
