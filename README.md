# Conditional Variational Autoencoder (CVAE) | 条件变分自编码器

### 选择语言 | Language

[中文简介](#简介) | [English](#introduction)

### 结果 | Result

**实战 1：CIFAR-10。** 每张图的损失从第 1 个 epoch 的 67.23 降到第 20 个 epoch 的 44.86。重构项从 61.32 降到 33.31，KL 从 5.91 升到 11.54 后停住。20 个 epoch 后，同一类的采样共享天空、地面或车身这类颜色，轮廓仍然糊。上排原图、下排重构能对上大致色块，笔画级的细节没有出现。



训练损失曲线（`outputs/loss.png`）：

<img width="1036" height="616" alt="image" src="https://github.com/user-attachments/assets/6a0d9823-a2b0-46ce-9937-775373b661a9" />



生成与重构（`outputs/progress.png`）：

<img width="1097" height="1788" alt="image" src="https://github.com/user-attachments/assets/0dadb8b0-df19-4210-8aed-644b2e83a8da" />


**实战 2：LFW。** 9164 张脸，训练 8248、测试 916。每张脸的损失从 137.66 降到 21.94，下半脸重构从 133.63 降到 17.71，KL 先升到约 12 再回到 4.23。第 1 个 epoch 的下半脸是灰块，第 25 个 epoch 出现五官，第 100 个 epoch 嘴和下巴能对上上半脸。第 124 个 epoch 里，同一张上半脸的五个样本在表情和嘴型上分开，上半脸保持原图。


训练损失曲线（`outputs_lfw/loss.png`）：

<img width="1036" height="616" alt="image" src="https://github.com/user-attachments/assets/7ae39b30-aaf9-4734-be84-b829c90d9e88" />


生成补全过程（`outputs_lfw/progress.png`）：

<img width="448" height="1901" alt="image" src="https://github.com/user-attachments/assets/8a691573-f981-4803-802a-2b8d406a4ceb" />


---

## 简介

条件变分自编码器（Conditional Variational Autoencoder，简称 CVAE）由 **Kihyuk Sohn、Honglak Lee 和 Xinchen Yan** 在 NeurIPS 2015 提出。论文要解决的是结构化输出：给定条件 `x`，输出 `y` 可以有很多种合理答案。CVAE 在 Kingma 与 Welling 的变分自编码器上加了一个条件，用潜变量 `z` 把这一个 `x` 对多个 `y` 的映射摊开。

生成过程是

```
z ~ p(z | x)
y ~ p(y | x, z)
```

训练时识别网络同时看 `x` 和 `y`，近似后验 `q(z|x,y)`。先验也是一个网络 `p(z|x)`，随条件改变，而不是对所有样本共用一个标准正态。目标是条件对数似然的变分下界：重构项减去 `KL(q(z|x,y) || p(z|x))`。采样仍然用重参数化，`z = μ + ε ⊙ σ`，梯度可以穿过随机节点。

原文的三个实验都是「给定输入，预测结构化输出」：MNIST 象限补全、Caltech-UCSD Birds 的鸟类分割，以及 LFW 上头发、皮肤、衣服、背景四类分割。本仓库用 PyTorch 实现论文第 4 节的三套网络。架构一格用展平的 28×28 和 10 维 one-hot 把维度走一遍。两个实战分别是 CIFAR-10 上的类别条件生成，以及 LFW 照片上的上半脸补全。像素级分割标注不在公开的 LFW 图像包里，所以实战 2 把论文里的一对多补全放到下半张脸上。

## 架构
CVAE（Conditional Variational Autoencoder）要做的事情非常直接：**给生成过程加一个“条件”** 。这个条件可以是类别标签、文本描述、属性向量，甚至是另一张图像。在 MNIST 的例子里，条件就是数字的 one-hot 标签。VAE 学的是 p(x)，而 CVAE 学的是 `p(x|c)`——**在给定条件 \(c\) 的前提下，数据 \(x\) 的分布**。用一句话概括：VAE 是“自由画家”，CVAE 是“命题画家”。模型分成先验、识别网络、生成网络。条件 `x` 进先验；`x` 和输出 `y` 一起进识别网络；采样得到的 `z` 再和 `x` 一起进生成网络。
<img width="919" height="662" alt="image" src="https://github.com/user-attachments/assets/3d42b587-6774-4907-b214-81c42cb22677" />


```
x, y → q(z|x,y) → μ_q, log σ_q² → z = μ_q + ε σ_q
x    → p(z|x)   → μ_p, log σ_p²
z, x → p(y|x,z) → ŷ
```

- **先验 p(z|x)**：只看条件。CIFAR-10 上条件是 10 维 one-hot，用两层 256 宽的 MLP 输出 128 维高斯的均值和对数方差。LFW 上条件是上半张脸和 mask，四层步长为 2 的卷积收到 4×4，再投到 128 维。
- **识别网络 q(z|x,y)**：训练时才用到。CIFAR-10 把标签铺成与图像同大小的通道，和 RGB 拼在一起，32→16→8→4→2。LFW 把完整的脸、上半脸和 mask 拼成 7 个通道，同样收到 4×4。两边都再输出均值和对数方差。
- **重参数化**：`z = μ_q + ε ⊙ exp(0.5 · log σ_q²)`，`ε ~ N(0, I)`。网络输出对数方差，指数之后标准差为正。测试时不再走识别网络，改从先验采样 `z`。
- **生成网络 p(y|x,z)**：CIFAR-10 把 `[z; x]` 投到 `256×2×2`，四层反卷积回到 `3×32×32`，Sigmoid 把均值限制在 0 到 1。LFW 先把上半脸编到 `256×4×4`，和 `z` 的特征在通道上拼接，反卷积回 `3×64×64`，再把真实的上半张贴回去。
- **损失**：CIFAR-10 的像素是连续值，重构项是固定方差高斯的负对数似然 `0.5||y-μ||²`，对整张图求和后按图片平均。LFW 只在下半张脸上算同一项。KL 是两个对角高斯之间的闭式，参照分布是条件先验 `p(z|x)`。每个样本抽 1 次 `z`，对应论文里 L = 1 的训练目标。

架构一格不训练。它用 batch 8、图像 784 维、条件 10 维、潜变量 20 维的 MLP 打印每一层形状。先验参数量 78,888，识别网络 279,592，生成网络 275,216。随机权重下每张图的二元交叉熵约 544.48，KL 约 0.13。

| | 架构 MLP | CIFAR-10 卷积 | LFW 卷积 |
| --- | --- | --- | --- |
| 参数量 | 633,696 | 1,924,291 | 7,455,587 |
| 条件 `x` | 10 维 one-hot | 10 维 one-hot | 上半张脸 + mask |
| 输出 `y` | 784 维，Sigmoid | 3×32×32 | 下半张 64×64 的脸 |
| 潜变量 | 20 | 128 | 128 |
| 重构项 | 二元交叉熵 | `0.5\|\|y-μ\|\|²` | 只在下半脸的 `0.5\|\|y-μ\|\|²` |
| 训练 | 只打印维度 | 20 epoch，batch 128 | 124 epoch，batch 64 |

优化器都是 Adam，学习率 `1e-3`，种子 0。对数方差层的权重和偏置初始化为 0，让训练开始时先验和后验接近标准正态。

## 数据集

两个实战共用上面的三套网络和重参数化。条件、输出和重构区域不同。都不做翻转。

### 实战 1：CIFAR-10

CIFAR-10 是 32×32 的 RGB 图像，10 类：airplane、automobile、bird、cat、deer、dog、frog、horse、ship、truck。这里用类别 one-hot 当条件，整张图当输出。

- **规模**：训练集 50,000 张，测试集 10,000 张。数据在 `data/cifar-10-batches-py`，训练时不重新下载。
- **预处理**：只做 `ToTensor()`，像素落到 0 到 1，与 Sigmoid 输出一致。
- **训练**：batch 128，潜变量 128，隐藏层 256，20 个 epoch。每个 epoch 按类别各采 8 张，并保存 8 张测试图的重构。
- **这一轮的数字**：第 20 个 epoch 每张图 loss 44.86，重构 33.31，KL 11.54。采样图按行分成 10 类，类间颜色分开，类内细节仍然糊。这是 20 个 epoch 的卷积 CVAE 在像素高斯似然下的结果，高斯解码器会把多种可能的图像平均成一块。

| 每张图的 loss | 这一轮里看到的现象 |
| --- | --- |
| 约 67（第 1 个 epoch） | 重构和采样都是平均色块 |
| 约 45（第 20 个 epoch） | 类间颜色分开，轮廓糊，KL 停在 11 附近 |

数据集官方地址：[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

### 实战 2：LFW

LFW（Labeled Faces in the Wild）是自然场景里的人脸照片。论文第 5.2 节用的是标好头发、皮肤、衣服、背景的子集。这份像素标注不在公开图像包里。本实验改做和论文第 5.1 节同一类的结构化补全：上半张脸是条件 `x`，下半张脸是输出 `y`。

- **规模**：`data/` 里一个人有多张照片的子集，共 9164 张。按 9:1 划分，训练 8248，测试 916，划分种子为 0。
- **预处理**：先把短边缩到 250，中心裁成 160×160，再缩到 64×64，`ToTensor()` 后像素在 0 到 1。上半张 mask 为 1，下半张为 0。
- **训练**：batch 64，潜变量 128。`outputs_lfw/` 里的曲线是 124 个 epoch。notebook 里的默认值是 20，要看到第 100 个 epoch 附近的五官，把 `EPOCHS` 加大。
- **这一轮的数字**：第 124 个 epoch 每张脸 loss 21.94，下半脸重构 17.71，KL 4.23。同一条件的五个样本共享上半脸，嘴型和下巴分开。

| 每张脸的 loss | 这一轮里看到的现象 |
| --- | --- |
| 约 138（第 1 个 epoch） | 下半脸是灰块 |
| 约 50（第 25 个 epoch 前后） | 出现眼鼻嘴的位置 |
| 约 22（第 124 个 epoch） | 表情随 `z` 变化，上半脸保持原图 |

数据集官方地址：[http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)

---

## Introduction

The Conditional Variational Autoencoder (CVAE) was introduced by **Kihyuk Sohn, Honglak Lee, and Xinchen Yan** at NeurIPS 2015. The paper studies structured outputs: given a condition `x`, many different outputs `y` can all be valid. CVAE adds that condition to the variational autoencoder of Kingma and Welling. A latent variable `z` carries the remaining choice, so one `x` can map to many `y`.

The generative story is

```
z ~ p(z | x)
y ~ p(y | x, z)
```

During training, a recognition network sees both `x` and `y` and approximates the posterior `q(z|x,y)`. The prior is also a network, `p(z|x)`, so it changes with the condition. The training objective is the variational lower bound on the conditional log-likelihood: a reconstruction term minus `KL(q(z|x,y) || p(z|x))`. Sampling uses the reparameterization `z = μ + ε ⊙ σ`, and the gradient passes through the random node.

The original experiments are all input-to-structured-output tasks: MNIST quadrant completion, bird segmentation on Caltech-UCSD Birds, and a four-class face segmentation (hair, skin, clothes, background) on LFW. This repository implements the three networks from Section 4 in PyTorch. The architecture cell walks tensor shapes on a flattened 28×28 image and a 10-dimensional one-hot condition. The two training runs are class-conditional generation on CIFAR-10, and lower-face completion on LFW photographs. The public LFW image archive does not include the pixel part labels, so the second run places the paper's one-to-many completion on the lower half of the face.

## Architecture

The model has a prior, a recognition network, and a generation network. The condition `x` enters the prior. `x` and the output `y` enter the recognition network together. The sampled `z` enters the generation network together with `x`.

```
x, y → q(z|x,y) → μ_q, log σ_q² → z = μ_q + ε σ_q
x    → p(z|x)   → μ_p, log σ_p²
z, x → p(y|x,z) → ŷ
```

- **Prior p(z|x)**: Sees only the condition. On CIFAR-10 the condition is a 10-dimensional one-hot vector. A two-layer MLP of width 256 outputs the mean and log-variance of a 128-dimensional Gaussian. On LFW the condition is the upper face and a mask. Four stride-2 convolutions reduce it to 4×4, then a linear layer produces 128 dimensions.
- **Recognition network q(z|x,y)**: Used only while training. On CIFAR-10 the label is tiled into extra channels and concatenated with RGB. The map is 32→16→8→4→2. On LFW the full face, the observed upper face, and the mask form 7 channels and are reduced to 4×4. Both heads then emit a mean and a log-variance.
- **Reparameterization**: `z = μ_q + ε ⊙ exp(0.5 · log σ_q²)`, with `ε ~ N(0, I)`. The network emits a log-variance so the standard deviation stays positive. At test time the recognition network is skipped and `z` is drawn from the prior.
- **Generation network p(y|x,z)**: On CIFAR-10, `[z; x]` is projected to `256×2×2` and four transposed convolutions return `3×32×32`. A sigmoid keeps the mean in 0 to 1. On LFW the upper face is encoded to `256×4×4`, concatenated with the features of `z`, and decoded to `3×64×64`. The true upper half is pasted back.
- **Loss**: CIFAR-10 pixels are continuous. The reconstruction term is the negative log-likelihood of a fixed-variance Gaussian, `0.5||y-μ||²`, summed over the image and then averaged per image. LFW computes the same term on the lower half only. The KL term is the closed form between two diagonal Gaussians. Its reference distribution is the conditional prior `p(z|x)`. Each sample draws `z` once, which is the paper's training objective with L = 1.

The architecture cell does not train. It prints every layer for an MLP with batch 8, a 784-dimensional image, a 10-dimensional condition, and a 20-dimensional latent code. The prior has 78,888 parameters, the recognition network 279,592, and the generation network 275,216. At random initialization the per-image binary cross-entropy is about 544.48 and the KL is about 0.13.

| | Architecture MLP | CIFAR-10 conv | LFW conv |
| --- | --- | --- | --- |
| Parameters | 633,696 | 1,924,291 | 7,455,587 |
| Condition `x` | 10-D one-hot | 10-D one-hot | upper face + mask |
| Output `y` | 784-D, Sigmoid | 3×32×32 | lower half of a 64×64 face |
| Latent size | 20 | 128 | 128 |
| Reconstruction | binary cross-entropy | `0.5\|\|y-μ\|\|²` | `0.5\|\|y-μ\|\|²` on the lower half |
| Training | shape print only | 20 epochs, batch 128 | 124 epochs, batch 64 |

Every run uses Adam with learning rate `1e-3` and seed 0. The log-variance layer is initialized at zero, so the prior and the posterior both start near a standard normal.

## Dataset

Both runs share the three networks and the reparameterization. The condition, the output, and the region inside the reconstruction term change. Neither run flips images.

### Experiment 1: CIFAR-10

CIFAR-10 is 32×32 RGB images in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The class one-hot vector is the condition and the whole image is the output.

- **Scale**: 50,000 training images and 10,000 test images. The files live in `data/cifar-10-batches-py` and are not downloaded again.
- **Preprocessing**: `ToTensor()` only, so pixels lie in 0 to 1 and match the sigmoid output.
- **Training**: Batch size 128, latent size 128, hidden width 256, 20 epochs. Each epoch saves 8 samples per class and the reconstruction of 8 test images.
- **This run**: At epoch 20 the per-image loss is 44.86, the reconstruction term is 33.31, and the KL is 11.54. Samples in the same row share a class. Colors separate across rows. Edges stay soft. A Gaussian decoder trained for 20 epochs averages the many images that share a label.

| Per-image loss | What this run shows |
| --- | --- |
| about 67 (epoch 1) | reconstructions and samples are average color blocks |
| about 45 (epoch 20) | class colors separate, outlines stay blurry, KL sits near 11 |

Official dataset page: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

### Experiment 2: LFW

Labeled Faces in the Wild is a set of face photographs taken outside the studio. Section 5.2 of the paper uses a subset labeled into hair, skin, clothes, and background. Those pixel labels are not in the public image archive. This run uses the same kind of structured completion as Section 5.1: the upper face is the condition `x` and the lower face is the output `y`.

- **Scale**: The multiface subset already stored under `data/`, 9,164 images of people who appear more than once. A 9:1 split with seed 0 gives 8,248 training images and 916 test images.
- **Preprocessing**: Resize the short side to 250, center-crop to 160×160, resize to 64×64, then `ToTensor()`. The upper-half mask is 1 and the lower-half mask is 0.
- **Training**: Batch size 64, latent size 128. The curves in `outputs_lfw/` are 124 epochs. The notebook default is 20. Raise `EPOCHS` to reach the faces visible around epoch 100.
- **This run**: At epoch 124 the per-image loss is 21.94, the lower-face reconstruction term is 17.71, and the KL is 4.23. Five samples that share one upper face keep that upper face and differ in the mouth and chin.

| Per-image loss | What this run shows |
| --- | --- |
| about 138 (epoch 1) | the lower face is a gray block |
| about 50 (around epoch 25) | eyes, nose, and mouth start to land in place |
| about 22 (epoch 124) | expression follows `z`, and the upper face stays the original |

Official dataset page: [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)

---

## 原文章 | Original article

Kihyuk Sohn, Xinchen Yan, and Honglak Lee. 2015. Learning structured output representation using deep conditional generative models. In Proceedings of the 29th International Conference on Neural Information Processing Systems - Volume 2 (NIPS'15), Vol. 2. MIT Press, Cambridge, MA, USA, 3483–3491.

arXiv: [https://papers.nips.cc/paper_files/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf](https://papers.nips.cc/paper_files/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf)

本复现用到的变分自编码器：

Kingma D P, Welling M. Auto-Encoding Variational Bayes[C]//International Conference on Learning Representations (ICLR). 2014.

arXiv: [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)
