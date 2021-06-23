#  〇、千言数据集：文本相似度比赛简介

aistudio地址：[打卡零基础PaddleNLP【千言数据集：文本相似度】比赛](https://aistudio.baidu.com/aistudio/projectdetail/2045895)

github地址：[https://github.com/livingbody/paddlenlp_text_similarity](https://github.com/livingbody/paddlenlp_text_similarity)

文本相似度旨在识别两段文本在语义上是否相似。文本相似度在自然语言处理领域是一个重要研究方向，同时在信息检索、新闻推荐、智能客服等领域都发挥重要作用，具有很高的商业价值。

[ 文本相似度：https://aistudio.baidu.com/aistudio/competition/detail/45](https://aistudio.baidu.com/aistudio/competition/detail/45)

目前学术界的一些公开中文文本相似度数据集，在相关论文的支撑下对现有的公开文本相似度模型进行了较全面的评估，具有较高权威性。因此，本开源项目收集了这些权威的数据集，期望对模型效果进行综合的评价，旨在为研究人员和开发者提供学术和技术交流的平台，进一步提升文本相似度的研究水平，推动文本相似度在自然语言处理领域的应用和发展。

## 1.数据集
本次评测的文本相似度数据集包括公开的三个文本相似度数据集，分别为哈尔滨工业大学（深圳）的 LCQMC [1] 和 BQ Coupus [2]，以及谷歌的 PAWS-X（中文） [3]。各数据集的简介如下：

### a) LCQMC
LCQMC（A Large-scale Chinese Question Matching Corpus）, 百度知道领域的中文问题匹配数据集，目的是为了解决在中文领域大规模问题匹配数据集的缺失。该数据集从百度知道不同领域的用户问题中抽取构建数据。

### b) BQ Corpus
BQ Corpus（Bank Question Corpus）, 银行金融领域的问题匹配数据，包括了从一年的线上银行系统日志里抽取的问题pair对，是目前最大的银行领域问题匹配数据。

### c) PAWS-X (中文)
PAWS (Paraphrase Adversaries from Word Scrambling)，谷歌发布的包含 7 种语言释义对的数据集，包括PAWS（英语） 与 PAWS-X（多语）。数据集里包含了释义对和非释义对，即识别一对句子是否具有相同的释义（含义），特点是具有高度重叠词汇，对于进一步提升模型对于强负例的判断很有帮助。

各个数据集的任务均一致，即判断两段文本在语义上是否相似的二分类任务。以LCQMC为例：

## 2.提交方式
```
# 文本相似度任务                                                                 
index   prediction                                                              
0   1                                                                           
1   0                                                                           
2   1    
```

# 二、思路

* 1.搭建网络
* 2.更换数据集
* 3.针对不同数据集Finetune 生成不同模型
* 4.用不同模型进行预测

# 三、环境准备
## 1.包引入


```python
!python -m pip install --upgrade paddlenlp==2.0.2 -i https://mirror.baidu.com/pypi/simple
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple
    Collecting paddlenlp==2.0.2
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/b1/e9/128dfc1371db3fc2fa883d8ef27ab6b21e3876e76750a43f58cf3c24e707/paddlenlp-2.0.2-py3-none-any.whl (426kB)
    [K     |████████████████████████████████| 430kB 18.9MB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (1.2.2)
    Requirement already satisfied, skipping upgrade: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (2.9.0)
    Requirement already satisfied, skipping upgrade: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (4.1.0)
    Requirement already satisfied, skipping upgrade: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (0.4.4)
    Requirement already satisfied, skipping upgrade: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (0.70.11.1)
    Requirement already satisfied, skipping upgrade: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (0.42.1)
    Requirement already satisfied, skipping upgrade: visualdl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp==2.0.2) (2.1.1)
    Requirement already satisfied, skipping upgrade: scikit-learn>=0.21.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp==2.0.2) (0.24.2)
    Requirement already satisfied, skipping upgrade: numpy>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp==2.0.2) (1.20.3)
    Requirement already satisfied, skipping upgrade: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from h5py->paddlenlp==2.0.2) (1.15.0)
    Requirement already satisfied, skipping upgrade: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp==2.0.2) (0.3.3)
    Requirement already satisfied, skipping upgrade: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (3.14.0)
    Requirement already satisfied, skipping upgrade: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (1.0.0)
    Requirement already satisfied, skipping upgrade: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (0.7.1.1)
    Requirement already satisfied, skipping upgrade: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (1.1.1)
    Requirement already satisfied, skipping upgrade: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (7.1.2)
    Requirement already satisfied, skipping upgrade: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (0.8.53)
    Requirement already satisfied, skipping upgrade: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (2.22.0)
    Requirement already satisfied, skipping upgrade: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (3.8.2)
    Requirement already satisfied, skipping upgrade: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp==2.0.2) (1.21.0)
    Requirement already satisfied, skipping upgrade: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp==2.0.2) (1.6.3)
    Requirement already satisfied, skipping upgrade: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp==2.0.2) (2.1.0)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp==2.0.2) (0.14.1)
    Requirement already satisfied, skipping upgrade: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp==2.0.2) (2.8.0)
    Requirement already satisfied, skipping upgrade: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp==2.0.2) (2019.3)
    Requirement already satisfied, skipping upgrade: Jinja2>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp==2.0.2) (2.10.1)
    Requirement already satisfied, skipping upgrade: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp==2.0.2) (7.0)
    Requirement already satisfied, skipping upgrade: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp==2.0.2) (0.16.0)
    Requirement already satisfied, skipping upgrade: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp==2.0.2) (1.1.0)
    Requirement already satisfied, skipping upgrade: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp==2.0.2) (0.18.0)
    Requirement already satisfied, skipping upgrade: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp==2.0.2) (3.9.9)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp==2.0.2) (2019.9.11)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp==2.0.2) (2.8)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp==2.0.2) (1.25.6)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp==2.0.2) (3.0.4)
    Requirement already satisfied, skipping upgrade: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp==2.0.2) (2.2.0)
    Requirement already satisfied, skipping upgrade: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp==2.0.2) (0.23)
    Requirement already satisfied, skipping upgrade: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp==2.0.2) (2.6.0)
    Requirement already satisfied, skipping upgrade: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp==2.0.2) (0.6.1)
    Requirement already satisfied, skipping upgrade: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (1.3.4)
    Requirement already satisfied, skipping upgrade: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (0.10.0)
    Requirement already satisfied, skipping upgrade: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (1.3.0)
    Requirement already satisfied, skipping upgrade: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (5.1.2)
    Requirement already satisfied, skipping upgrade: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (16.7.9)
    Requirement already satisfied, skipping upgrade: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (2.0.1)
    Requirement already satisfied, skipping upgrade: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp==2.0.2) (1.4.10)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.5->Flask-Babel>=1.0.0->visualdl->paddlenlp==2.0.2) (1.1.1)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl->paddlenlp==2.0.2) (0.6.0)
    Requirement already satisfied, skipping upgrade: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl->paddlenlp==2.0.2) (7.2.0)
    Installing collected packages: paddlenlp
      Found existing installation: paddlenlp 2.0.1
        Uninstalling paddlenlp-2.0.1:
          Successfully uninstalled paddlenlp-2.0.1
    Successfully installed paddlenlp-2.0.2



```python
import time
import os
import numpy as np

import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
import paddlenlp

# 一键加载 Lcqmc 的训练集、验证集
train_ds, dev_ds = load_dataset("lcqmc", splits=["train", "dev"])
```

    100%|██████████| 6827/6827 [00:00<00:00, 51482.40it/s]


## 2.数据下载


```python
# paddlenlp 会自动下载 lcqmc 数据集解压到 "${HOME}/.paddlenlp/datasets/LCQMC/lcqmc/lcqmc/" 目录下
! ls ${HOME}/.paddlenlp/datasets/LCQMC/lcqmc/lcqmc
print(paddlenlp.__version__)
```

    dev.tsv  License.pdf  test.tsv	train.tsv  User_Agreement.pdf
    2.0.2


## 3.数据查看


```python
# 输出训练集的前 20 条样本
for idx, example in enumerate(train_ds):
    if idx <= 20:
        print(example)
```

    {'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生', 'label': 1}
    {'query': '我手机丢了，我想换个手机', 'title': '我想买个新手机，求推荐', 'label': 1}
    {'query': '大家觉得她好看吗', 'title': '大家觉得跑男好看吗？', 'label': 0}
    {'query': '求秋色之空漫画全集', 'title': '求秋色之空全集漫画', 'label': 1}
    {'query': '晚上睡觉带着耳机听音乐有什么害处吗？', 'title': '孕妇可以戴耳机听音乐吗?', 'label': 0}
    {'query': '学日语软件手机上的', 'title': '手机学日语的软件', 'label': 1}
    {'query': '打印机和电脑怎样连接，该如何设置', 'title': '如何把带无线的电脑连接到打印机上', 'label': 0}
    {'query': '侠盗飞车罪恶都市怎样改车', 'title': '侠盗飞车罪恶都市怎么改车', 'label': 1}
    {'query': '什么花一年四季都开', 'title': '什么花一年四季都是开的', 'label': 1}
    {'query': '看图猜一电影名', 'title': '看图猜电影！', 'label': 1}
    {'query': '这上面写的是什么？', 'title': '胃上面是什么', 'label': 0}
    {'query': '建议您重新注册，辛苦您了。', 'title': '可以的，您注销成功后，可以重新注册的，辛苦您了。', 'label': 0}
    {'query': '小草有什么的特点,可以象征什么?', 'title': '小草有什么特点可以象征什么', 'label': 1}
    {'query': '校验失败了，', 'title': '您好，您还是访客状态呢', 'label': 0}
    {'query': '尼玛什么意思', 'title': '尼玛啊是什么意思？', 'label': 0}
    {'query': '自找苦吃的地方是哪儿？', 'title': '自找苦吃的地方是哪儿', 'label': 1}
    {'query': '尾号4位多少', 'title': '尾号是多少后4位', 'label': 1}
    {'query': '谢文东能在哪里看', 'title': '谢文东在哪里看', 'label': 1}
    {'query': '新概念英语第二册练习册41课答案', 'title': '新概念英语第二册练习册21练习答案', 'label': 0}
    {'query': '过年送礼送什么好', 'title': '过年前什么时候送礼？', 'label': 0}
    {'query': '壮丁是什么生肖', 'title': '欲钱买壮丁指的是什么生肖？', 'label': 0}


## 4.数据预处理
通过 paddlenlp 加载进来的 LCQMC 数据集是原始的明文数据集，这部分我们来实现组 batch、tokenize 等预处理逻辑，将原始明文数据转换成网络训练的输入数据

### 4.1定义样本转换函数


```python
# 因为是基于预训练模型 ERNIE-Gram 来进行，所以需要首先加载 ERNIE-Gram 的 tokenizer，
# 后续样本转换函数基于 tokenizer 对文本进行切分

# tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained('ernie-gram-zh')

tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')
# tokenizer = paddlenlp.transformers.NeZhaTokenizer.from_pretrained('nezha-large-wwm-chinese')
```

    [2021-06-15 13:39:36,356] [    INFO] - Downloading vocab.txt from https://paddlenlp.bj.bcebos.com/models/transformers/ernie/vocab.txt
    100%|██████████| 90/90 [00:00<00:00, 3296.23it/s]



```python
# 将 1 条明文数据的 query、title 拼接起来，根据预训练模型的 tokenizer 将明文转换为 ID 数据
# 返回 input_ids 和 token_type_ids

def convert_example(example, tokenizer, max_seq_length=512, is_test=False):

    query, title = example["query"], example["title"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    # 在预测或者评估阶段，不返回 label 字段
    else:
        return input_ids, token_type_ids
```


```python
### 对训练集的第 1 条数据进行转换
input_ids, token_type_ids, label = convert_example(train_ds[0], tokenizer)
```


```python
print(input_ids)
```

    [1, 692, 811, 445, 2001, 497, 5, 654, 21, 692, 811, 614, 356, 314, 5, 291, 21, 2, 329, 445, 2001, 497, 5, 654, 21, 692, 811, 614, 356, 314, 5, 291, 21, 2]



```python
print(token_type_ids)
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]



```python
# 为了后续方便使用，我们给 convert_example 赋予一些默认参数
from functools import partial

# 训练集和验证集的样本转换函数
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=512)
```

### 4.2 组装 Batch 数据 & Padding
上一小节，我们完成了对单条样本的转换，本节我们需要将样本组合成 Batch 数据，对于不等长的数据还需要进行 Padding 操作，便于 GPU 训练。

PaddleNLP 提供了许多关于 NLP 任务中构建有效的数据 pipeline 的常用 API


```python
from paddlenlp.data import Stack, Pad, Tuple
# 我们的训练数据会返回 input_ids, token_type_ids, labels 3 个字段
# 因此针对这 3 个字段需要分别定义 3 个组 batch 操作
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]
```

### 4.3 定义 Dataloader
下面我们基于组 batchify_fn 函数和样本转换函数 trans_func 来构造训练集的 DataLoader, 支持多卡训练


```python
# 定义分布式 Sampler: 自动对训练数据进行切分，支持多卡并行训练
batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=200, shuffle=True) # batch_size=32

# 基于 train_ds 定义 train_data_loader
# 因为我们使用了分布式的 DistributedBatchSampler, train_data_loader 会自动对训练数据进行切分
train_data_loader = paddle.io.DataLoader(
        dataset=train_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

# 针对验证集数据加载，我们使用单卡进行评估，所以采用 paddle.io.BatchSampler 即可
# 定义 dev_data_loader
batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=200, shuffle=False)
dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
```

# 四、模型训练
## 1.模型搭建
自从 2018 年 10 月以来，NLP 个领域的任务都通过 Pretrain + Finetune 的模式相比传统 DNN 方法在效果上取得了显著的提升，本节我们以百度开源的预训练模型 ERNIE-Gram 为基础模型，在此之上构建 Point-wise 语义匹配网络。


```python
import paddle.nn as nn

# 我们基于 ERNIE-Gram 模型结构搭建 Point-wise 语义匹配网络
# 所以此处先定义 ERNIE-Gram 的 pretrained_model
# pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')
pretrained_model = paddlenlp.transformers.ErnieModel.from_pretrained('ernie-1.0')
# pretrained_model = paddlenlp.transformers.NeZhaModel.from_pretrained('nezha-large-wwm-chinese')



class PointwiseMatching(nn.Layer):
   
    # 此处的 pretained_model 在本例中会被 ERNIE-Gram 预训练模型初始化
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.07) #0.1

        # 语义匹配任务: 相似、不相似 2 分类任务
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        # 此处的 Input_ids 由两条文本的 token ids 拼接而成
        # token_type_ids 表示两段文本的类型编码
        # 返回的 cls_embedding 就表示这两段文本经过模型的计算之后而得到的语义表示向量
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)


        cls_embedding = self.dropout(cls_embedding)

        # 基于文本对的语义表示向量进行 2 分类任务
        logits = self.classifier(cls_embedding)
        probs = F.softmax(logits)

        return probs

# 定义 Point-wise 语义匹配网络
model = PointwiseMatching(pretrained_model)
```

    [2021-06-15 13:39:36,533] [    INFO] - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/ernie/ernie_v1_chn_base.pdparams and saved to /home/aistudio/.paddlenlp/models/ernie-1.0
    [2021-06-15 13:39:36,536] [    INFO] - Downloading ernie_v1_chn_base.pdparams from https://paddlenlp.bj.bcebos.com/models/transformers/ernie/ernie_v1_chn_base.pdparams
    100%|██████████| 392507/392507 [00:07<00:00, 53183.89it/s]
    [2021-06-15 13:39:50,553] [    INFO] - Weights from pretrained model not used in ErnieModel: ['cls.predictions.layer_norm.weight', 'cls.predictions.decoder_bias', 'cls.predictions.transform.bias', 'cls.predictions.transform.weight', 'cls.predictions.layer_norm.bias']


## 2. 模型训练（引入visual dl）


```python
from paddlenlp.transformers import LinearDecayWithWarmup

epochs = 100
num_training_steps = len(train_data_loader) * epochs

# 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
lr_scheduler = LinearDecayWithWarmup(5E-5, num_training_steps, 0.0)

# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 定义 Optimizer
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=0.0,
    apply_decay_param_fun=lambda x: x in decay_params)

# 采用交叉熵 损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()

# 评估的时候采用准确率指标
metric = paddle.metric.Accuracy()
```


```python
# 加入日志显示
from visualdl import LogWriter

writer = LogWriter("./log")
```


```python
# 因为训练过程中同时要在验证集进行模型评估，因此我们先定义评估函数

@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, phase="dev"):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        # probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(probs, labels)
        losses.append(loss.numpy())
        correct = metric.compute(probs, labels)
        metric.update(correct)
        accu = metric.accumulate()
        tmpacc=accu
    print("eval {} loss: {:.5}, accu: {:.5}".format(phase,
                                                    np.mean(losses), accu))
    model.train()
    metric.reset()
    return np.mean(losses), accu
```


```python
# 接下来，开始正式训练模型

global_step = 0
tic_train = time.time()
best_val_acc=0
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):

        input_ids, token_type_ids, labels = batch
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(probs, labels)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        
        # 每间隔 10 step 输出训练指标
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()


        # 每间隔 200 step 在验证集和测试集上进行评估
        if global_step %100 == 0:
            eval_loss, eval_accu=evaluate(model, criterion, metric, dev_data_loader, "dev")
            # 加入eval日志显示
            writer.add_scalar(tag="eval/loss", step=global_step, value=eval_loss)
            writer.add_scalar(tag="eval/acc", step=global_step, value=eval_accu)    
        
        # 加入train日志显示
            writer.add_scalar(tag="train/loss", step=global_step, value=loss)
            writer.add_scalar(tag="train/acc", step=global_step, value=acc)
            
            save_dir = "checkpoint"
            # 加入保存       
            if eval_accu>best_val_acc:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                best_val_acc=eval_accu
                print(f"模型保存在 {global_step} 步， 最佳eval准确度为{best_val_acc:.8f}！")
                save_param_path = os.path.join(save_dir, 'best_model.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                fh = open('checkpoint/best_model.txt', 'w', encoding='utf-8')
                fh.write(f"模型保存在 {global_step} 步， 最佳eval准确度为{best_val_acc:.8f}！")
                fh.close()
```

    global step 10, epoch: 1, batch: 10, loss: 0.55927, accu: 0.61200, speed: 1.53 step/s
    global step 20, epoch: 1, batch: 20, loss: 0.45002, accu: 0.73675, speed: 1.55 step/s
    global step 30, epoch: 1, batch: 30, loss: 0.40220, accu: 0.78633, speed: 1.54 step/s
    global step 40, epoch: 1, batch: 40, loss: 0.43648, accu: 0.80875, speed: 1.51 step/s
    global step 50, epoch: 1, batch: 50, loss: 0.39423, accu: 0.82470, speed: 1.46 step/s
    global step 60, epoch: 1, batch: 60, loss: 0.43224, accu: 0.83667, speed: 1.54 step/s
    global step 70, epoch: 1, batch: 70, loss: 0.39952, accu: 0.84457, speed: 1.54 step/s
    global step 80, epoch: 1, batch: 80, loss: 0.38990, accu: 0.85175, speed: 1.51 step/s
    global step 90, epoch: 1, batch: 90, loss: 0.39998, accu: 0.85694, speed: 1.55 step/s
    global step 100, epoch: 1, batch: 100, loss: 0.40286, accu: 0.86130, speed: 1.55 step/s
    eval dev loss: 0.47756, accu: 0.82402
    模型保存在 100 步， 最佳eval准确度为0.82401727！
    global step 110, epoch: 1, batch: 110, loss: 0.42226, accu: 0.88400, speed: 0.47 step/s
    global step 120, epoch: 1, batch: 120, loss: 0.44029, accu: 0.89050, speed: 1.51 step/s
    global step 130, epoch: 1, batch: 130, loss: 0.40722, accu: 0.88933, speed: 1.50 step/s
    global step 140, epoch: 1, batch: 140, loss: 0.41380, accu: 0.89000, speed: 1.53 step/s
    global step 150, epoch: 1, batch: 150, loss: 0.40734, accu: 0.89090, speed: 1.55 step/s
    global step 160, epoch: 1, batch: 160, loss: 0.39751, accu: 0.89183, speed: 1.54 step/s
    global step 170, epoch: 1, batch: 170, loss: 0.39328, accu: 0.89164, speed: 1.34 step/s
    global step 180, epoch: 1, batch: 180, loss: 0.38559, accu: 0.89463, speed: 1.52 step/s
    global step 190, epoch: 1, batch: 190, loss: 0.41626, accu: 0.89600, speed: 1.49 step/s
    global step 200, epoch: 1, batch: 200, loss: 0.43215, accu: 0.89635, speed: 1.48 step/s
    eval dev loss: 0.47762, accu: 0.8272
    模型保存在 200 步， 最佳eval准确度为0.82719836！
    global step 210, epoch: 1, batch: 210, loss: 0.41650, accu: 0.89200, speed: 0.45 step/s
    global step 220, epoch: 1, batch: 220, loss: 0.41674, accu: 0.89300, speed: 1.52 step/s
    global step 230, epoch: 1, batch: 230, loss: 0.41376, accu: 0.89667, speed: 1.49 step/s
    global step 240, epoch: 1, batch: 240, loss: 0.37856, accu: 0.89750, speed: 1.51 step/s
    global step 250, epoch: 1, batch: 250, loss: 0.39733, accu: 0.90010, speed: 1.54 step/s
    global step 260, epoch: 1, batch: 260, loss: 0.40637, accu: 0.89867, speed: 1.50 step/s
    global step 270, epoch: 1, batch: 270, loss: 0.40310, accu: 0.90057, speed: 1.48 step/s
    global step 280, epoch: 1, batch: 280, loss: 0.40721, accu: 0.90019, speed: 1.48 step/s
    global step 290, epoch: 1, batch: 290, loss: 0.40230, accu: 0.90017, speed: 1.53 step/s
    global step 300, epoch: 1, batch: 300, loss: 0.39265, accu: 0.90035, speed: 1.54 step/s
    eval dev loss: 0.44257, accu: 0.86128
    模型保存在 300 步， 最佳eval准确度为0.86128153！
    global step 310, epoch: 1, batch: 310, loss: 0.43021, accu: 0.89200, speed: 0.46 step/s
    global step 320, epoch: 1, batch: 320, loss: 0.40045, accu: 0.89950, speed: 1.52 step/s
    global step 330, epoch: 1, batch: 330, loss: 0.39953, accu: 0.90133, speed: 1.49 step/s
    global step 340, epoch: 1, batch: 340, loss: 0.39352, accu: 0.90225, speed: 1.53 step/s
    global step 350, epoch: 1, batch: 350, loss: 0.39910, accu: 0.90220, speed: 1.49 step/s
    global step 360, epoch: 1, batch: 360, loss: 0.41054, accu: 0.90083, speed: 1.38 step/s
    global step 370, epoch: 1, batch: 370, loss: 0.41669, accu: 0.90071, speed: 1.54 step/s
    global step 380, epoch: 1, batch: 380, loss: 0.41449, accu: 0.90031, speed: 1.49 step/s
    global step 390, epoch: 1, batch: 390, loss: 0.38298, accu: 0.90117, speed: 1.48 step/s
    global step 400, epoch: 1, batch: 400, loss: 0.42166, accu: 0.90250, speed: 1.57 step/s
    eval dev loss: 0.45521, accu: 0.84878
    global step 410, epoch: 1, batch: 410, loss: 0.37910, accu: 0.90250, speed: 0.59 step/s
    global step 420, epoch: 1, batch: 420, loss: 0.38127, accu: 0.90825, speed: 1.54 step/s
    global step 430, epoch: 1, batch: 430, loss: 0.38075, accu: 0.90850, speed: 1.54 step/s
    global step 440, epoch: 1, batch: 440, loss: 0.40869, accu: 0.90775, speed: 1.51 step/s
    global step 450, epoch: 1, batch: 450, loss: 0.39471, accu: 0.90980, speed: 1.53 step/s
    global step 460, epoch: 1, batch: 460, loss: 0.41130, accu: 0.90983, speed: 1.53 step/s
    global step 470, epoch: 1, batch: 470, loss: 0.40753, accu: 0.91121, speed: 1.49 step/s
    global step 480, epoch: 1, batch: 480, loss: 0.37481, accu: 0.91100, speed: 1.58 step/s
    global step 490, epoch: 1, batch: 490, loss: 0.42214, accu: 0.91106, speed: 1.46 step/s
    global step 500, epoch: 1, batch: 500, loss: 0.39688, accu: 0.91020, speed: 1.54 step/s
    eval dev loss: 0.45693, accu: 0.84651
    global step 510, epoch: 1, batch: 510, loss: 0.41445, accu: 0.90950, speed: 0.60 step/s
    global step 520, epoch: 1, batch: 520, loss: 0.39528, accu: 0.91075, speed: 1.56 step/s
    global step 530, epoch: 1, batch: 530, loss: 0.39704, accu: 0.91033, speed: 1.49 step/s
    global step 540, epoch: 1, batch: 540, loss: 0.38489, accu: 0.91025, speed: 1.36 step/s
    global step 550, epoch: 1, batch: 550, loss: 0.40745, accu: 0.91310, speed: 1.48 step/s
    global step 560, epoch: 1, batch: 560, loss: 0.39164, accu: 0.91350, speed: 1.42 step/s
    global step 570, epoch: 1, batch: 570, loss: 0.39892, accu: 0.91371, speed: 1.51 step/s
    global step 580, epoch: 1, batch: 580, loss: 0.41857, accu: 0.91350, speed: 1.59 step/s
    global step 590, epoch: 1, batch: 590, loss: 0.41095, accu: 0.91306, speed: 1.49 step/s
    global step 600, epoch: 1, batch: 600, loss: 0.37011, accu: 0.91320, speed: 1.50 step/s
    eval dev loss: 0.45457, accu: 0.84924
    global step 610, epoch: 1, batch: 610, loss: 0.39869, accu: 0.90950, speed: 0.59 step/s
    global step 620, epoch: 1, batch: 620, loss: 0.37111, accu: 0.91050, speed: 1.47 step/s
    global step 630, epoch: 1, batch: 630, loss: 0.38497, accu: 0.91567, speed: 1.51 step/s
    global step 640, epoch: 1, batch: 640, loss: 0.39871, accu: 0.91587, speed: 1.55 step/s
    global step 650, epoch: 1, batch: 650, loss: 0.40798, accu: 0.91250, speed: 1.55 step/s
    global step 660, epoch: 1, batch: 660, loss: 0.37312, accu: 0.91442, speed: 1.54 step/s
    global step 670, epoch: 1, batch: 670, loss: 0.42369, accu: 0.91357, speed: 1.51 step/s
    global step 680, epoch: 1, batch: 680, loss: 0.42630, accu: 0.91300, speed: 1.50 step/s
    global step 690, epoch: 1, batch: 690, loss: 0.41396, accu: 0.91244, speed: 1.45 step/s
    global step 700, epoch: 1, batch: 700, loss: 0.41857, accu: 0.91270, speed: 1.54 step/s
    eval dev loss: 0.44714, accu: 0.8564
    global step 710, epoch: 1, batch: 710, loss: 0.43590, accu: 0.90900, speed: 0.57 step/s
    global step 720, epoch: 1, batch: 720, loss: 0.39676, accu: 0.91000, speed: 1.44 step/s
    global step 730, epoch: 1, batch: 730, loss: 0.38611, accu: 0.91517, speed: 1.53 step/s
    global step 740, epoch: 1, batch: 740, loss: 0.40781, accu: 0.91525, speed: 1.59 step/s
    global step 750, epoch: 1, batch: 750, loss: 0.42281, accu: 0.91480, speed: 1.52 step/s
    global step 760, epoch: 1, batch: 760, loss: 0.39892, accu: 0.91517, speed: 1.43 step/s
    global step 770, epoch: 1, batch: 770, loss: 0.40465, accu: 0.91314, speed: 1.53 step/s
    global step 780, epoch: 1, batch: 780, loss: 0.38713, accu: 0.91425, speed: 1.48 step/s
    global step 790, epoch: 1, batch: 790, loss: 0.39440, accu: 0.91439, speed: 1.49 step/s
    global step 800, epoch: 1, batch: 800, loss: 0.40207, accu: 0.91385, speed: 1.58 step/s
    eval dev loss: 0.45289, accu: 0.85242
    global step 810, epoch: 1, batch: 810, loss: 0.39810, accu: 0.91250, speed: 0.58 step/s
    global step 820, epoch: 1, batch: 820, loss: 0.38296, accu: 0.91375, speed: 1.49 step/s
    global step 830, epoch: 1, batch: 830, loss: 0.40718, accu: 0.91367, speed: 1.52 step/s
    global step 840, epoch: 1, batch: 840, loss: 0.37438, accu: 0.91425, speed: 1.53 step/s
    global step 850, epoch: 1, batch: 850, loss: 0.36066, accu: 0.91490, speed: 1.66 step/s
    global step 860, epoch: 1, batch: 860, loss: 0.37241, accu: 0.91575, speed: 1.55 step/s
    global step 870, epoch: 1, batch: 870, loss: 0.41526, accu: 0.91521, speed: 1.48 step/s
    global step 880, epoch: 1, batch: 880, loss: 0.39890, accu: 0.91500, speed: 1.52 step/s
    global step 890, epoch: 1, batch: 890, loss: 0.38569, accu: 0.91539, speed: 1.43 step/s
    global step 900, epoch: 1, batch: 900, loss: 0.38990, accu: 0.91475, speed: 1.46 step/s
    eval dev loss: 0.45034, accu: 0.85617
    global step 910, epoch: 1, batch: 910, loss: 0.40836, accu: 0.91400, speed: 0.59 step/s
    global step 920, epoch: 1, batch: 920, loss: 0.39080, accu: 0.91450, speed: 1.49 step/s
    global step 930, epoch: 1, batch: 930, loss: 0.42781, accu: 0.91533, speed: 1.50 step/s
    global step 940, epoch: 1, batch: 940, loss: 0.39237, accu: 0.91700, speed: 1.52 step/s
    global step 950, epoch: 1, batch: 950, loss: 0.40112, accu: 0.91800, speed: 1.47 step/s
    global step 960, epoch: 1, batch: 960, loss: 0.42229, accu: 0.91708, speed: 1.50 step/s
    global step 970, epoch: 1, batch: 970, loss: 0.41169, accu: 0.91657, speed: 1.51 step/s
    global step 980, epoch: 1, batch: 980, loss: 0.42340, accu: 0.91612, speed: 1.52 step/s
    global step 990, epoch: 1, batch: 990, loss: 0.39019, accu: 0.91561, speed: 1.60 step/s
    global step 1000, epoch: 1, batch: 1000, loss: 0.38822, accu: 0.91485, speed: 1.54 step/s
    eval dev loss: 0.44509, accu: 0.85969
    global step 1010, epoch: 1, batch: 1010, loss: 0.38983, accu: 0.92350, speed: 0.58 step/s
    global step 1020, epoch: 1, batch: 1020, loss: 0.39013, accu: 0.91900, speed: 1.50 step/s
    global step 1030, epoch: 1, batch: 1030, loss: 0.40220, accu: 0.91650, speed: 1.46 step/s
    global step 1040, epoch: 1, batch: 1040, loss: 0.37968, accu: 0.91687, speed: 1.52 step/s
    global step 1050, epoch: 1, batch: 1050, loss: 0.37212, accu: 0.91570, speed: 1.61 step/s
    global step 1060, epoch: 1, batch: 1060, loss: 0.37968, accu: 0.91525, speed: 1.33 step/s
    global step 1070, epoch: 1, batch: 1070, loss: 0.39186, accu: 0.91686, speed: 1.53 step/s
    global step 1080, epoch: 1, batch: 1080, loss: 0.38139, accu: 0.91856, speed: 1.50 step/s
    global step 1090, epoch: 1, batch: 1090, loss: 0.41501, accu: 0.91761, speed: 1.48 step/s
    global step 1100, epoch: 1, batch: 1100, loss: 0.37309, accu: 0.91710, speed: 1.52 step/s
    eval dev loss: 0.4321, accu: 0.87242
    模型保存在 1100 步， 最佳eval准确度为0.87241536！
    global step 1110, epoch: 1, batch: 1110, loss: 0.40310, accu: 0.91900, speed: 0.45 step/s
    global step 1120, epoch: 1, batch: 1120, loss: 0.40751, accu: 0.91400, speed: 1.50 step/s
    global step 1130, epoch: 1, batch: 1130, loss: 0.39057, accu: 0.91300, speed: 1.52 step/s
    global step 1140, epoch: 1, batch: 1140, loss: 0.38183, accu: 0.91400, speed: 1.46 step/s
    global step 1150, epoch: 1, batch: 1150, loss: 0.39050, accu: 0.91340, speed: 1.49 step/s
    global step 1160, epoch: 1, batch: 1160, loss: 0.42050, accu: 0.91425, speed: 1.57 step/s
    global step 1170, epoch: 1, batch: 1170, loss: 0.45101, accu: 0.91379, speed: 1.51 step/s
    global step 1180, epoch: 1, batch: 1180, loss: 0.38997, accu: 0.91406, speed: 1.58 step/s
    global step 1190, epoch: 1, batch: 1190, loss: 0.37371, accu: 0.91389, speed: 1.50 step/s
    global step 1200, epoch: 2, batch: 6, loss: 0.36284, accu: 0.91546, speed: 1.55 step/s
    eval dev loss: 0.42643, accu: 0.87946
    模型保存在 1200 步， 最佳eval准确度为0.87945921！
    global step 1210, epoch: 2, batch: 16, loss: 0.38161, accu: 0.91800, speed: 0.46 step/s
    global step 1220, epoch: 2, batch: 26, loss: 0.37974, accu: 0.92550, speed: 1.47 step/s
    global step 1230, epoch: 2, batch: 36, loss: 0.37207, accu: 0.92450, speed: 1.46 step/s
    global step 1240, epoch: 2, batch: 46, loss: 0.37258, accu: 0.92650, speed: 1.52 step/s
    global step 1250, epoch: 2, batch: 56, loss: 0.35081, accu: 0.92690, speed: 1.55 step/s
    global step 1260, epoch: 2, batch: 66, loss: 0.38850, accu: 0.92792, speed: 1.52 step/s
    global step 1270, epoch: 2, batch: 76, loss: 0.37272, accu: 0.92857, speed: 1.49 step/s
    global step 1280, epoch: 2, batch: 86, loss: 0.39407, accu: 0.92912, speed: 1.42 step/s
    global step 1290, epoch: 2, batch: 96, loss: 0.40960, accu: 0.92994, speed: 1.55 step/s
    global step 1300, epoch: 2, batch: 106, loss: 0.39469, accu: 0.92955, speed: 1.46 step/s
    eval dev loss: 0.43702, accu: 0.87014
    global step 1310, epoch: 2, batch: 116, loss: 0.37599, accu: 0.93000, speed: 0.59 step/s
    global step 1320, epoch: 2, batch: 126, loss: 0.39369, accu: 0.92325, speed: 1.56 step/s
    global step 1330, epoch: 2, batch: 136, loss: 0.38606, accu: 0.92083, speed: 1.57 step/s
    global step 1340, epoch: 2, batch: 146, loss: 0.36683, accu: 0.92412, speed: 1.62 step/s
    global step 1350, epoch: 2, batch: 156, loss: 0.39604, accu: 0.92600, speed: 1.59 step/s
    global step 1360, epoch: 2, batch: 166, loss: 0.37944, accu: 0.92492, speed: 1.54 step/s
    global step 1370, epoch: 2, batch: 176, loss: 0.36562, accu: 0.92593, speed: 1.48 step/s
    global step 1380, epoch: 2, batch: 186, loss: 0.39843, accu: 0.92481, speed: 1.47 step/s
    global step 1390, epoch: 2, batch: 196, loss: 0.38107, accu: 0.92611, speed: 1.47 step/s
    global step 1400, epoch: 2, batch: 206, loss: 0.38513, accu: 0.92615, speed: 1.51 step/s
    eval dev loss: 0.44261, accu: 0.86174
    global step 1410, epoch: 2, batch: 216, loss: 0.39192, accu: 0.92750, speed: 0.60 step/s
    global step 1420, epoch: 2, batch: 226, loss: 0.38695, accu: 0.92650, speed: 1.49 step/s
    global step 1430, epoch: 2, batch: 236, loss: 0.40527, accu: 0.92450, speed: 1.51 step/s
    global step 1440, epoch: 2, batch: 246, loss: 0.38349, accu: 0.92288, speed: 1.51 step/s
    global step 1450, epoch: 2, batch: 256, loss: 0.38075, accu: 0.92430, speed: 1.35 step/s
    global step 1460, epoch: 2, batch: 266, loss: 0.36111, accu: 0.92625, speed: 1.53 step/s
    global step 1470, epoch: 2, batch: 276, loss: 0.36870, accu: 0.92614, speed: 1.54 step/s
    global step 1480, epoch: 2, batch: 286, loss: 0.37553, accu: 0.92669, speed: 1.49 step/s
    global step 1490, epoch: 2, batch: 296, loss: 0.38992, accu: 0.92772, speed: 1.47 step/s
    global step 1500, epoch: 2, batch: 306, loss: 0.40250, accu: 0.92720, speed: 1.53 step/s
    eval dev loss: 0.43651, accu: 0.86923
    global step 1510, epoch: 2, batch: 316, loss: 0.39067, accu: 0.93300, speed: 0.59 step/s
    global step 1520, epoch: 2, batch: 326, loss: 0.39857, accu: 0.93125, speed: 1.52 step/s
    global step 1530, epoch: 2, batch: 336, loss: 0.38602, accu: 0.93017, speed: 1.52 step/s
    global step 1540, epoch: 2, batch: 346, loss: 0.41213, accu: 0.92963, speed: 1.56 step/s
    global step 1550, epoch: 2, batch: 356, loss: 0.39912, accu: 0.92910, speed: 1.46 step/s
    global step 1560, epoch: 2, batch: 366, loss: 0.36847, accu: 0.92867, speed: 1.49 step/s
    global step 1570, epoch: 2, batch: 376, loss: 0.36586, accu: 0.92957, speed: 1.48 step/s
    global step 1580, epoch: 2, batch: 386, loss: 0.35957, accu: 0.92869, speed: 1.56 step/s
    global step 1590, epoch: 2, batch: 396, loss: 0.39730, accu: 0.92800, speed: 1.47 step/s
    global step 1600, epoch: 2, batch: 406, loss: 0.38699, accu: 0.92675, speed: 1.55 step/s
    eval dev loss: 0.44158, accu: 0.8631
    global step 1610, epoch: 2, batch: 416, loss: 0.39576, accu: 0.93300, speed: 0.59 step/s
    global step 1620, epoch: 2, batch: 426, loss: 0.37555, accu: 0.93075, speed: 1.55 step/s
    global step 1630, epoch: 2, batch: 436, loss: 0.42451, accu: 0.92400, speed: 1.47 step/s
    global step 1640, epoch: 2, batch: 446, loss: 0.37433, accu: 0.92312, speed: 1.49 step/s
    global step 1650, epoch: 2, batch: 456, loss: 0.37456, accu: 0.92450, speed: 1.53 step/s
    global step 1660, epoch: 2, batch: 466, loss: 0.39618, accu: 0.92367, speed: 1.46 step/s
    global step 1670, epoch: 2, batch: 476, loss: 0.37087, accu: 0.92379, speed: 1.47 step/s
    global step 1680, epoch: 2, batch: 486, loss: 0.35047, accu: 0.92463, speed: 1.52 step/s
    global step 1690, epoch: 2, batch: 496, loss: 0.39081, accu: 0.92494, speed: 1.49 step/s
    global step 1700, epoch: 2, batch: 506, loss: 0.37495, accu: 0.92580, speed: 1.43 step/s
    eval dev loss: 0.42823, accu: 0.87798
    global step 1710, epoch: 2, batch: 516, loss: 0.37251, accu: 0.93550, speed: 0.58 step/s
    global step 1720, epoch: 2, batch: 526, loss: 0.40298, accu: 0.93150, speed: 1.50 step/s
    global step 1730, epoch: 2, batch: 536, loss: 0.42429, accu: 0.92800, speed: 1.38 step/s
    global step 1740, epoch: 2, batch: 546, loss: 0.37076, accu: 0.92600, speed: 1.52 step/s
    global step 1750, epoch: 2, batch: 556, loss: 0.41379, accu: 0.92450, speed: 1.50 step/s
    global step 1760, epoch: 2, batch: 566, loss: 0.40013, accu: 0.92350, speed: 1.47 step/s
    global step 1770, epoch: 2, batch: 576, loss: 0.38375, accu: 0.92429, speed: 1.46 step/s
    global step 1780, epoch: 2, batch: 586, loss: 0.36462, accu: 0.92412, speed: 1.49 step/s
    global step 1790, epoch: 2, batch: 596, loss: 0.37884, accu: 0.92378, speed: 1.46 step/s
    global step 1800, epoch: 2, batch: 606, loss: 0.37829, accu: 0.92365, speed: 1.54 step/s
    eval dev loss: 0.43281, accu: 0.87242
    global step 1810, epoch: 2, batch: 616, loss: 0.40508, accu: 0.92650, speed: 0.59 step/s
    global step 1820, epoch: 2, batch: 626, loss: 0.36056, accu: 0.92425, speed: 1.60 step/s
    global step 1830, epoch: 2, batch: 636, loss: 0.41733, accu: 0.92400, speed: 1.50 step/s
    global step 1840, epoch: 2, batch: 646, loss: 0.37600, accu: 0.92688, speed: 1.46 step/s
    global step 1850, epoch: 2, batch: 656, loss: 0.38907, accu: 0.92520, speed: 1.44 step/s
    global step 1860, epoch: 2, batch: 666, loss: 0.36955, accu: 0.92442, speed: 1.45 step/s
    global step 1870, epoch: 2, batch: 676, loss: 0.38263, accu: 0.92207, speed: 1.53 step/s
    global step 1880, epoch: 2, batch: 686, loss: 0.38497, accu: 0.92219, speed: 1.55 step/s
    global step 1890, epoch: 2, batch: 696, loss: 0.38157, accu: 0.92322, speed: 1.52 step/s
    global step 1900, epoch: 2, batch: 706, loss: 0.36460, accu: 0.92425, speed: 1.50 step/s
    eval dev loss: 0.43259, accu: 0.8748
    global step 1910, epoch: 2, batch: 716, loss: 0.35524, accu: 0.94400, speed: 0.60 step/s
    global step 1920, epoch: 2, batch: 726, loss: 0.38695, accu: 0.93150, speed: 1.52 step/s
    global step 1930, epoch: 2, batch: 736, loss: 0.36272, accu: 0.93083, speed: 1.47 step/s
    global step 1940, epoch: 2, batch: 746, loss: 0.39484, accu: 0.92912, speed: 1.63 step/s
    global step 1950, epoch: 2, batch: 756, loss: 0.38748, accu: 0.92820, speed: 1.60 step/s
    global step 1960, epoch: 2, batch: 766, loss: 0.39207, accu: 0.92908, speed: 1.48 step/s
    global step 1970, epoch: 2, batch: 776, loss: 0.37150, accu: 0.92879, speed: 1.50 step/s
    global step 1980, epoch: 2, batch: 786, loss: 0.38373, accu: 0.92819, speed: 1.50 step/s
    global step 1990, epoch: 2, batch: 796, loss: 0.35864, accu: 0.92750, speed: 1.49 step/s
    global step 2000, epoch: 2, batch: 806, loss: 0.36082, accu: 0.92790, speed: 1.48 step/s
    eval dev loss: 0.43659, accu: 0.86935
    global step 2010, epoch: 2, batch: 816, loss: 0.39817, accu: 0.92900, speed: 0.58 step/s
    global step 2020, epoch: 2, batch: 826, loss: 0.39527, accu: 0.92750, speed: 1.52 step/s
    global step 2030, epoch: 2, batch: 836, loss: 0.37368, accu: 0.92917, speed: 1.54 step/s
    global step 2040, epoch: 2, batch: 846, loss: 0.41305, accu: 0.92875, speed: 1.51 step/s
    global step 2050, epoch: 2, batch: 856, loss: 0.38985, accu: 0.92710, speed: 1.40 step/s
    global step 2060, epoch: 2, batch: 866, loss: 0.36330, accu: 0.92783, speed: 1.48 step/s
    global step 2070, epoch: 2, batch: 876, loss: 0.39412, accu: 0.92707, speed: 1.48 step/s
    global step 2080, epoch: 2, batch: 886, loss: 0.35765, accu: 0.92719, speed: 1.54 step/s
    global step 2090, epoch: 2, batch: 896, loss: 0.39354, accu: 0.92689, speed: 1.55 step/s
    global step 2100, epoch: 2, batch: 906, loss: 0.38777, accu: 0.92630, speed: 1.34 step/s
    eval dev loss: 0.43694, accu: 0.8698
    global step 2110, epoch: 2, batch: 916, loss: 0.37692, accu: 0.92000, speed: 0.59 step/s
    global step 2120, epoch: 2, batch: 926, loss: 0.41645, accu: 0.91725, speed: 1.55 step/s
    global step 2130, epoch: 2, batch: 936, loss: 0.41213, accu: 0.91817, speed: 1.53 step/s
    global step 2140, epoch: 2, batch: 946, loss: 0.40499, accu: 0.91987, speed: 1.51 step/s
    global step 2150, epoch: 2, batch: 956, loss: 0.38145, accu: 0.92140, speed: 1.57 step/s
    global step 2160, epoch: 2, batch: 966, loss: 0.37325, accu: 0.92208, speed: 1.53 step/s
    global step 2170, epoch: 2, batch: 976, loss: 0.37907, accu: 0.92314, speed: 1.54 step/s
    global step 2180, epoch: 2, batch: 986, loss: 0.39771, accu: 0.92275, speed: 1.49 step/s
    global step 2190, epoch: 2, batch: 996, loss: 0.36795, accu: 0.92194, speed: 1.49 step/s
    global step 2200, epoch: 2, batch: 1006, loss: 0.38210, accu: 0.92410, speed: 1.58 step/s
    eval dev loss: 0.43505, accu: 0.87219
    global step 2210, epoch: 2, batch: 1016, loss: 0.42190, accu: 0.92050, speed: 0.59 step/s
    global step 2220, epoch: 2, batch: 1026, loss: 0.39137, accu: 0.92500, speed: 1.44 step/s
    global step 2230, epoch: 2, batch: 1036, loss: 0.39337, accu: 0.92533, speed: 1.57 step/s
    global step 2240, epoch: 2, batch: 1046, loss: 0.37963, accu: 0.92737, speed: 1.48 step/s
    global step 2250, epoch: 2, batch: 1056, loss: 0.36758, accu: 0.92680, speed: 1.48 step/s
    global step 2260, epoch: 2, batch: 1066, loss: 0.37390, accu: 0.92758, speed: 1.49 step/s
    global step 2270, epoch: 2, batch: 1076, loss: 0.40220, accu: 0.92750, speed: 1.55 step/s
    global step 2280, epoch: 2, batch: 1086, loss: 0.38150, accu: 0.92688, speed: 1.52 step/s
    global step 2290, epoch: 2, batch: 1096, loss: 0.37348, accu: 0.92711, speed: 1.53 step/s
    global step 2300, epoch: 2, batch: 1106, loss: 0.38637, accu: 0.92720, speed: 1.44 step/s
    eval dev loss: 0.43496, accu: 0.87139
    global step 2310, epoch: 2, batch: 1116, loss: 0.37873, accu: 0.93450, speed: 0.58 step/s
    global step 2320, epoch: 2, batch: 1126, loss: 0.40530, accu: 0.93100, speed: 1.53 step/s
    global step 2330, epoch: 2, batch: 1136, loss: 0.41668, accu: 0.93100, speed: 1.49 step/s
    global step 2340, epoch: 2, batch: 1146, loss: 0.40324, accu: 0.92850, speed: 1.52 step/s
    global step 2350, epoch: 2, batch: 1156, loss: 0.38993, accu: 0.92950, speed: 1.54 step/s
    global step 2360, epoch: 2, batch: 1166, loss: 0.39557, accu: 0.92808, speed: 1.49 step/s
    global step 2370, epoch: 2, batch: 1176, loss: 0.38592, accu: 0.92757, speed: 1.50 step/s
    global step 2380, epoch: 2, batch: 1186, loss: 0.38208, accu: 0.92725, speed: 1.51 step/s
    global step 2390, epoch: 3, batch: 2, loss: 0.37624, accu: 0.92703, speed: 1.52 step/s
    global step 2400, epoch: 3, batch: 12, loss: 0.36255, accu: 0.92878, speed: 1.51 step/s
    eval dev loss: 0.43308, accu: 0.87344
    global step 2410, epoch: 3, batch: 22, loss: 0.35416, accu: 0.93150, speed: 0.56 step/s
    global step 2420, epoch: 3, batch: 32, loss: 0.37620, accu: 0.93000, speed: 1.49 step/s
    global step 2430, epoch: 3, batch: 42, loss: 0.41669, accu: 0.93217, speed: 1.49 step/s
    global step 2440, epoch: 3, batch: 52, loss: 0.39688, accu: 0.93175, speed: 1.52 step/s
    global step 2480, epoch: 3, batch: 92, loss: 0.36313, accu: 0.93488, speed: 1.62 step/s
    global step 2490, epoch: 3, batch: 102, loss: 0.36965, accu: 0.93478, speed: 1.47 step/s
    global step 2500, epoch: 3, batch: 112, loss: 0.39608, accu: 0.93520, speed: 1.48 step/s
    eval dev loss: 0.42233, accu: 0.88457
    模型保存在 2500 步， 最佳eval准确度为0.88457169！
    global step 2510, epoch: 3, batch: 122, loss: 0.38996, accu: 0.93600, speed: 0.45 step/s
    global step 2520, epoch: 3, batch: 132, loss: 0.37074, accu: 0.93325, speed: 1.49 step/s
    global step 2530, epoch: 3, batch: 142, loss: 0.39092, accu: 0.93450, speed: 1.50 step/s
    global step 2540, epoch: 3, batch: 152, loss: 0.34925, accu: 0.93512, speed: 1.56 step/s
    global step 2550, epoch: 3, batch: 162, loss: 0.37017, accu: 0.93450, speed: 1.50 step/s
    global step 2560, epoch: 3, batch: 172, loss: 0.38337, accu: 0.93483, speed: 1.48 step/s
    global step 2570, epoch: 3, batch: 182, loss: 0.37674, accu: 0.93550, speed: 1.55 step/s
    global step 2580, epoch: 3, batch: 192, loss: 0.36768, accu: 0.93525, speed: 1.45 step/s
    global step 2590, epoch: 3, batch: 202, loss: 0.39090, accu: 0.93506, speed: 1.54 step/s
    global step 2600, epoch: 3, batch: 212, loss: 0.38577, accu: 0.93575, speed: 1.49 step/s
    eval dev loss: 0.42998, accu: 0.87651
    global step 2610, epoch: 3, batch: 222, loss: 0.38901, accu: 0.94150, speed: 0.59 step/s
    global step 2620, epoch: 3, batch: 232, loss: 0.38292, accu: 0.94150, speed: 1.48 step/s
    global step 2630, epoch: 3, batch: 242, loss: 0.33759, accu: 0.94300, speed: 1.46 step/s
    global step 2640, epoch: 3, batch: 252, loss: 0.39608, accu: 0.94325, speed: 1.49 step/s
    global step 2650, epoch: 3, batch: 262, loss: 0.37559, accu: 0.94310, speed: 1.57 step/s
    global step 2660, epoch: 3, batch: 272, loss: 0.38877, accu: 0.94033, speed: 1.54 step/s
    global step 2670, epoch: 3, batch: 282, loss: 0.33660, accu: 0.94014, speed: 1.45 step/s
    global step 2680, epoch: 3, batch: 292, loss: 0.37593, accu: 0.93994, speed: 1.52 step/s
    global step 2690, epoch: 3, batch: 302, loss: 0.36838, accu: 0.93972, speed: 1.49 step/s
    global step 2700, epoch: 3, batch: 312, loss: 0.37029, accu: 0.93980, speed: 1.48 step/s
    eval dev loss: 0.42591, accu: 0.88309
    global step 2710, epoch: 3, batch: 322, loss: 0.40399, accu: 0.93200, speed: 0.60 step/s
    global step 2720, epoch: 3, batch: 332, loss: 0.38051, accu: 0.93575, speed: 1.48 step/s
    global step 2730, epoch: 3, batch: 342, loss: 0.36072, accu: 0.93767, speed: 1.50 step/s
    global step 2740, epoch: 3, batch: 352, loss: 0.38421, accu: 0.93688, speed: 1.52 step/s
    global step 2750, epoch: 3, batch: 362, loss: 0.37103, accu: 0.93890, speed: 1.53 step/s
    global step 2760, epoch: 3, batch: 372, loss: 0.34278, accu: 0.93900, speed: 1.51 step/s
    global step 2770, epoch: 3, batch: 382, loss: 0.39587, accu: 0.93871, speed: 1.50 step/s
    global step 2780, epoch: 3, batch: 392, loss: 0.39233, accu: 0.93881, speed: 1.45 step/s
    global step 2790, epoch: 3, batch: 402, loss: 0.38693, accu: 0.93806, speed: 1.47 step/s
    global step 2800, epoch: 3, batch: 412, loss: 0.35395, accu: 0.93815, speed: 1.54 step/s
    eval dev loss: 0.42745, accu: 0.8806
    global step 2810, epoch: 3, batch: 422, loss: 0.38485, accu: 0.92350, speed: 0.60 step/s
    global step 2820, epoch: 3, batch: 432, loss: 0.38753, accu: 0.92425, speed: 1.54 step/s
    global step 2830, epoch: 3, batch: 442, loss: 0.36296, accu: 0.93100, speed: 1.52 step/s
    global step 2840, epoch: 3, batch: 452, loss: 0.36985, accu: 0.93212, speed: 1.50 step/s
    global step 2850, epoch: 3, batch: 462, loss: 0.37540, accu: 0.93230, speed: 1.48 step/s
    global step 2860, epoch: 3, batch: 472, loss: 0.36526, accu: 0.93492, speed: 1.55 step/s
    global step 2870, epoch: 3, batch: 482, loss: 0.38562, accu: 0.93429, speed: 1.48 step/s
    global step 2880, epoch: 3, batch: 492, loss: 0.38209, accu: 0.93456, speed: 1.53 step/s
    global step 2890, epoch: 3, batch: 502, loss: 0.36392, accu: 0.93528, speed: 1.50 step/s
    global step 2900, epoch: 3, batch: 512, loss: 0.34129, accu: 0.93625, speed: 1.48 step/s
    eval dev loss: 0.42552, accu: 0.8823
    global step 2910, epoch: 3, batch: 522, loss: 0.36945, accu: 0.93850, speed: 0.59 step/s
    global step 2920, epoch: 3, batch: 532, loss: 0.37721, accu: 0.93200, speed: 1.50 step/s
    global step 2930, epoch: 3, batch: 542, loss: 0.36858, accu: 0.93483, speed: 1.57 step/s
    global step 2940, epoch: 3, batch: 552, loss: 0.40137, accu: 0.93375, speed: 1.51 step/s
    global step 2950, epoch: 3, batch: 562, loss: 0.36643, accu: 0.93600, speed: 1.52 step/s
    global step 2960, epoch: 3, batch: 572, loss: 0.36087, accu: 0.93567, speed: 1.56 step/s
    global step 2970, epoch: 3, batch: 582, loss: 0.37896, accu: 0.93529, speed: 1.51 step/s
    global step 2980, epoch: 3, batch: 592, loss: 0.41104, accu: 0.93463, speed: 1.51 step/s
    global step 2990, epoch: 3, batch: 602, loss: 0.36314, accu: 0.93478, speed: 1.52 step/s
    global step 3000, epoch: 3, batch: 612, loss: 0.36522, accu: 0.93425, speed: 1.44 step/s
    eval dev loss: 0.42771, accu: 0.87912
    global step 3010, epoch: 3, batch: 622, loss: 0.35789, accu: 0.94300, speed: 0.59 step/s
    global step 3020, epoch: 3, batch: 632, loss: 0.39158, accu: 0.93400, speed: 1.54 step/s
    global step 3030, epoch: 3, batch: 642, loss: 0.36852, accu: 0.93317, speed: 1.50 step/s
    global step 3040, epoch: 3, batch: 652, loss: 0.37099, accu: 0.93675, speed: 1.47 step/s
    global step 3050, epoch: 3, batch: 662, loss: 0.35942, accu: 0.93700, speed: 1.43 step/s
    global step 3060, epoch: 3, batch: 672, loss: 0.38116, accu: 0.93625, speed: 1.55 step/s
    global step 3070, epoch: 3, batch: 682, loss: 0.38340, accu: 0.93450, speed: 1.46 step/s
    global step 3080, epoch: 3, batch: 692, loss: 0.39638, accu: 0.93394, speed: 1.51 step/s
    global step 3090, epoch: 3, batch: 702, loss: 0.33417, accu: 0.93472, speed: 1.53 step/s
    global step 3100, epoch: 3, batch: 712, loss: 0.37438, accu: 0.93480, speed: 1.50 step/s
    eval dev loss: 0.42188, accu: 0.88491
    模型保存在 3100 步， 最佳eval准确度为0.88491252！
    global step 3110, epoch: 3, batch: 722, loss: 0.35337, accu: 0.93650, speed: 0.45 step/s
    global step 3120, epoch: 3, batch: 732, loss: 0.37482, accu: 0.93900, speed: 1.55 step/s
    global step 3130, epoch: 3, batch: 742, loss: 0.37821, accu: 0.93833, speed: 1.52 step/s
    global step 3140, epoch: 3, batch: 752, loss: 0.36280, accu: 0.93775, speed: 1.46 step/s
    global step 3150, epoch: 3, batch: 762, loss: 0.37027, accu: 0.93650, speed: 1.51 step/s
    global step 3160, epoch: 3, batch: 772, loss: 0.37834, accu: 0.93608, speed: 1.53 step/s
    global step 3170, epoch: 3, batch: 782, loss: 0.39605, accu: 0.93600, speed: 1.51 step/s
    global step 3180, epoch: 3, batch: 792, loss: 0.38949, accu: 0.93631, speed: 1.49 step/s
    global step 3190, epoch: 3, batch: 802, loss: 0.39597, accu: 0.93617, speed: 1.56 step/s
    global step 3200, epoch: 3, batch: 812, loss: 0.38174, accu: 0.93465, speed: 1.50 step/s
    eval dev loss: 0.42549, accu: 0.88207
    global step 3210, epoch: 3, batch: 822, loss: 0.38965, accu: 0.93650, speed: 0.59 step/s
    global step 3220, epoch: 3, batch: 832, loss: 0.38372, accu: 0.93450, speed: 1.34 step/s
    global step 3230, epoch: 3, batch: 842, loss: 0.39353, accu: 0.93400, speed: 1.47 step/s
    global step 3240, epoch: 3, batch: 852, loss: 0.36659, accu: 0.93463, speed: 1.53 step/s
    global step 3250, epoch: 3, batch: 862, loss: 0.38172, accu: 0.93420, speed: 1.53 step/s
    global step 3260, epoch: 3, batch: 872, loss: 0.36932, accu: 0.93492, speed: 1.53 step/s
    global step 3270, epoch: 3, batch: 882, loss: 0.35730, accu: 0.93586, speed: 1.43 step/s
    global step 3280, epoch: 3, batch: 892, loss: 0.38015, accu: 0.93488, speed: 1.47 step/s
    global step 3290, epoch: 3, batch: 902, loss: 0.39504, accu: 0.93344, speed: 1.50 step/s
    global step 3300, epoch: 3, batch: 912, loss: 0.38336, accu: 0.93355, speed: 1.51 step/s
    eval dev loss: 0.43423, accu: 0.87367
    global step 3310, epoch: 3, batch: 922, loss: 0.38159, accu: 0.92750, speed: 0.59 step/s
    global step 3320, epoch: 3, batch: 932, loss: 0.36495, accu: 0.92450, speed: 1.53 step/s
    global step 3330, epoch: 3, batch: 942, loss: 0.35236, accu: 0.92400, speed: 1.49 step/s
    global step 3340, epoch: 3, batch: 952, loss: 0.37281, accu: 0.92525, speed: 1.55 step/s
    global step 3350, epoch: 3, batch: 962, loss: 0.38050, accu: 0.92680, speed: 1.46 step/s
    global step 3360, epoch: 3, batch: 972, loss: 0.34803, accu: 0.92817, speed: 1.52 step/s
    global step 3370, epoch: 3, batch: 982, loss: 0.39216, accu: 0.92821, speed: 1.47 step/s
    global step 3380, epoch: 3, batch: 992, loss: 0.39303, accu: 0.92825, speed: 1.49 step/s
    global step 3390, epoch: 3, batch: 1002, loss: 0.37790, accu: 0.92883, speed: 1.46 step/s
    global step 3400, epoch: 3, batch: 1012, loss: 0.35329, accu: 0.92855, speed: 1.38 step/s
    eval dev loss: 0.43189, accu: 0.87423
    global step 3410, epoch: 3, batch: 1022, loss: 0.37775, accu: 0.93200, speed: 0.59 step/s
    global step 3420, epoch: 3, batch: 1032, loss: 0.38463, accu: 0.93350, speed: 1.45 step/s
    global step 3430, epoch: 3, batch: 1042, loss: 0.36326, accu: 0.93650, speed: 1.53 step/s
    global step 3440, epoch: 3, batch: 1052, loss: 0.37241, accu: 0.93537, speed: 1.58 step/s
    global step 3450, epoch: 3, batch: 1062, loss: 0.39410, accu: 0.93530, speed: 1.49 step/s
    global step 3460, epoch: 3, batch: 1072, loss: 0.41592, accu: 0.93233, speed: 1.50 step/s
    global step 3470, epoch: 3, batch: 1082, loss: 0.38917, accu: 0.93193, speed: 1.55 step/s
    global step 3480, epoch: 3, batch: 1092, loss: 0.38183, accu: 0.93100, speed: 1.46 step/s
    global step 3490, epoch: 3, batch: 1102, loss: 0.38094, accu: 0.93117, speed: 1.43 step/s
    global step 3500, epoch: 3, batch: 1112, loss: 0.39900, accu: 0.92990, speed: 1.45 step/s
    eval dev loss: 0.43208, accu: 0.87446
    global step 3510, epoch: 3, batch: 1122, loss: 0.38114, accu: 0.93050, speed: 0.59 step/s
    global step 3520, epoch: 3, batch: 1132, loss: 0.38534, accu: 0.93475, speed: 1.51 step/s
    global step 3530, epoch: 3, batch: 1142, loss: 0.38292, accu: 0.93600, speed: 1.46 step/s
    global step 3540, epoch: 3, batch: 1152, loss: 0.37553, accu: 0.93675, speed: 1.55 step/s
    global step 3550, epoch: 3, batch: 1162, loss: 0.37499, accu: 0.93410, speed: 1.55 step/s
    global step 3560, epoch: 3, batch: 1172, loss: 0.40679, accu: 0.93408, speed: 1.57 step/s
    global step 3570, epoch: 3, batch: 1182, loss: 0.36577, accu: 0.93300, speed: 1.50 step/s
    global step 3580, epoch: 3, batch: 1192, loss: 0.39443, accu: 0.93375, speed: 1.58 step/s
    global step 3590, epoch: 4, batch: 8, loss: 0.39499, accu: 0.93415, speed: 1.60 step/s
    global step 3600, epoch: 4, batch: 18, loss: 0.37901, accu: 0.93384, speed: 1.52 step/s
    eval dev loss: 0.42901, accu: 0.87832
    global step 3610, epoch: 4, batch: 28, loss: 0.35560, accu: 0.93400, speed: 0.60 step/s
    global step 3620, epoch: 4, batch: 38, loss: 0.37279, accu: 0.94025, speed: 1.50 step/s
    global step 3630, epoch: 4, batch: 48, loss: 0.36635, accu: 0.94033, speed: 1.52 step/s
    global step 3640, epoch: 4, batch: 58, loss: 0.39046, accu: 0.93800, speed: 1.57 step/s
    global step 3650, epoch: 4, batch: 68, loss: 0.35310, accu: 0.93820, speed: 1.48 step/s
    global step 3660, epoch: 4, batch: 78, loss: 0.37144, accu: 0.93892, speed: 1.51 step/s
    global step 3670, epoch: 4, batch: 88, loss: 0.35536, accu: 0.94007, speed: 1.54 step/s
    global step 3680, epoch: 4, batch: 98, loss: 0.36349, accu: 0.94106, speed: 1.48 step/s
    global step 3690, epoch: 4, batch: 108, loss: 0.37111, accu: 0.94211, speed: 1.53 step/s
    global step 3700, epoch: 4, batch: 118, loss: 0.36880, accu: 0.94240, speed: 1.49 step/s
    eval dev loss: 0.43105, accu: 0.87651
    global step 3710, epoch: 4, batch: 128, loss: 0.36443, accu: 0.93200, speed: 0.58 step/s
    global step 3720, epoch: 4, batch: 138, loss: 0.36232, accu: 0.93700, speed: 1.44 step/s
    global step 3730, epoch: 4, batch: 148, loss: 0.37408, accu: 0.94000, speed: 1.52 step/s
    global step 3740, epoch: 4, batch: 158, loss: 0.37071, accu: 0.94063, speed: 1.49 step/s
    global step 3750, epoch: 4, batch: 168, loss: 0.36296, accu: 0.94230, speed: 1.47 step/s
    global step 3760, epoch: 4, batch: 178, loss: 0.38324, accu: 0.94125, speed: 1.50 step/s
    global step 3770, epoch: 4, batch: 188, loss: 0.40992, accu: 0.94114, speed: 1.52 step/s
    global step 3780, epoch: 4, batch: 198, loss: 0.38006, accu: 0.94075, speed: 1.52 step/s
    global step 3790, epoch: 4, batch: 208, loss: 0.38927, accu: 0.94061, speed: 1.48 step/s
    global step 3800, epoch: 4, batch: 218, loss: 0.33963, accu: 0.94135, speed: 1.50 step/s
    eval dev loss: 0.42401, accu: 0.88412
    global step 3810, epoch: 4, batch: 228, loss: 0.36093, accu: 0.93450, speed: 0.59 step/s
    global step 3820, epoch: 4, batch: 238, loss: 0.36947, accu: 0.93950, speed: 1.54 step/s
    global step 3830, epoch: 4, batch: 248, loss: 0.37647, accu: 0.93717, speed: 1.51 step/s
    global step 3840, epoch: 4, batch: 258, loss: 0.34997, accu: 0.94025, speed: 1.49 step/s
    global step 3850, epoch: 4, batch: 268, loss: 0.37466, accu: 0.94150, speed: 1.52 step/s
    global step 3860, epoch: 4, batch: 278, loss: 0.39072, accu: 0.94233, speed: 1.46 step/s
    global step 3870, epoch: 4, batch: 288, loss: 0.38144, accu: 0.94343, speed: 1.47 step/s
    global step 3880, epoch: 4, batch: 298, loss: 0.35403, accu: 0.94419, speed: 1.49 step/s
    global step 3890, epoch: 4, batch: 308, loss: 0.34428, accu: 0.94328, speed: 1.51 step/s
    global step 3900, epoch: 4, batch: 318, loss: 0.37691, accu: 0.94295, speed: 1.48 step/s
    eval dev loss: 0.41926, accu: 0.88912
    模型保存在 3900 步， 最佳eval准确度为0.88911611！
    global step 3910, epoch: 4, batch: 328, loss: 0.36345, accu: 0.93500, speed: 0.46 step/s
    global step 3920, epoch: 4, batch: 338, loss: 0.35641, accu: 0.93800, speed: 1.50 step/s
    global step 3930, epoch: 4, batch: 348, loss: 0.37291, accu: 0.93683, speed: 1.58 step/s
    global step 3940, epoch: 4, batch: 358, loss: 0.37807, accu: 0.93725, speed: 1.59 step/s
    global step 3950, epoch: 4, batch: 368, loss: 0.37960, accu: 0.94010, speed: 1.45 step/s
    global step 3960, epoch: 4, batch: 378, loss: 0.38077, accu: 0.94000, speed: 1.52 step/s
    global step 3970, epoch: 4, batch: 388, loss: 0.36645, accu: 0.94186, speed: 1.53 step/s
    global step 3980, epoch: 4, batch: 398, loss: 0.36161, accu: 0.94150, speed: 1.46 step/s
    global step 3990, epoch: 4, batch: 408, loss: 0.37637, accu: 0.94156, speed: 1.52 step/s
    global step 4000, epoch: 4, batch: 418, loss: 0.35858, accu: 0.94155, speed: 1.44 step/s
    eval dev loss: 0.43469, accu: 0.87242
    global step 4010, epoch: 4, batch: 428, loss: 0.34503, accu: 0.94150, speed: 0.59 step/s
    global step 4020, epoch: 4, batch: 438, loss: 0.39073, accu: 0.93475, speed: 1.56 step/s
    global step 4030, epoch: 4, batch: 448, loss: 0.37228, accu: 0.93683, speed: 1.55 step/s
    global step 4040, epoch: 4, batch: 458, loss: 0.37215, accu: 0.93800, speed: 1.59 step/s
    global step 4050, epoch: 4, batch: 468, loss: 0.36453, accu: 0.93850, speed: 1.53 step/s
    global step 4060, epoch: 4, batch: 478, loss: 0.36707, accu: 0.93900, speed: 1.49 step/s
    global step 4070, epoch: 4, batch: 488, loss: 0.35858, accu: 0.93771, speed: 1.45 step/s
    global step 4080, epoch: 4, batch: 498, loss: 0.40287, accu: 0.93812, speed: 1.59 step/s
    global step 4090, epoch: 4, batch: 508, loss: 0.36708, accu: 0.93883, speed: 1.51 step/s
    global step 4100, epoch: 4, batch: 518, loss: 0.39264, accu: 0.93850, speed: 1.49 step/s
    eval dev loss: 0.42774, accu: 0.87923
    global step 4110, epoch: 4, batch: 528, loss: 0.35168, accu: 0.93850, speed: 0.59 step/s
    global step 4120, epoch: 4, batch: 538, loss: 0.37005, accu: 0.93725, speed: 1.45 step/s
    global step 4130, epoch: 4, batch: 548, loss: 0.38220, accu: 0.93983, speed: 1.48 step/s
    global step 4140, epoch: 4, batch: 558, loss: 0.36143, accu: 0.94212, speed: 1.50 step/s
    global step 4150, epoch: 4, batch: 568, loss: 0.37314, accu: 0.94230, speed: 1.55 step/s
    global step 4160, epoch: 4, batch: 578, loss: 0.37550, accu: 0.94283, speed: 1.54 step/s
    global step 4170, epoch: 4, batch: 588, loss: 0.34542, accu: 0.94186, speed: 1.42 step/s
    global step 4180, epoch: 4, batch: 598, loss: 0.37046, accu: 0.94144, speed: 1.55 step/s
    global step 4190, epoch: 4, batch: 608, loss: 0.41137, accu: 0.94033, speed: 1.50 step/s
    global step 4200, epoch: 4, batch: 618, loss: 0.35828, accu: 0.94125, speed: 1.58 step/s
    eval dev loss: 0.42616, accu: 0.88048
    global step 4210, epoch: 4, batch: 628, loss: 0.37383, accu: 0.93450, speed: 0.59 step/s
    global step 4220, epoch: 4, batch: 638, loss: 0.39054, accu: 0.93900, speed: 1.47 step/s
    global step 4230, epoch: 4, batch: 648, loss: 0.36759, accu: 0.94350, speed: 1.59 step/s
    global step 4240, epoch: 4, batch: 658, loss: 0.38310, accu: 0.94512, speed: 1.47 step/s
    global step 4250, epoch: 4, batch: 668, loss: 0.39443, accu: 0.94350, speed: 1.51 step/s
    global step 4260, epoch: 4, batch: 678, loss: 0.37135, accu: 0.94258, speed: 1.52 step/s
    global step 4270, epoch: 4, batch: 688, loss: 0.36455, accu: 0.94314, speed: 1.53 step/s
    global step 4280, epoch: 4, batch: 698, loss: 0.40895, accu: 0.94281, speed: 1.53 step/s
    global step 4290, epoch: 4, batch: 708, loss: 0.36187, accu: 0.94244, speed: 1.56 step/s
    global step 4300, epoch: 4, batch: 718, loss: 0.39506, accu: 0.94220, speed: 1.52 step/s
    eval dev loss: 0.42217, accu: 0.88594
    global step 4310, epoch: 4, batch: 728, loss: 0.34999, accu: 0.94400, speed: 0.59 step/s
    global step 4320, epoch: 4, batch: 738, loss: 0.39262, accu: 0.94275, speed: 1.48 step/s
    global step 4330, epoch: 4, batch: 748, loss: 0.37606, accu: 0.94333, speed: 1.51 step/s
    global step 4340, epoch: 4, batch: 758, loss: 0.38096, accu: 0.94275, speed: 1.52 step/s
    global step 4350, epoch: 4, batch: 768, loss: 0.36903, accu: 0.94110, speed: 1.53 step/s
    global step 4360, epoch: 4, batch: 778, loss: 0.38158, accu: 0.94050, speed: 1.51 step/s
    global step 4370, epoch: 4, batch: 788, loss: 0.37696, accu: 0.93986, speed: 1.47 step/s
    global step 4380, epoch: 4, batch: 798, loss: 0.39192, accu: 0.94056, speed: 1.59 step/s
    global step 4390, epoch: 4, batch: 808, loss: 0.37772, accu: 0.94150, speed: 1.53 step/s
    global step 4400, epoch: 4, batch: 818, loss: 0.37008, accu: 0.94205, speed: 1.52 step/s
    eval dev loss: 0.42124, accu: 0.88628
    global step 4410, epoch: 4, batch: 828, loss: 0.37302, accu: 0.93300, speed: 0.59 step/s
    global step 4420, epoch: 4, batch: 838, loss: 0.35399, accu: 0.93850, speed: 1.48 step/s
    global step 4430, epoch: 4, batch: 848, loss: 0.37052, accu: 0.93650, speed: 1.53 step/s
    global step 4440, epoch: 4, batch: 858, loss: 0.36451, accu: 0.93912, speed: 1.59 step/s
    global step 4450, epoch: 4, batch: 868, loss: 0.36563, accu: 0.94000, speed: 1.50 step/s
    global step 4460, epoch: 4, batch: 878, loss: 0.36557, accu: 0.94042, speed: 1.43 step/s
    global step 4470, epoch: 4, batch: 888, loss: 0.35084, accu: 0.93979, speed: 1.59 step/s
    global step 4480, epoch: 4, batch: 898, loss: 0.36889, accu: 0.93944, speed: 1.54 step/s
    global step 4490, epoch: 4, batch: 908, loss: 0.38990, accu: 0.93961, speed: 1.53 step/s
    global step 4500, epoch: 4, batch: 918, loss: 0.38442, accu: 0.93965, speed: 1.56 step/s
    eval dev loss: 0.42184, accu: 0.8865
    global step 4510, epoch: 4, batch: 928, loss: 0.34947, accu: 0.93500, speed: 0.56 step/s
    global step 4520, epoch: 4, batch: 938, loss: 0.41597, accu: 0.93625, speed: 1.50 step/s
    global step 4530, epoch: 4, batch: 948, loss: 0.36256, accu: 0.94067, speed: 1.57 step/s
    global step 4540, epoch: 4, batch: 958, loss: 0.35667, accu: 0.94050, speed: 1.49 step/s
    global step 4550, epoch: 4, batch: 968, loss: 0.40563, accu: 0.93990, speed: 1.61 step/s
    global step 4560, epoch: 4, batch: 978, loss: 0.37892, accu: 0.93908, speed: 1.50 step/s
    global step 4570, epoch: 4, batch: 988, loss: 0.37644, accu: 0.93814, speed: 1.44 step/s
    global step 4580, epoch: 4, batch: 998, loss: 0.38686, accu: 0.93812, speed: 1.53 step/s
    global step 4590, epoch: 4, batch: 1008, loss: 0.41330, accu: 0.93800, speed: 1.51 step/s
    global step 4600, epoch: 4, batch: 1018, loss: 0.38275, accu: 0.93810, speed: 1.56 step/s
    eval dev loss: 0.42597, accu: 0.8815
    global step 4610, epoch: 4, batch: 1028, loss: 0.39425, accu: 0.94450, speed: 0.59 step/s
    global step 4620, epoch: 4, batch: 1038, loss: 0.39118, accu: 0.93975, speed: 1.52 step/s
    global step 4630, epoch: 4, batch: 1048, loss: 0.37131, accu: 0.93833, speed: 1.52 step/s
    global step 4640, epoch: 4, batch: 1058, loss: 0.37409, accu: 0.93875, speed: 1.57 step/s
    global step 4650, epoch: 4, batch: 1068, loss: 0.37925, accu: 0.93850, speed: 1.49 step/s
    global step 4660, epoch: 4, batch: 1078, loss: 0.39414, accu: 0.93650, speed: 1.59 step/s
    global step 4670, epoch: 4, batch: 1088, loss: 0.36100, accu: 0.93879, speed: 1.51 step/s
    global step 4680, epoch: 4, batch: 1098, loss: 0.39191, accu: 0.93875, speed: 1.46 step/s
    global step 4690, epoch: 4, batch: 1108, loss: 0.37352, accu: 0.93800, speed: 1.54 step/s
    global step 4700, epoch: 4, batch: 1118, loss: 0.37136, accu: 0.93840, speed: 1.33 step/s
    eval dev loss: 0.42098, accu: 0.88707
    global step 4710, epoch: 4, batch: 1128, loss: 0.35813, accu: 0.94400, speed: 0.58 step/s
    global step 4720, epoch: 4, batch: 1138, loss: 0.37324, accu: 0.94475, speed: 1.52 step/s
    global step 4730, epoch: 4, batch: 1148, loss: 0.38025, accu: 0.94267, speed: 1.45 step/s
    global step 4740, epoch: 4, batch: 1158, loss: 0.37031, accu: 0.94012, speed: 1.48 step/s
    global step 4750, epoch: 4, batch: 1168, loss: 0.34159, accu: 0.94160, speed: 1.45 step/s
    global step 4760, epoch: 4, batch: 1178, loss: 0.38796, accu: 0.94083, speed: 1.35 step/s



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-17-f1765037eb45> in <module>
         26         optimizer.step()
         27         lr_scheduler.step()
    ---> 28         optimizer.clear_grad()
         29 
         30 


    <decorator-gen-314> in clear_grad(self)


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/wrapped_decorator.py in __impl__(func, *args, **kwargs)
         23     def __impl__(func, *args, **kwargs):
         24         wrapped_func = decorator_func(func)
    ---> 25         return wrapped_func(*args, **kwargs)
         26 
         27     return __impl__


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py in __impl__(*args, **kwargs)
        223         assert in_dygraph_mode(
        224         ), "We only support '%s()' in dynamic graph mode, please call 'paddle.disable_static()' to enter dynamic graph mode." % func.__name__
    --> 225         return func(*args, **kwargs)
        226 
        227     return __impl__


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/optimizer/optimizer.py in clear_grad(self)
        843         for p in self._parameter_list:
        844             if not p.stop_gradient:
    --> 845                 p.clear_gradient()
        846 
        847     @imperative_base.no_grad


    KeyboardInterrupt: 



```python
# 训练结束后，存储模型参数
# save_dir = os.path.join("checkpoint", "model_%d" % global_step)
# os.makedirs(save_dir)
tokenizer.save_pretrained(save_dir)
# save_param_path = os.path.join(save_dir, 'model_state.pdparams')
# paddle.save(model.state_dict(), save_param_path)
# tokenizer.save_pretrained(save_dir)
break!!!
```


      File "<ipython-input-18-fb6bd3295eff>", line 8
        break!!!
             ^
    SyntaxError: invalid syntax



# 五、模型预测
接下来我们使用已经训练好的语义匹配模型对一些预测数据进行预测。待预测数据为每行都是文本对的 tsv 文件，我们使用 Lcqmc 数据集的测试集作为我们的预测数据，进行预测并提交预测结果到 千言文本相似度竞赛

下载我们已经训练好的语义匹配模型, 并解压


```python
break
```


```python
# 下载我们基于 Lcqmc 事先训练好的语义匹配模型并解压
! wget https://paddlenlp.bj.bcebos.com/models/text_matching/ernie_gram_zh_pointwise_matching_model.tar
! tar -xvf ernie_gram_zh_pointwise_matching_model.tar
```

    --2021-06-15 14:41:34--  https://paddlenlp.bj.bcebos.com/models/text_matching/ernie_gram_zh_pointwise_matching_model.tar
    Resolving paddlenlp.bj.bcebos.com (paddlenlp.bj.bcebos.com)... 182.61.200.195, 182.61.200.229, 2409:8c00:6c21:10ad:0:ff:b00e:67d
    Connecting to paddlenlp.bj.bcebos.com (paddlenlp.bj.bcebos.com)|182.61.200.195|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 597667840 (570M) [application/x-tar]
    Saving to: ‘ernie_gram_zh_pointwise_matching_model.tar.1’
    
    ernie_gram_zh_point 100%[===================>] 569.98M  54.6MB/s    in 12s     
    
    2021-06-15 14:41:46 (45.7 MB/s) - ‘ernie_gram_zh_pointwise_matching_model.tar.1’ saved [597667840/597667840]
    
    ernie_gram_zh_pointwise_matching_model/
    ernie_gram_zh_pointwise_matching_model/model_state.pdparams
    ernie_gram_zh_pointwise_matching_model/vocab.txt
    ernie_gram_zh_pointwise_matching_model/tokenizer_config.json



```python
# 测试数据由 2 列文本构成 tab 分隔
# Lcqmc 默认下载到如下路径
! head -n10 "${HOME}/.paddlenlp/datasets/LCQMC/lcqmc/lcqmc/test.tsv"
```

## 1.定义预测函数


```python
def predict(model, data_loader):
    
    batch_probs = []

    # 预测阶段打开 eval 模式，模型中的 dropout 等操作会关掉
    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data
            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)
            
            # 获取每个样本的预测概率: [batch_size, 2] 的矩阵
            batch_prob = model(
                input_ids=input_ids, token_type_ids=token_type_ids).numpy()

            batch_probs.append(batch_prob)
        batch_probs = np.concatenate(batch_probs, axis=0)

        return batch_probs
```

## 2.定义预测数据的 data_loader


```python
!head paws-x-zh/test.tsv
!head paws-x-zh/train.tsv
```


```python
!unzip data/data52714/lcqmc.zip -d lcqmc
```


```python
# 预测数据的转换函数
# predict 数据没有 label, 因此 convert_exmaple 的 is_test 参数设为 True
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=512,
    is_test=True)

# 预测数据的组 batch 操作
# predict 数据只返回 input_ids 和 token_type_ids，因此只需要 2 个 Pad 对象作为 batchify_fn
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
): [data for data in fn(samples)]

# 加载预测数据
test_ds = load_dataset("lcqmc", splits='test')
# test_ds = load_dataset("lcqmc", data_files='paws-x-zh/test.tsv')
# test_ds = load_dataset("lcqmc", data_files='bq_corpus/test.tsv')
```


```python
batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=32, shuffle=False)

# 生成预测数据 data_loader
predict_data_loader =paddle.io.DataLoader(
        dataset=test_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
```

## 3. 定义预测模型


```python
# pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')
pretrained_model = paddlenlp.transformers.ErnieModel.from_pretrained('ernie-1.0')
model = PointwiseMatching(pretrained_model)
```

    [2021-06-15 14:42:12,101] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-1.0/ernie_v1_chn_base.pdparams
    [2021-06-15 14:42:13,309] [    INFO] - Weights from pretrained model not used in ErnieModel: ['cls.predictions.layer_norm.weight', 'cls.predictions.decoder_bias', 'cls.predictions.transform.bias', 'cls.predictions.transform.weight', 'cls.predictions.layer_norm.bias']


## 4.加载已训练好的模型参数（此处也可以直接加载预训练模型进行预测，也可以加载最佳训练的checkpoint进行训练）



```python
# 刚才下载的模型解压之后存储路径为 ./ernie_gram_zh_pointwise_matching_model/model_state.pdparams
# state_dict = paddle.load("./ernie_gram_zh_pointwise_matching_model/model_state.pdparams")

state_dict = paddle.load("checkpoint_lcqmc/best_model.pdparams")
# 刚才下载的模型解压之后存储路径为 ./pointwise_matching_model/ernie1.0_base_pointwise_matching.pdparams
# state_dict = paddle.load("pointwise_matching_model/ernie1.0_base_pointwise_matching.pdparams")
model.set_dict(state_dict)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for ptm.embeddings.word_embeddings.weight. ptm.embeddings.word_embeddings.weight receives a shape [18018, 768], but the expected shape is [18000, 768].
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for ptm.embeddings.position_embeddings.weight. ptm.embeddings.position_embeddings.weight receives a shape [512, 768], but the expected shape is [513, 768].
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))


## 5.开始预测


```python
for idx, batch in enumerate(predict_data_loader):
    if idx < 1:
        print(batch)
```

    [Tensor(shape=[32, 38], dtype=int64, place=CUDAPinnedPlace, stop_gradient=True,
           [[1   , 1022, 9   , ..., 0   , 0   , 0   ],
            [1   , 514 , 904 , ..., 0   , 0   , 0   ],
            [1   , 47  , 10  , ..., 0   , 0   , 0   ],
            ...,
            [1   , 733 , 404 , ..., 0   , 0   , 0   ],
            [1   , 134 , 170 , ..., 0   , 0   , 0   ],
            [1   , 379 , 3122, ..., 0   , 0   , 0   ]]), Tensor(shape=[32, 38], dtype=int64, place=CUDAPinnedPlace, stop_gradient=True,
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]])]



```python
# 执行预测函数
y_probs = predict(model, predict_data_loader)

# 根据预测概率获取预测 label
y_preds = np.argmax(y_probs, axis=1)
```

##  6.输出预测结果


```python
# 我们按照千言文本相似度竞赛的提交格式将预测结果存储在 lcqmc.tsv 中，用来后续提交
# 同时将预测结果输出到终端，便于大家直观感受模型预测效果

# test_ds = load_dataset("lcqmc", splits=["test"])

# with open("lcqmc.tsv", 'w', encoding="utf-8") as f:
# with open("paws-x.tsv", 'w', encoding="utf-8") as f:
with open("lcqmc.tsv", 'w', encoding="utf-8") as f:
    f.write("index\tprediction\n")    
    for idx, y_pred in enumerate(y_preds):
        f.write("{}\t{}\n".format(idx, y_pred))
        print("{}\t{}\n".format(idx, y_pred))
        # text_pair = test_ds[idx]
        # text_pair["id"] = test_ds[idx]
        # text_pair["label"] = y_pred
        # print(text_pair)
```

    

    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    1809	1
    
    1810	1
    
    1811	1
    
    1812	1
    
    1813	1
    
    1814	0
    
    1815	1
    
    1816	1
    
    1817	1
    
    1818	1
    
    1819	1
    
    1820	0
    
    1821	0
    
    1822	1
    
    1823	0
    
    1824	1
    
    1825	0
    
    1826	1
    
    1827	0
    
    1828	0
    
    1829	0
    
    1830	1
    
    1831	1
    
    1832	0
    
    1833	0
    
    1834	0
    
    1835	0
    
    1836	1
    
    1837	0
    
    1838	1
    
    1839	1
    
    1840	0
    
    1841	0
    
    1842	1
    
    1843	1
    
    1844	0
    
    1845	0
    
    1846	0
    
    1847	1
    
    1848	0
    
    1849	0
    
    1850	1
    
    1851	1
    
    1852	0
    
    1853	0
    
    1854	1
    
    1855	1
    
    1856	0
    
    1857	0
    
    1858	1
    
    1859	1
    
    1860	0
    
    1861	0
    
    1862	1
    
    1863	1
    
    1864	1
    
    1865	0
    
    1866	0
    
    1867	1
    
    1868	1
    
    1869	1
    
    1870	0
    
    1871	1
    
    1872	1
    
    1873	1
    
    1874	0
    
    1875	0
    
    1876	0
    
    1877	1
    
    1878	0
    
    1879	1
    
    1880	1
    
    1881	1
    
    1882	1
    
    1883	1
    
    1884	1
    
    1885	1
    
    1886	1
    
    1887	1
    
    1888	1
    
    1889	1
    
    1890	1
    
    1891	1
    
    1892	0
    
    1893	1
    
    1894	0
    
    1895	0
    
    1896	1
    
    1897	1
    
    1898	1
    
    1899	1
    
    1900	1
    
    1901	1
    
    1902	0
    
    1903	0
    4378	1

    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


    
    4900	1
    
    4901	1
    
    4902	1
    
    4903	1
    
    4904	1
    
    4905	0
    
    4906	1
    
    4907	1
    
    4908	1
    
    4909	1
    
    4910	1
    
    4911	0
    
    4912	0
    
    4913	1
    
    4914	0
    
    4915	1
    
    4916	0
    
    4917	1
    
    4918	0
    
    4919	0
    
    4920	1
    
    4921	1
    
    4922	0
    
    4923	0
    
    4924	0
    
    4925	1
    
    4926	0
    
    4927	1
    
    4928	0
    
    4929	1
    
    4930	0
    
    4931	0
    
    4932	0
    
    4933	1
    
    4934	1
    
    4935	0
    
    4936	1
    
    4937	0
    
    4938	1
    
    4939	1
    
    4940	0
    
    4941	1
    
    4942	1
    
    4943	1
    
    4944	1
    
    4945	1
    
    4946	1
    
    4947	1
    
    4948	0
    
    4949	1
    
    4950	0
    
    4951	1
    
    4952	1
    
    4953	1
    
    4954	0
    
    4955	0
    
    4956	1
    
    4957	0
    
    4958	1
    
    4959	0
    
    4960	1
    
    4961	0
    
    4962	0
    
    4963	0
    
    4964	0
    
    4965	0
    
    4966	1
    
    4967	0
    
    4968	1
    
    4969	0
    
    4970	1
    
    4971	1
    
    4972	1
    
    4973	1
    
    4974	1
    
    4975	1
    
    4976	0
    
    4977	1
    
    4978	0
    
    4979	1
    
    4980	1
    
    4981	0
    
    4982	1
    
    4983	0
    
    4984	1
    
    4985	0
    
    4986	1
    
    4987	1
    
    4988	1
    
    4989	0
    
    4990	0
    
    4991	1
    
    4992	0
    
    4993	0
    
    4994	0
    
    4995	1
    
    4996	0
    
    4997	1
    
    4998	0
    
    4999	1
    
    5000	1
    
    5001	0
    
    5002	1
    
    5003	1
    
    5004	1
    
    5005	1
    
    5006	1
    
    5007	1
    
    5008	1
    
    5009	1
    
    5010	1
    
    5011	0
    
    5012	1
    
    5013	1
    
    5014	0
    
    5015	1
    
    5016	1
    
    5017	1
    
    5018	1
    
    5019	0
    
    5020	0
    
    5021	1
    
    5022	0
    
    5023	1
    
    5024	1
    
    

# 六、BQ Corpus、PAWS-X (中文) 相似度预测流程

## 1.解压缩bq_corpus.zip、paws-x-zh.zip数据集
## 2.自定义数据集训练其他两类模型

代码如下自定义数据集，注意一个是数据集类型 **lcqmc**， 一个是文件夹所在位置，**splits** 提示所需返回的dataset
```
from paddlenlp.datasets import load_dataset
train_ds, dev_ds = load_dataset("lcqmc", data_files='bq_corpus/', splits=("train", "dev"))
```
## 3.替换预测数据集开始预测
指明数据格式和文件名
`test_ds = load_dataset("lcqmc", data_files='bq_corpus/test.tsv')`


```python
!unzip -qa data/data52714/bq_corpus.zip
!unzip -qa data/data52714/paws-x-zh.zip
```

## 4.提交 lcqmc 预测结果千言文本相似度竞赛
千言文本相似度竞赛一共有 3 个数据集: lcqmc、bq_corpus、paws-x, 我们刚才生成了 lcqmc 的预测结果 lcqmc.tsv, 同时我们在项目内提供了 bq_corpus、paw-x 数据集的空预测结果，我们将这 3 个文件打包提交到千言文本相似度竞赛，即可看到自己的模型在 Lcqmc 数据集上的竞赛成绩。


```python
# 打包预测结果
!zip submit.zip lcqmc.tsv paws-x.tsv bq_corpus.tsv
```

## 5.跑个demo看看
有时间思考思考跑起来啊啊啊

![](https://ai-studio-static-online.cdn.bcebos.com/f262f80927c14919b2ed2017d7c420d1e323293bd3ed4426ae3f688fc130ace5)


## 6.加入visual dl 观察训练
将train、val情况写入日志，并用visual dl查看训练进展

## 7.保存仍有问题不能够对比保存最佳结果
