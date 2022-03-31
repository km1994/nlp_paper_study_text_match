# 【关于 NLP】 那些你不知道的事 —— 文本匹配任务篇

> 作者：杨夕
> 
> 介绍：研读顶会论文，复现论文相关代码
> 
> NLP 百面百搭 地址：https://github.com/km1994/NLP-Interview-Notes
> 
> **[手机版NLP百面百搭](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100005719&idx=3&sn=5d8e62993e5ecd4582703684c0d12e44&chksm=1bbff26d2cc87b7bf2504a8a4cafc60919d722b6e9acbcee81a626924d80f53a49301df9bd97&scene=18#wechat_redirect)**
> 
> 推荐系统 百面百搭 地址：https://github.com/km1994/RES-Interview-Notes
> 
> **[手机版推荐系统百面百搭](https://mp.weixin.qq.com/s/b_KBT6rUw09cLGRHV_EUtw)**
> 
> 搜索引擎 百面百搭 地址：https://github.com/km1994/search-engine-Interview-Notes 【编写ing】
> 
> NLP论文学习笔记：https://github.com/km1994/nlp_paper_study
> 
> 推荐系统论文学习笔记：https://github.com/km1994/RS_paper_study
> 
> GCN 论文学习笔记：https://github.com/km1994/GCN_study
> 
> **推广搜 军火库**：https://github.com/km1994/recommendation_advertisement_search
![](other_study/resource/pic/微信截图_20210301212242.png)

> 手机版笔记，可以关注公众号 **【关于NLP那些你不知道的事】** 获取，并加入 【NLP && 推荐学习群】一起学习！！！

> 注：github 网页版 看起来不舒服，可以看 **[手机版NLP论文学习笔记](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100005719&idx=1&sn=14d34d70a7e7cbf9700f804cca5be2d0&chksm=1bbff26d2cc87b7b9d2ed12c8d280cd737e270cd82c8850f7ca2ee44ec8883873ff5e9904e7e&scene=18#wechat_redirect)**

- [【关于 NLP】 那些你不知道的事 —— 文本匹配任务篇](#关于-nlp-那些你不知道的事--文本匹配任务篇)
  - [介绍](#介绍)
    - [NLP 学习篇](#nlp-学习篇)
      - [经典会议论文研读篇](#经典会议论文研读篇)
      - [理论学习篇](#理论学习篇)
        - [经典论文研读篇](#经典论文研读篇)
        - [【关于 文本匹配】 那些的你不知道的事](#关于-文本匹配-那些的你不知道的事)
  - [参考资料](#参考资料)

## 介绍

### NLP 学习篇

#### 经典会议论文研读篇

- [ACL2020](ACL/ACL2020.md)
  - [【关于 CHECKLIST】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/other_study/meeting/ACL_study/ACL2020_bertpaper_CHECKLIST/)
    - 阅读理由：ACL2020 best paper ，利用 软件工程 的 思想 思考 深度学习
    - 动机：针对 train-val-test 分割方法 评估 模型性能容易出现 不全面、偏向性、可解性差问题；
    - 方法：提出了一种模型无关和任务无关的测试方法checklist，它使用三种不同的测试类型来测试模型的独立性。
    - 效果：checklist揭示了大型软件公司开发的商业系统中的关键缺陷，表明它是对当前实践的补充好吧。测试使用 checklist 创建的模型可以应用于任何模型，这样就可以很容易地将其纳入当前的基准测试或评估中管道。

#### 理论学习篇

##### 经典论文研读篇

- 那些你所不知道的事
  - [【关于Transformer】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/transformer_study/Transformer/)
  - [【关于Bert】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/bert_study/T1_bert/)


##### [【关于 文本匹配】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/text_match_study/) 

- [【关于 SimCSE】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/text_match_study/SimCSE/) **【推荐阅读】**
  - 论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings
  - 会议：
  - 论文地址：https://arxiv.org/abs/2104.08821
  - 论文代码：https://github.com/princeton-nlp/SimCSE
  - 思路：
    - 首先描述了一种无监督方法，它采用输入句子并在对比目标中预测自己，仅将标准 dropout 用作噪声。这种简单的方法效果出奇地好，与以前的受监督计数器部件相当。我们假设 dropout 充当最小数据增强的大小，删除它会导致表示崩溃。
    - 然后，我们从最近从自然语言推理 (NLI) 数据集中学习句子嵌入的成功中汲取灵感，并将 NLI 数据集中的注释对合并到对比学习中，方法是使用“蕴含”对作为正例，将“矛盾”对作为硬负例。
  - 实验结果：
    - 作者评估了标准语义文本相似性（STS）任务上的 SimCSE，使用 BERT-base 的无监督和监督模型分别平均实现了 74.5％ 和 81.6％ 的 Spearman 相关性，与之前的最佳结果相比，分别提高了 7.9 和 4.6点。
    - 作者还表明，对比学习理论上将嵌入分布得更均匀，并且在有监督信号可用时，它可以更好地对齐正样本。
- [【关于 BERT-flow 】那些你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/text_match_study/BERTFlow/)
  - 论文：On the Sentence Embeddings from Pre-trained Language Models
  - 会议：EMNLP2020
  - 论文地址：https://arxiv.org/pdf/2011.05864.pdf
  - 论文代码：https://github.com/bohanli/BERT-flow
  - 前沿：像BERT这样的经过预训练的上下文表示在自然语言处理中取得了巨大的成功；
  - 动机：已经发现，未经微调的来自预训练语言模型的句子嵌入很难捕获句子的语义；
  - 论文方法：在本文中，我们认为BERT嵌入中的语义信息没有得到充分利用。我们首先从理论上揭示了掩盖的语言模型预训练目标与语义相似性任务之间的理论联系，然后从经验上分析了BERT句子的嵌入。
  - 实验结果：我们发现BERT总是诱发非光滑的各向异性语义空间，这会损害其语义相似性的表现。为解决此问题，我们建议通过将非正则化的流量标准化来将各向异性的语义嵌入分布转换为平滑的各向异性高斯分布。实验结果表明，我们提出的BERT流方法在各种语义文本相似性任务上比最先进的句子嵌入方法具有明显的性能提升。
- [【关于 Sentence-BERT】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/text_match_study/sentence_bert/)
  - 项目地址：https://github.com/km1994/nlp_paper_study
  - 论文：Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
  - github:https://github.com/UKPLab/sentence-transformers
  - 动机：
    - 方法一：BERT使用交叉编码器：将两个句子传递到变压器网络，并预测目标值；
      - 问题： 由于太多可能的组合，此设置不适用于各种对回归任务。 在n = 10000个句子的集合中找到相似度最高的对需要BERT n·（n-1）/ 2 = 49 995 000推理计算。 在现代V100 GPU上，这大约需要65个小时。 类似地，对于一个新问题，找到Quora的超过4,000万个现有问题中最相似的一个可以建模为与BERT的成对比较，但是，回答单个查询将需要50多个小时。
    - 方法二：解决聚类和语义搜索的常用方法是将每个句子映射到向量空间，以使语义相似的句子接近。 研究人员已开始将单个句子输入BERT，并得出固定大小的句子嵌入。 最常用的方法是平均BERT输出层（称为BERT嵌入）或通过使用第一个令牌的输出（[CLS]令牌）；
      - 问题：就像我们将要展示的那样，这种常规做法产生的句子嵌入效果很差，通常比平均GloVe嵌入效果更差。
  - 论文方法：
    - 我们开发了SBERT。 siamese network 体系结构使得可以导出输入句子的固定大小矢量。 使用余弦相似度或Manhatten / Euclidean距离之类的相似度度量，可以找到语义上相似的句子。 
  - 存在问题解答：
    - 小问题：[在语义相似度任务中，SBERT的计算速度为什么比纯bert进行句子编码要快？](https://github.com/km1994/nlp_paper_study_text_match/tree/master/text_match_study/sentence_bert/)
- [【关于 语义相似度匹配任务中的 BERT】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/text_match_study/bert_similairity/)  **【推荐阅读】**
  - 阅读理由：BERT 在 语义相似度匹配任务 中的应用，可以由很多种方式，然而，你真的了解这些方式的区别和优缺点么？
  - 动机：BERT 在 语义相似度匹配任务 中的应用，可以常用 Sentence Pair Classification Task：使用 [CLS]、cosine similairity、sentence/word embedding、siamese network 方法，那么哪种是最佳的方式呢？你是否考虑过呢?
- [【关于 MPCNN】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/text_match_study/Multi-PerspectiveSentenceSimilarityModelingwithCNN/)
  - 论文：Multi-Perspective Sentence Similarity Modeling with Convolution Neural Networks
- [【关于 RE2】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/text_match_study/Multi-RE2_study/)
  - 论文：Simple and Effective Text Matching with Richer Alignment Features
  - 动机： 可以使用多个序列间比对层构建更强大的模型。 代替基于单个对准过程的比较结果进行预测，具有多个对准层的堆叠模型将保持其中间状态并逐渐完善其预测。**但是，由于底层特征的传播效率低下和梯度消失，这些更深的体系结构更难训练。** 
  - 介绍：一种快速强大的神经体系结构，具有用于通用文本匹配的多个对齐过程。 我们对以前文献中介绍的文本匹配方法中许多慢速组件的必要性提出了质疑，包括复杂的多向对齐机制，对齐结果的大量提炼，外部句法特征或当模型深入时用于连接堆叠块的密集连接。 这些设计选择会极大地减慢模型的速度，并且可以用重量更轻且效果相同的模型代替。 同时，我们重点介绍了有效文本匹配模型的三个关键组成部分。 这些组件（名称为RE2代表）是以前的对齐特征（残差矢量），原始点向特征（嵌入矢量）和上下文特征（编码矢量）。 其余组件可能尽可能简单，以保持模型快速，同时仍能产生出色的性能。
- [【关于 DSSM】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/text_match_study/cikm2013_DSSM/)
  - 论文：Deep Structured Semantic Model
  - 论文会议：CIKM2013
  - 问题：语义相似度问题
    - 字面匹配体现
      - 召回：在召回时，传统的文本相似性如 BM25，无法有效发现语义类 Query-Doc 结果对，如"从北京到上海的机票"与"携程网"的相似性、"快递软件"与"菜鸟裹裹"的相似性
      - 排序：在排序时，一些细微的语言变化往往带来巨大的语义变化，如"小宝宝生病怎么办"和"狗宝宝生病怎么办"、"深度学习"和"学习深度"；
    - 使用 LSA 类模型进行语义匹配，但是效果不好
  - 思路：
    - 利用 表示层 将 Query 和 Title 表达为低维语义向量；
    - 通过 cosine 距离来计算两个语义向量的距离，最终训练出语义相似度模型。
  - 优点
    - 减少切词的依赖：解决了LSA、LDA、Autoencoder等方法存在的一个最大的问题，因为在英文单词中，词的数量可能是没有限制，但是字母 n-gram 的数量通常是有限的
    - 基于词的特征表示比较难处理新词，字母的 n-gram可以有效表示，鲁棒性较强；
    - 传统的输入层是用 Embedding 的方式（如 Word2Vec 的词向量）或者主题模型的方式（如 LDA 的主题向量）来直接做词的映射，再把各个词的向量累加或者拼接起来，由于 Word2Vec 和 LDA 都是无监督的训练，这样会给整个模型引入误差，DSSM 采用统一的有监督训练，不需要在中间过程做无监督模型的映射，因此精准度会比较高；
    - 省去了人工的特征工程；
  - 缺点
    - word hashing可能造成冲突
    - DSSM采用了词袋模型，损失了上下文信息
    - 在排序中，搜索引擎的排序由多种因素决定，由于用户点击时doc的排名越靠前，点击的概率就越大，如果仅仅用点击来判断是否为正负样本，噪声比较大，难以收敛
- [【关于 ABCNN 】那些你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/text_match_study/TACL2016_ABCNN/)
  - 论文：ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs
  - 会议：TACL 2016
  - 论文方法：采用了CNN的结构来提取特征，并用attention机制进行进一步的特征处理，作者一共提出了三种attention的建模方法
- [【关于 ESIM 】那些你不知道的事 ](https://github.com/km1994/nlp_paper_study_text_match/tree/master/text_match_study/TACL2017_ESIM/)
  - 论文：Enhanced LSTM for Natural Language Inference
  - 会议：TACL2017
  - 自然语言推理（NLI: natural language inference）问题：
    - 即判断能否从一个前提p中推导出假设h
    - 简单来说，就是判断给定两个句子的三种关系：蕴含、矛盾或无关
  - 论文方法：
    - 模型结构图分为左右两边：
    - 左侧就是 ESIM，
    - 右侧是基于句法树的 tree-LSTM，两者合在一起交 HIM (Hybrid Inference Model)。
    - 整个模型从下往上看，分为三部分：
      - input encoding；
      - local inference modeling；
      - inference composition；
      - Prediction
- [【关于 BiMPM 】那些你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/text_match_study/IJCAI2017_BiMPM/)
  - 论文：Bilateral multi-perspective matching for natural language sentences
  - 会议：IJCAI2017
  - 方法：
    - Word Representation Layer:其中词表示层使用预训练的Glove或Word2Vec词向量表示, 论文中还将每个单词中的字符喂给一个LSTM得到字符级别的字嵌入表示, 文中使用两者构造了一个dd维的词向量表示, 于是两个句子可以分别表示为 P:[p1,⋯,pm],Q:[q1,⋯,qn].
    - Context Representation Layer: 上下文表示层, 使用相同的双向LSTM来对两个句子进行编码. 分别得到两个句子每个时间步的输出.
    - Matching layer: 对两个句子PP和QQ从两个方向进行匹配, 其中⊗⊗表示某个句子的某个时间步的输出对另一个句子所有时间步的输出进行匹配的结果. 最终匹配的结果还是代表两个句子的匹配向量序列.
    - Aggregation Layer: 使用另一个双向LSTM模型, 将两个匹配向量序列两个方向的最后一个时间步的表示(共4个)进行拼接, 得到两个句子的聚合表示.
- Prediction Layer: 对拼接后的表示, 使用全连接层, 再进行softmax得到最终每个标签的概率.
- [【关于 DIIN 】那些你不知道的事 ](https://github.com/km1994/nlp_paper_study_text_match/tree/master/text_match_study/T2017_DIIN/)
  - 论文：Densely Interactive Inference Network
  - 会议：TACL2017
  - 模型主要包括五层：嵌入层（Embedding Layer）、编码层（Encoding Layer）、交互层（Interaction Layer ）、特征提取层（Feature Extraction Layer）和输出层（Output Layer）
- [【关于 DC-BERT】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_text_match/tree/master/QA_study/SIGIR2020_DCBert/)
  - 论文名称：DC-BERT : DECOUPLING QUESTION AND DOCUMENT FOR EFFICIENT CONTEXTUAL ENCODING
  - 阅读理由：Bert 在 QA 上面的应用
  - 动机：Bert 无法处理传入问题的高吞吐量，每个问题都有大量检索到的文档；
  - 论文方法：具有双重BERT模型的解耦上下文编码框架：
    - 一个在线BERT，仅对问题进行一次编码；
    - 一个正式的BERT，对所有文档进行预编码并缓存其编码；
- [【关于 tBERT 】那些你不知道的事 ](https://github.com/km1994/nlp_paper_study_text_match/tree/master/QA_study/SIGIR2020_DCBert/)
   - 论文：tBERT: Topic Models and BERT Joining Forces for Semantic Similarity Detection
   - 会议：ACL2020
   - 论文地址：https://www.aclweb.org/anthology/2020.acl-main.630/
   - 论文代码：https://github.com/wuningxi/tBERT
   - 动机：未存在将主题模型和BERT结合的方法。 语义相似度检测是自然语言的一项基本任务理解。添加主题信息对于以前的特征工程语义相似性模型和神经网络模型都是有用的其他任务。在那里目前还没有标准的方法将主题与预先训练的内容表示结合起来比如 BERT。
   - 方法：我们提出了一种新颖的基于主题的基于BERT的语义相似度检测体系结构，并证明了我们的模型在不同的英语语言数据集上的性能优于强神经基线。我们发现在BERT中添加主题特别有助于解决特定领域的情况。

## 参考资料

1. [【ACL2020放榜!】事件抽取、关系抽取、NER、Few-Shot 相关论文整理](https://www.pianshen.com/article/14251297031/)
2. [第58届国际计算语言学协会会议（ACL 2020）有哪些值得关注的论文？](https://www.zhihu.com/question/385259014)
