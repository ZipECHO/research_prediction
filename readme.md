# research_pred
通过分析近些年的不用学科领域的文章的发表情况，对某会议/学者的未来的研究热点进行预测

# DataSource：arxiv Dataset
该数据集包括1.7M+的论文数据集，超过一半的论文都有其发表的会议/期刊，所有的论文都有其作者。详情可以查看：[arxiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
# 方法
1. 利用论文的分类`category`字段来构建不同的研究点网络，使用网络表征学习方法对该网络的节点进行表征学习
2. 在学者/期刊的发表论文序列中设置不同的时间窗口，并进行简单的的预测下一步发表的论文类型