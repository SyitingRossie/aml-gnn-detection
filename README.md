# Anti-Money Laundering Detection with GNN (Account-Level)
基于图神经网络的反洗钱账户识别项目，由开源基线迭代优化而来。本项目针对Kaggle开源基线进行全方位的重构，重点解决了传统GNN在金融风控场景中的时序数据泄露、特征工程疏漏、样本极度不平衡等核心缺陷。


## 一、基本信息
1、初次获取渠道：Kaggle Notebook 

2、原作Kaggle链接：https://www.kaggle.com/code/issacchanjj/anti-money-laundering-detection-with-gnn/notebook

3、原作GitHub仓库链接：https://github.com/issacchan26/AntiMoneyLaunderingDetectionWithGNN/tree/main

4、项目结构：dataset.py、model.py、train.py

5、使用IBM数据集：HI-Small_Trans.csv  

6、原始数据集字段：【'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1','Amount Received', 'Receiving Currency', 'Amount Paid','Payment Currency', 'Payment Format', 'Is Laundering'】  其中'Is Laundering'是标签，1=洗钱客户，0=正常客户

7、文件结构：
代码分为两个版本
- baseline_source: 第一版基线GNN代码
- optimized_code: 本人优化版本



## 二、核心优化
## 1.严格时序防泄露设计

**基线痛点**：原demo是基于所有交易统计节点标签和节点特征后，再进行随机切分训练集和验证集，忽略了金融风控实际场景中的时间先后顺序。

(1)、特征泄露：训练集和验证集均采用全局的节点特征，训练集的节点已经融合了验证集的交易数据，容易出现验证集指标虚高，模型在真实线下业务场景推理时效果大幅滑坡，泛化能力差。

(2)、标签泄露：训练集和验证集的标签都是全局标签。例如：a节点全局有过洗钱交易标签，那么该节点标签始终是1。反之，正常客户也是如此，导致训练阶段提前知道了未来的洗钱结果。

(3)、拓扑泄露：由于交易边用的是一张全局的图，训练集卷积了未来时序生成的交易边，致使训练节点利用了未来资金流向的网络结构。

**优化方案**：采用时序窗口切分（交易按时间排序后再切分）

(1)、严格时序切分：训练图基于0-70%（观测窗口）交易构建，预测70-85%（预测窗口）交易时段的账户洗钱标签；验证图基于0-85%交易构建，预测85-100%交易时段的账户标签。

(2)、独立构图：训练、验证阶段独立构图，独立统计节点特征，交易边特征，交易标签。

(3)、主节点筛选：限定训练目标节点为在预测窗口内有活动且在观察期有拓扑历史节点，暂不考虑冷启动节点。



## 2.特征工程重构

### 节点特征基线痛点
原demo节点特征仅由15种货币的进款均值、出款均值，以及bank列（LabelEncoder编码）构成（31维稀疏特征），存在严重的特征表达缺陷。
1.未量纲化且分布偏态：交易金额绝对值跨度大，直接使用原始金额的均值，会引发梯度震荡。
2.缺乏风控聚合意义：反洗钱的核心识别对象是拓扑层面的资金异常流动模式（如“多对一汇聚”、“快进快出”、“大额分拆”等），并非单一币种的静态金额。
3.类别编码缺陷：bank列使用LabelEncoder编码转化为连续整数,无意义地引入了顺序关系和数值大小关系。

### 节点特征优化方案
摒弃原31维稀疏特征，将节点特征重构为 **12维风控业务节点特征**，并在各时序窗口内独立进行log1p取对数并归一化处理，同时去掉bank列：

1.资金规模与体量
目标：捕捉大体量+小笔均的拆单洗钱模式
- total_amount_paid / total_amount_received：总转出 / 总转入金额 
- avg_amount_paid / avg_amount_received：笔均转出 / 笔均转入金额

#### (2) 资金流动与留存
目标：识别过路中转户（资金零留存）、高复杂度交易
- net_flow_ratio：资金净流转率，计算公式 $\boldsymbol{(Rec-Paid)/(Rec+Paid+1e^{-5})}$ 
- unique_currency_count：涉及交易货币种类去重总数 

#### (3) 图拓扑与交互度
目标：识别账户在网络中的归集与分发角色
- unique_out_accounts / unique_in_accounts：出度 / 入度去重对手账户数 

#### (4) 行为频次与时序
目标：捕捉高频拆单、自动化定时归集行为
- total_out_count / total_in_count：总转出 / 总转入笔数 
- avg_T_out / avg_T_in：对手平均转出 / 转入频次 

**数据标准化补充说明：**
1. 金额、频次类特征：先执行log1p()对数变换，再使用StandardScaler标准化处理
2. 比率、计数类特例：
- net_flow_ratio：本身取值区间[-1,1]，跳过log1p，仅标准化
- unique_currency_count：货币种类是小范围离散计数，无长尾分布，跳过log1p，仅标准化
3. 交易金额统一货币单位为USD计算的金额，使用的汇率是交易数据中的汇率总位数。

### 交易边特征基线痛点
原交易边特征(6维）：时间戳、转入金额、转入货币、转出金额、转出货币、转账方式
1. 转入货币、转出货币、转账方式使用LabelEncoder编码，引入了不存在的数值大小关系，干扰图模型权重学习与梯度更新。
2. 转入、转出金额未统一币种，不同货币的交易规模不具备可比性，且仅做简单对数压缩。
3. 时间戳直接使用MinMaxScaler归一化到[0,1]区间。当测试集新交易时间晚于训练集最大时间，归一化后数值会大于1，引发特征分布漂移。

### 交易边特征优化方案
重构为信息更密集的边特征，8维+转账方式（独热编码）
1. format_*：           转账方式字段，使用One‑Hot独热编码，规避类别特征产生虚假大小关系
2. Amount_USD_log：     交易金额统一换算为美元后(汇率使用交易数据的中位数），`log1p`对数压缩，消除币种差异、压缩金额长尾极值
3. hour_sin、hour_cos： 小时周期编码。将0‑23小时映射到单位圆，保留时间周期性，解决跨零点断层问题
4. dow_sin、dow_cos：   星期周期编码。将周一~周日映射到单位圆，保留星期周期规律，解决跨周断层问题
5. delta_last_in_log：  本笔交易距离账户上一笔转入的时间间隔(秒)，`log1p`对数压缩，捕捉资金流入频率
6. delta_last_out_log： 本笔交易距离账户上一笔转出的时间间隔(秒)，`log1p`对数压缩，识别短时间过桥快转、连续拆分交易行为
7. is_cross_currency：  是否为跨币种交易（1/0)

**数据标准化补充说明：**
1.交易金额、delta时间差，均进行对数压缩（在预处理阶段完成）
2.除转账方式(独热编码)、is_cross_currency外，均进行了标准化(StandardScaler)处理


## 3.样本数据不平衡的处理
(1)、原是在epoch循环之前一次性随机采样，那每轮epoch样本顺序都一样，多轮后容易过拟合。

3、不平衡样本加权采样：

(1)、采用WeightedRandomSampler加权有放回采样：洗钱节点设置采样权重=5，正常账户权重=1。WeightedRandomSampler为迭代器，每轮Epoch重新执行一次随机采样，避免模型过拟合到固定的采样结构。

1. 损失函数补偿：计算了训练集中的正负样本比例，并传入 BCEWithLogitsLoss 的 pos_weight（设置了最大截断值 15 防止梯度剧烈震荡）。
2. 评估指标选择：放弃了容易受负样本主导的 ROC-AUC，转而以 PR-AUC (Average Precision) 为核心监控指标。
3. 动态阈值选取：由于反洗钱业务更倾向于‘宁可错杀、不能放过’，我在验证集上通过最大化 $F_4$-Score（给 Recall 赋以 4 倍于 Precision 的权重）来动态搜索最佳决策阈值。”

## 4.GAT模型构建
**基线痛点**：
1. 原设计中应有边特征，但在模型里未加edge_dim参数，致使模型没有学习到交易边特征。
2. 原设计没有使用自还边，会导致卷积过程中没有融合节点自身的特征信息。
   
**优化方案**：
1. 补充edge_dim参数，两层卷积均卷交易边信息。
2. 两层卷积均设置自环边add_self_loops=True
3. 使用GATv2Conv代替GATConv，



## 5.优化器选择
原用SGD随机梯度下降。
5、优化器：优化器从SGD换成Adam。

## 6.优化指标选择
缺乏模型验证指标：在极度不平衡的数据集中仅使用Accuracy是无意义的(全预测为0就能达到99%的准确率）。

完善验证指标：增加PR-AUC（AP)、ROC-AUC、recall、precision、F4 score 


4、GAT卷积网络搭建：

(1)、GATConv中加入edge_dim参数，使模型没有学习交易边特征；加入自环边add_self_loops=True，加入自环边的交易边信息fill_value="mean"。

(2）、用GATv2Conv替代GATConv。


5、优化器：优化器从SGD换成Adam。

6、完善验证指标：增加PR-AUC（AP)、ROC-AUC、recall、precision、F4 score 


五、其他优化设想

目前ImprovedGAT.forward 中预留了return_embedding=True 开关，下一步可以写一段代码：抽取 GNN 的节点 Embedding，拼上原始节点的 tabular 特征，喂给 XGBoost / LightGBM 做二阶段分类。

六、局限

1. 当前是静态基线，工业级要用滚动时间窗口动态构图。
2. 后期需要再建一个边分类的模型，作为另一个任务。



