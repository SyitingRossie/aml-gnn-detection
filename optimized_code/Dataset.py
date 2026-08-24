import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from torch.utils.data import WeightedRandomSampler  
from torch_geometric.nn import GATConv, Linear
from typing import Optional,Callable
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch_geometric.data import InMemoryDataset,Data
import numpy as np
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,precision_recall_curve,auc)
import os
import random
import matplotlib.pyplot as plt


class AML_to_Graph(InMemoryDataset):
    def __init__(self, root, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)   #检查 root/processed/ 目录下是否存在：'data_train.pt' 和 'data_val.pt'  没有就自动调用 process() 去生成并保存文件；如果有就直接跳过 process()。
        self.train_data, _ = torch.load(self.processed_paths[0], weights_only=False)  #正式打开文件并加载到内存
        self.val_data, _ = torch.load(self.processed_paths[1], weights_only=False)    #正式打开文件并加载到内存

    @property
    def raw_file_names(self):
        return ["HI-Small_Trans.csv"]

    @property
    def processed_file_names(self):
        return ['data_train.pt', 'data_val.pt']

    #这里注销的原因是，原demo是把currency进行编码0，1，2，3……，但我们已经把货币统一成美元了，所以不需要了。且原先的format转行方式会编码，但我们用热独替代了。
    # def df_label_encoder(self, df, columns):
    #     le = preprocessing.LabelEncoder()
    #     for i in columns:
    #         df[i] = le.fit_transform(df[i])
    #     return df

    def process_payment_format(self,df):
        format_dummies = pd.get_dummies(df['Payment Format'], prefix='format', dtype=float)
        # print(f"[Step 1 完成] 生成的渠道 One-Hot 列 ({format_dummies.shape[1]} 维):")
        # print(list(format_dummies.columns))
        return format_dummies

    # 汇率字典：基于跨币种交易的中位数计算各货币相对于基准货币
    def build_fx_rates(self, df_base, base_currency="US Dollar"):
        fx_rates = {base_currency: 1.0}
        df_cross = df_base[df_base['Payment Currency'] != df_base['Receiving Currency']].copy()

        from_base = df_cross[df_cross['Payment Currency'] == base_currency]
        if not from_base.empty:
            rates = (from_base['Amount Received'] / (from_base['Amount Paid'] + 1e-5)).groupby(
                from_base['Receiving Currency']).median()
            for curr, rate in rates.items():
                if rate > 0:
                    fx_rates[curr] = 1.0 / rate  # 转为统一基准币种的乘数

        to_base = df_cross[df_cross['Receiving Currency'] == base_currency]
        if not to_base.empty:
            rates = (to_base['Amount Received'] / (to_base['Amount Paid'] + 1e-5)).groupby(
                to_base['Payment Currency']).median()
            for curr, rate in rates.items():
                if rate > 0 and curr not in fx_rates:
                    fx_rates[curr] = rate

        return fx_rates

    # 继续时间戳的处理,此前已经按timestamp排序过了。       #最后得到解耦时间的两列
    def compute_decoupled_time_deltas(self,df):
        df = df.copy()
        # 0. 确保索引有序且生成交易唯一 ID
        df['tx_id'] = np.arange(len(df))

        # 1. 第一步：构建双向统一时间轴
        out_events = df[['tx_id', 'Account', 'ts_sec']].rename(columns={'Account': 'account_id'})
        in_events = df[['tx_id', 'Account.1', 'ts_sec']].rename(columns={'Account.1': 'account_id'})
        out_events['is_out'] = 1
        in_events['is_out'] = 0

        full_events = pd.concat([out_events, in_events], axis=0)
        full_events = full_events.sort_values(by=['ts_sec', 'tx_id']).reset_index(drop=True)   #在这一步的时候交易id顺序已经打乱了，是按照sec时间戳顺序排序的。

        # 2. 第二步：信号分离
        full_events['in_ts'] = full_events['ts_sec'].where(full_events['is_out'] == 0)
        full_events['out_ts'] = full_events['ts_sec'].where(full_events['is_out'] == 1)

        # 3. 第三步：状态记忆传承 (ffill)
        # (1) 距离最近一次进账时间戳
        full_events['last_in_ts'] = full_events.groupby('account_id')['in_ts'].ffill()  #继承本行，如本行null，继承上一次有行的数。

        # (2) 距离上一次出账时间戳 (必须先 shift(1) 排除当前笔自身)
        full_events['out_ts_shifted'] = full_events.groupby('account_id')['out_ts'].shift(1)   #用上一行
        full_events['last_out_ts'] = full_events.groupby('account_id')['out_ts_shifted'].ffill()

        # 4. 第四步：只保留出账事件，计算独立时间差
        sender_events = full_events[full_events['is_out'] == 1].copy()

        # 秒数差计算 + 18天缺失值兜底
        delta_in = (sender_events['ts_sec'] - sender_events['last_in_ts']).fillna(1555200.0)
        delta_out = (sender_events['ts_sec'] - sender_events['last_out_ts']).fillna(1555200.0)

        # 5. 第五步：对数平滑并拼回原表
        sender_events['time_delta_last_in_log'] = np.log1p(delta_in)    #sender_events不是原始的顺序，因为full_events.sort_values(by=['ts_sec', 'tx_id'])
        sender_events['time_delta_last_out_log'] = np.log1p(delta_out)

        temp = sender_events[["tx_id", "time_delta_last_in_log", "time_delta_last_out_log"]]
        df = df.merge(temp, on="tx_id", how="left")
        fill_log = np.log1p(1555200.0)
        df['time_delta_last_in_log'] = df['time_delta_last_in_log'].fillna(fill_log)  #防御性
        df['time_delta_last_out_log'] = df['time_delta_last_out_log'].fillna(fill_log)#防御性

        return df

    def preprocess(self, df):
        df = df.copy()
        # le_curr = preprocessing.LabelEncoder()
        # all_curr = pd.concat([df["Payment Currency"], df["Receiving Currency"]])  # 默认上下堆叠axis=0
        # le_curr.fit(all_curr)
        # df["Payment Currency"] = le_curr.transform(df["Payment Currency"])
        # df["Receiving Currency"] = le_curr.transform(df["Receiving Currency"])
        # mapping_curr = {cls: idx for idx, cls in
        #                 enumerate(le_curr.classes_)}  # le_curr.classes_ 得到类似array(['USD','RMB',,,,])一维数组
        # currency_ls = [idx for idx, cls in enumerate(le_curr.classes_)]

        # 支付渠道单独编码
        # le_form = preprocessing.LabelEncoder()
        # df["Payment Format"] = le_form.fit_transform(df["Payment Format"])
        # 时间戳转成秒数，并按时间排序。
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])     #变成时间格式
        df = df.sort_values(by='Timestamp', kind='mergesort').reset_index(drop=True)
        df['ts_sec'] = df["Timestamp"].astype('int64') // 10 ** 9

        df["Account"] = df["From Bank"].astype(str) + "_" + df["Account"]
        df["Account.1"] = df["To Bank"].astype(str) + "_" + df["Account.1"]

        #转账方式热独编码
        format_dummies = pd.get_dummies(df['Payment Format'], prefix='format', dtype=float)        # print(list(format_dummies.columns))
        df = pd.concat([df, format_dummies], axis=1)

        #是否跨币种交易
        df['is_cross_currency'] = (df['Payment Currency']!= df['Receiving Currency']).astype(float)

        #统一折算成usd的交易金额,并取对数
        fx_rates = self.build_fx_rates(df, base_currency="US Dollar")
        amount_paid_usd = df['Amount Paid'] * df['Payment Currency'].map(fx_rates)
        df['Amount_USD_log'] = np.log1p(amount_paid_usd)

        # 提取小时
        hour = df["Timestamp"].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

        # 提取星期几 (Monday=0, Sunday=6) 并计算弧度与 sin/cos
        dow = df["Timestamp"].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)

        df=self.compute_decoupled_time_deltas(df)   #最后得到解耦时间的两列

        return df

    # 4.列出所有账户，去重，这里不打标签了。
    def get_all_accounts(self, df):
        ldf = df[["Account", "From Bank"]].rename(columns={"From Bank": "Bank"})
        rdf = df[["Account.1", "To Bank"]].rename(columns={"To Bank": "Bank", "Account.1": "Account"})

        accounts_df = pd.concat([ldf, rdf],axis=0,ignore_index=True)   #ignore_index=True) 时，Pandas 会丢弃旧索引，重新编号。
        accounts_df = accounts_df.drop_duplicates(subset=["Account"]).reset_index(drop=True)
        account_id_map={acc:idx for idx, acc in enumerate(accounts_df["Account"])}  #需要用账号去查数字。
        print(f"全局唯一账户总数: {len(account_id_map)}")

        return accounts_df,account_id_map    #该表有2列【Account,Bank】

    def financial_aggregate(self,df,accounts,fx_rates):   #accounts表：Account（”银行_账号“）、Bank（LabelEncoder编码）两列
        acc= accounts[["Account"]].copy()
        df_c= df.copy()

        # 匹配不到汇率的未知货币（如 Val 集新出现的货币），默认系数取 1.0

        #total_amount_paid : 总付出金额 (对数压缩) 、total_amount_received : 总接收金额 (对数压缩)
        # avg_amount_paid : 笔均付出金额 (对数压缩) 、avg_amount_received :  笔均接收金额 (对数压缩)
        paid_rate = df_c["Payment Currency"].map(fx_rates).fillna(1.0)
        rec_rate = df_c["Receiving Currency"].map(fx_rates).fillna(1.0)
        df_c["Amount_Paid_USD"]=df_c["Amount Paid"]*paid_rate
        df_c["Amount_Received_USD"] = df_c["Amount Received"] * rec_rate

        paid_stats = df_c.groupby("Account")["Amount_Paid_USD"].agg(total_amount_paid="sum",avg_amount_paid="mean").reset_index()
        rec_stats = df_c.groupby("Account.1")["Amount_Received_USD"].agg(total_amount_received="sum",avg_amount_received="mean").reset_index().rename(columns={"Account.1": "Account"})
        acc = acc.merge(paid_stats, on="Account", how="left")
        acc = acc.merge(rec_stats, on="Account", how="left")

        #统计一个账户涉及多少种货币
        paid_curr = df_c[["Account", "Payment Currency"]].rename(columns={"Payment Currency": "Currency"})
        recv_curr = df_c[["Account.1", "Receiving Currency"]].rename(columns={"Account.1": "Account", "Receiving Currency": "Currency"})
        all_curr = pd.concat([paid_curr, recv_curr], axis=0, ignore_index=True)
        curr_stats = all_curr.groupby("Account")["Currency"].nunique().reset_index(name="unique_currency_count")  #新生成列列名
        acc = acc.merge(curr_stats, on="Account", how="left")

        cols1=["total_amount_paid","avg_amount_paid","total_amount_received","avg_amount_received", "unique_currency_count"]  # unique_currency_count 币种数量不建议做 log1p，保留原频次或单独处理更好
        acc[cols1]=acc[cols1].fillna(0)   #防止只有单向交易的客户会有列是空的，比如只有转出的账户，转入总额会是0。

        #先计算资金净流转率，再对数压缩"total_amount_received"和"total_amount_paid"
        rec = acc["total_amount_received"]
        paid = acc["total_amount_paid"]
        acc["net_flow_ratio"] = (rec - paid) / (rec + paid + 1e-5)

        cols2 = ["total_amount_paid", "avg_amount_paid", "total_amount_received", "avg_amount_received"]
        acc[cols2] = np.log1p(acc[cols2])   #节点特征对金额指标取对数压缩

        cols_order = [
            "Account",
            "total_amount_paid",
            "total_amount_received",
            "avg_amount_paid",
            "avg_amount_received",
            "net_flow_ratio",
            "unique_currency_count"
        ]
        return acc[cols_order]

    # 改进：在特征节点新增3列，账户的资金来源账户个数，账户的出钱账户个数，总交易笔数
    # 改进：在特征节点新增宏观拓扑统计特征
    def topological_aggregate(self, df, node_df):
        acc_df = node_df.copy()

        out_deg = df.groupby("Account")["Account.1"].nunique().reset_index(name="unique_out_accounts")
        in_deg = df.groupby("Account.1")["Account"].nunique().reset_index(name="unique_in_accounts")
        in_deg = in_deg.rename(columns={"Account.1": "Account"})

        out_cnt = df.groupby("Account").size().reset_index(name="total_out_count")
        in_cnt = df.groupby("Account.1").size().reset_index(name="total_in_count")
        in_cnt = in_cnt.rename(columns={"Account.1": "Account"})

        acc_df = acc_df.merge(out_deg, on="Account", how="left")
        acc_df = acc_df.merge(in_deg, on="Account", how="left")
        acc_df = acc_df.merge(out_cnt, on="Account", how="left")
        acc_df = acc_df.merge(in_cnt, on="Account", how="left")

        topo_cols = ["unique_out_accounts", "unique_in_accounts", "total_out_count", "total_in_count"]
        acc_df[topo_cols] = acc_df[topo_cols].fillna(0)

        acc_df["avg_T_out"] = np.where(acc_df["unique_out_accounts"] > 0,   #防止unique_out_accounts0，被除数，除数都是0的情况
            acc_df["total_out_count"] / acc_df["unique_out_accounts"],0.0)
        acc_df["avg_T_in"] = np.where(acc_df["unique_in_accounts"] > 0,
            acc_df["total_in_count"] / acc_df["unique_in_accounts"],0.0)

        density_cols = ["avg_T_out", "avg_T_in"]
        acc_df[density_cols] = np.log1p(acc_df[density_cols])
        acc_df[topo_cols] = np.log1p(acc_df[topo_cols])  # 对数压缩：防止极端大户或洗钱中转站的数值过大导致梯度爆炸

        return acc_df

    # 6.node features
    def get_node_attr(self, accounts, df, fx_rates,local_id_map):
        fin_df = self.financial_aggregate(df, accounts, fx_rates)
        topo_df = self.topological_aggregate(df, fin_df)

        #把节点顺序按照局部account id映射表来排序
        topo_df["local_id"]=topo_df["Account"].map(local_id_map)
        topo_df=topo_df.sort_values("local_id").reset_index(drop=True)
        node_x_df = topo_df.drop(columns=["Account","local_id"])

        return node_x_df.to_numpy(dtype=np.float32)
        # return torch.tensor(node_x_df.to_numpy(), dtype=torch.float)

    # 打标
    def node_labels_and_mask(self, accounts, df_label_window): #这里account就是0-70%和0-85%的
        active_accounts = set(df_label_window['Account']).union(set(df_label_window['Account.1']))
        df_laundering = df_label_window[df_label_window['Is Laundering'] == 1]
        laundering_accounts = set(df_laundering['Account']).union(set(df_laundering['Account.1']))

        # 直接向量化生成 mask 和 y
        mask = torch.tensor([acc in active_accounts for acc in accounts], dtype=torch.bool)
        y = torch.tensor([1.0 if acc in laundering_accounts else 0.0 for acc in accounts], dtype=torch.float)

        return y, mask

    # 7.Edge features,这步会把原本df的Is Laundering列去掉。
    def get_edge_attr(self, df, local_id_map):
        df = df.copy()
        # accounts = accounts.reset_index(drop=True)  # 确保从0开始编号，把所有唯一账户重新编码，丢弃原索引。谨慎。
        # accounts["ID"] = accounts.index
        # mapping_dict = dict(zip(accounts["Account"], accounts["ID"]))

        # 1. 构建边索引 edge_index
        from_np = df["Account"].map(local_id_map).to_numpy()
        to_np = df["Account.1"].map(local_id_map).to_numpy()
        edge_index = torch.tensor(np.array([from_np, to_np]), dtype=torch.long)

        # 备注：原边特征['Timestamp', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format'],边特征是6列。
        # 改进：df数据预处理:调时间格式，Account调成Bank+Account，转账方式热独编码、是否跨币种交易（1/0)、折算成usd的交易金额并取对数。
        # 此时原df.columns: Index(['Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1',
        #        'Amount Received', 'Receiving Currency', 'Amount Paid',
        #        'Payment Currency', 'Payment Format', 'Is Laundering', 'format_Cheque',
        #        'format_Credit Card', 'format_Reinvestment', 'is_cross_currency',
        #        'Amount_USD_log'],
        #       dtype='str')
        # 更改后df.columns: Index(['Timestamp','is_cross_currency','Amount_USD_log',
        #                         'format_Cheque','format_Credit Card', 'format_Reinvestment'])
        format_cols = [col for col in df.columns if col.startswith("format_")]
        edge_feature_cols = ['is_cross_currency',
                             'Amount_USD_log',
                             'hour_sin', 'hour_cos',  # 替代原 Timestamp：日周期
                             'dow_sin', 'dow_cos',  # 替代原 Timestamp：周周期
                             'time_delta_last_in_log','time_delta_last_out_log' # 替代原 Timestamp：双向资金停留过桥时间差,是否快出
                            ] + format_cols
        # edge_attr = torch.from_numpy(df[edge_feature_cols].to_numpy(dtype=np.float32))
        # edge_attr_np = df[edge_feature_cols].to_numpy(dtype=np.float32)
        edge_attr=df[edge_feature_cols]
        return edge_attr, edge_index         # edge_attr的账户和银行的信息都去掉了，edge_index是“账户+银行”唯一编码的序号，从0开始。

    def process(self):
        # df=pd.read_csv(self.raw_paths[0])                    #内存够的话直接读全部，更快。
        chunk_list = []
        batch_size = 100000
        reader = pd.read_csv(self.raw_paths[0], chunksize=batch_size)
        for chunk in reader:
            chunk_list.append(chunk)
        df = pd.concat(chunk_list, axis=0, ignore_index=True)

        df= self.preprocess(df)
        all_accounts,global_id_map= self.get_all_accounts(df)

        # 2. 按时间严格切分 3 个窗口的数据段
        n_total = len(df)
        idx_70 = int(n_total * 0.70)
        idx_85 = int(n_total * 0.85)

        df_train = df.iloc[:idx_70].copy()        # 0% ~ 70% 的历史交易：节点特征、边特征
        df_val = df.iloc[:idx_85].copy()          # 0% ~ 85% 的历史交易

        df_label_train = df.iloc[idx_70:idx_85]  # 70%-85%：训练集节点标签来源：y、mask
        df_label_val = df.iloc[idx_85:]          # 85%-100%：验证集节点标签来源

        # ------------------------------构建第一个图：训练集------------------------------
        # 统计节点
        accounts_train,local_id_map_train = self.get_all_accounts(df_train)           #节点，这里节点的顺序就是0,1,2,3……排好的了

        #训练集（切好之后再去得到汇率字典）
        fx_rates_train = self.build_fx_rates(df_train, base_currency="US Dollar")   #汇率
        node_x_train = self.get_node_attr(accounts_train, df_train, fx_rates_train,local_id_map_train) #节点特征
        edge_attr_train, edge_index_train = self.get_edge_attr(df_train,local_id_map_train)  #边特征，边index
        y_train, train_mask = self.node_labels_and_mask(accounts_train["Account"], df_label_train)    # y标签 #accounts_train有两列：Account和bank列

        # ------------------------------构建第二个图：验证集------------------------------
        #统计节点
        accounts_val,local_id_map_val = self.get_all_accounts(df_val)           #节点，这里节点的顺序就是0,1,2,3……排好的了

        # x节点特征  （切好之后再去得到汇率字典）
        fx_rates_val = self.build_fx_rates(df_val, base_currency="US Dollar")   #汇率
        node_x_val = self.get_node_attr(accounts_val, df_val, fx_rates_val,local_id_map_val) #节点特征
        edge_attr_val, edge_index_val = self.get_edge_attr(df_val,local_id_map_val)  #边特征，边index
        y_val, val_mask = self.node_labels_and_mask(accounts_val["Account"], df_label_val) # y标签 #accounts_val有两列：Account和bank列

        # ----------------------------------先把数据标准化，再统一转为张量存入 Data 对象------------------------------------------
        scaler_node=StandardScaler()
        node_x_train_scaled=scaler_node.fit_transform(node_x_train)
        node_x_val_scaled=scaler_node.transform(node_x_val)

        # 边特征拆分列名，没有'is_cross_currency',没有'交易方式'
        num_edge_cols = [
            'Amount_USD_log', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'time_delta_last_in_log', 'time_delta_last_out_log'
        ]
        # 自动获取所有以 format_ 开头的渠道列 + 跨币种标记
        cat_edge_cols = ['is_cross_currency'] + [col for col in edge_attr_train.columns if col.startswith("format_")]

        scaler_edge=StandardScaler()
        edge_attr_train_scaled=scaler_edge.fit_transform(edge_attr_train[num_edge_cols])     #fit_transform出来就是numpy格式
        edge_attr_val_scaled=scaler_edge.transform(edge_attr_val[num_edge_cols])

        # 4. 离散/二值特征组：转账交易，不做标准化，直接转为 numpy
        edge_cat_train = edge_attr_train[cat_edge_cols].to_numpy(dtype=np.float32)
        edge_cat_val = edge_attr_val[cat_edge_cols].to_numpy(dtype=np.float32)

        # 5. 横向拼接（np.hstack），形成最终的边特征矩阵
        edge_attr_train_scaled = np.hstack([edge_attr_train_scaled, edge_cat_train])
        edge_attr_val_scaled = np.hstack([edge_attr_val_scaled, edge_cat_val])

        #节点特征，和交易边特征，统一转为张量（转成tensor）
        #这里之所以必须显式写上 dtype=torch.float，是因为 StandardScaler.fit_transform() 默认会把输出结果强行转换为 float64（双精度浮点数），即使你之前用 to_numpy(dtype=np.float32) 把它转成了 32 位。
        #在 PyTorch 中，torch.float 是 torch.float32 的简写形式。因为神经网络的权重（Model Weights）默认都是 32 位浮点数（float32），所以输入的数据张量也必须统一为 torch.float（或者 torch.float32）。
        node_x_train = torch.tensor(node_x_train_scaled, dtype=torch.float)
        node_x_val = torch.tensor(node_x_val_scaled, dtype=torch.float)
        edge_attr_train = torch.tensor(edge_attr_train_scaled, dtype=torch.float)
        edge_attr_val = torch.tensor(edge_attr_val_scaled, dtype=torch.float)

        # ------------------------------封装为 PyG Data 对象-----------------------------
        data_train = Data(
            x=node_x_train,
            edge_index=edge_index_train,
            edge_attr=edge_attr_train,
            y=y_train,
            train_mask=train_mask  # 此处掩码都是True
        )
        data_val = Data(
            x=node_x_val,
            edge_index=edge_index_val,
            edge_attr=edge_attr_val,
            y=y_val,
            val_mask=val_mask      #此处掩码都是True
        )

        # 独立序列化保存
        # 1. 训练图单独 collate 并存入 processed_paths[0]
        data, slices = self.collate([data_train])
        torch.save((data, slices), self.processed_paths[0])

        # 2. 验证图单独 collate 并存入 processed_paths[1]
        data, slices = self.collate([data_val])
        torch.save((data, slices), self.processed_paths[1])
