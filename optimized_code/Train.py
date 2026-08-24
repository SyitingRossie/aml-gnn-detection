# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = AML_to_Graph("/kaggle/working")

train_data = dataset.train_data
val_data = dataset.val_data

# 找到train_mask中所有为True的节点索引
train_node_idx = torch.nonzero(train_data.train_mask).flatten()    #torch.nonzero() 返回的结果也强制是二维的 [M, 1],所以要flattern变1维。
train_y = train_data.y[train_node_idx]

#自动计算正负样本比例，作为损失函数的权重补偿
pos_num = (train_y == 1).sum().item()
neg_num = (train_y == 0).sum().item()

raw_pos_weight = neg_num / max(pos_num, 1)
pos_weight_val = min(raw_pos_weight, 45.0)  
print(neg_num,pos_num,raw_pos_weight)

pos_weight = torch.tensor([pos_weight_val], device=device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model = ImprovedGAT(in_channels=train_data.num_features,
                    hidden_channels=16, out_channels=1, heads=8,
                    edge_dim=train_data.num_edge_features
                    ).to(device)

# 改进：优化器从SGD换成Adam！  SGD随机梯度下降
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)  # 学习率衰减：每20轮缩小学习率，后期收敛更平稳

train_input_nodes = torch.nonzero(train_data.train_mask).flatten()
val_input_nodes = torch.nonzero(val_data.val_mask).flatten()

train_loader = NeighborLoader(train_data,
                              num_neighbors=[30] * 2,
                              batch_size=256,
                              input_nodes=train_input_nodes,
                              shuffle=True)                         
val_loader = NeighborLoader(val_data,                                
                            num_neighbors=[30] * 2,
                            batch_size=256,
                            input_nodes=val_input_nodes,
                            shuffle=False)                        

best_val_ap = 0.0  #初始化
best_model_path = "best_aml_gat_model.pth"
epoch=100

for ep in range(epoch):
    total_loss = 0
    model.train()

    for t in train_loader:                  
        train_batch = t.to(device)         
        optimizer.zero_grad()                
        logits = model(train_batch.x, train_batch.edge_index,train_batch.edge_attr) 
        train_seed_num = train_batch.batch_size   

        seed_logits = logits[:train_seed_num]
        seed_y = train_batch.y[:train_seed_num].float().view_as(seed_logits)

        loss = criterion(seed_logits, seed_y)        
        loss.backward()  
                
        # 新增梯度剪裁，防止梯度爆炸震荡
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    
        optimizer.step()          
        total_loss += loss.item()  

    if ep % 10 == 0 or ep == epoch - 1:
        print(f"Epoch:{ep:01d},本轮epoch每个训练集的batch平均loss:{total_loss /len(train_loader)},每个epoch总批次数量:{len(train_loader)}")
        model.eval()
        y_true_all = []
        y_prob_all = []
        with torch.no_grad():        # 关闭梯度计算
            for val in val_loader:  
                val = val.to(device)
                logits = model(val.x, val.edge_index,val.edge_attr)  # 二维张量，pred=[[0.8],[0.65],[0.7],[0.1],[0.4]]类似，shape是（256，1）
                pred = torch.sigmoid(logits)                         # 压成概率    #如果有warning，换成pred = torch.sigmoid(logits)标准

                seed_num = val.batch_size
                gt_seed = val.y[:seed_num]
                pred_seed = pred[:seed_num]

                y_true_all.extend(gt_seed.cpu().numpy())               # gt_seed是一维张量[256,]    y_true_all所有客户的标签   #转numpy之前是tensor      #真实标签是一维的
                y_prob_all.extend(pred_seed.cpu().numpy().flatten())   #把可能带维度的二维预测概率（如 [256, 1]）拉平变成一维数组（[256,]），方便后续拼接。    #概率之前logis是二维的

        # 核心：利用验证集的预测概率动态寻找最优截断阈值
        y_true_arr = np.array(y_true_all)
        y_prob_arr = np.array(y_prob_all)
        if len(np.unique(y_true_arr)) > 1:
            precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
            # 优化：画出PR-AUC图
            pr_auc = auc(recalls, precisions)
            AP = average_precision_score(y_true_arr, y_prob_arr)  # 计算PR-AUC（Area Under Precision-Recall Curve）
            roc_auc = roc_auc_score(y_true_arr, y_prob_arr)       # 这里是预测概率

            precisions_sub = precisions[:-1]
            recalls_sub = recalls[:-1]

            beta = 4
            f_beta_scores = (1 + beta ** 2) * (precisions_sub * recalls_sub) / (beta ** 2 * precisions_sub + recalls_sub + 1e-10)
            best_idx = np.argmax(f_beta_scores)
            best_threshold = thresholds[best_idx]
            best_f_score = f_beta_scores[best_idx]  

            # 用找到的最佳阈值生成最终的 0/1 预测标签,此步骤的时候已经确定了阈值。
            y_pred_arr = (y_prob_arr >= best_threshold).astype(float)  # 这里要注意：一定是>=,因为precision_recall_curve内部是大于等于阈值才是洗钱，才计算的recall 和precision，所以要统一。

            acc = accuracy_score(y_true_arr, y_pred_arr)      
            precision = precision_score(y_true_arr, y_pred_arr, zero_division=0)
            recall = recall_score(y_true_arr, y_pred_arr, zero_division=0)

            print(f"验证集Acc：{acc:.4f}|PR-AUC（AP): {AP:.4f}|PR-AUC: {pr_auc:.4f}|ROC-AUC:{roc_auc:.4f}")
            print(f"Best Threshold: {best_threshold:.4f}|Precision:{precision:.4f}|recall:{recall:.4f},F4 score:{best_f_score:.4f}")
            if AP > best_val_ap:            #初始化 best_val_ap=0.0
                best_val_ap =AP
                torch.save({'model_state_dict': model.state_dict(),'best_threshold': best_threshold},'best_aml_model.pth')

                pos_ratio = np.sum(y_true_arr == 1) / len(y_true_arr)
                plt.figure(figsize=(8, 6))
                plt.plot(recalls, precisions, color="#1f77b4", lw=2, label=f"PR curve")
                plt.hlines(y=pos_ratio, xmin=0, xmax=1, color="red", linestyle="--", label="Random Baseline")
                plt.xlabel("Recall", fontsize=12)
                plt.ylabel("Precision", fontsize=12)
                plt.title("AML Model Precision-Recall cure", fontsize=14)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.legend(loc="best")
                plt.grid(alpha=0.3)
                plt.savefig(f"pr_curve_epoch{ep}.png", dpi=150)
                print(f"🌟 发现更好的模型！已保存。当前最高 PR-AUC: {best_val_ap:.4f}")
            plt.close()  
    scheduler.step()  # 每轮更新学习率，逐步缩小步长
