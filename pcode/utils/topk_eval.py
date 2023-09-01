from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader


class TopkEval():
    def __init__(self, dataset, train_data, test_data, k_list=[20]):
        topk_eval=Path(f"data/{dataset}/{dataset}.topk_eval")
        if topk_eval.exists():
            obj = torch.load(topk_eval)
            self.user_list = obj.user_list
            self.train_record = obj.train_record
            self.test_record = obj.test_record
            self.item_set = obj.item_set
            self.test_dataset = obj.test_dataset
            self.train_item_mask = obj.train_item_mask
            self.test_pos_item_binary = obj.test_pos_item_binary
            self.n_test_pos_item = obj.n_test_pos_item
        else:
            # only test 100 user
            user_num = 100
            train_record = self.get_user_record(train_data)
            test_record = self.get_user_record(test_data)
            user_list = list(set(train_record.keys()) & set(test_record.keys()))
            if len(user_list) > user_num:
                user_list = np.random.choice(user_list, size=user_num, replace=False)
            n_item = max(max(train_data[:][0][1]), max(test_data[:][0][1]))
            item_set = set(list(range(n_item)))
            test_dataset = torch.vstack([torch.tensor(user_list).repeat([n_item, 1]).t().flatten(),
                                         torch.arange(n_item).repeat([len(user_list)])]).t()
            train_item_mask = torch.zeros([user_num, n_item])
            test_pos_item_binary = torch.zeros([user_num, n_item])
            for i, user in enumerate(user_list):
                train_item_mask[i][list(train_record[user])] = 1
                test_pos_item_binary[i][list(test_record[user])] = 1


            self.user_list = user_list
            self.train_record = train_record
            self.test_record = test_record
            self.item_set = item_set
            self.test_dataset = test_dataset
            self.train_item_mask = train_item_mask
            self.test_pos_item_binary = test_pos_item_binary
            self.n_test_pos_item = torch.tensor([len(self.test_record[self.user_list[i]]) for i in range(len(
                self.user_list))])
            torch.save(self, topk_eval)
        self.k_list = k_list

    def eval(self, model, comm_round,device='cuda',batch_size=None, test_by_item=False):
        with torch.no_grad():
            origin_decive= next(model.parameters()).device
            model.to(device)
            model.eval()
            precision_list = {k: [] for k in self.k_list}
            recall_list = {k: [] for k in self.k_list}
            ndcg_list = {k: [] for k in self.k_list}
            if test_by_item:
                for i in self.user_list:
                    test_item_list = torch.tensor(list(self.item_set - self.train_record[i]))
                    score_list = []
                    for items in DataLoader(dataset=test_item_list, batch_size=batch_size, shuffle=False):
                        data_batch = [[i] * items.shape[0], items]
                        score_list.append(model(data_batch))
                    _, indices = torch.concat(score_list).sort(descending=True)
                    item_sorted = test_item_list[indices]
                    binary_hit = torch.tensor([i in self.test_record[i] for i in item_sorted])

                    for k in self.k_list:
                        hit_num = binary_hit[:k].sum()
                        precision_list[k].append(hit_num / k)
                        recall_list[k].append(hit_num / len(self.test_record[i]))
                        ndcg_list[k].append(self.ndcg_at_k_batch(binary_hit, k))
            else:
                n_item = len(self.item_set)
                score_list = []
                if batch_size is None:
                    # batch_size = len(self.test_dataset)
                    batch_size=n_item*10000
                for data_batch in DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False):
                    data_batch= data_batch.to(device)
                    score_list.append(model([data_batch[:, 0], data_batch[:, 1]]))
                score = torch.concat(score_list).reshape(-1, n_item)
                score[self.train_item_mask] = -np.inf
                _, indices = score.sort(axis=1, descending=True)
                binary_hit = torch.zeros_like(indices,device=device)
                for i in range(indices.shape[0]):
                    binary_hit[i]=self.test_pos_item_binary.to(device)[i][[indices[i]]]
                    # user_id = self.user_list[i]
                    # for j in range(indices.shape[1]):
                    #     if indices[i, j] in self.test_record[user_id]:
                    #         binary_hit[i, j] = True
                for k in self.k_list:
                    hit_num = binary_hit[:, :k].sum(axis=1)
                    precision_list[k].append(hit_num / k)
                    recall_list[k].append(hit_num / self.n_test_pos_item.to(device))
                    ndcg_list[k].append(self.ndcg_at_k_batch(binary_hit, k))

            precision = [torch.stack(precision_list[k]).mean().to(origin_decive) for k in self.k_list]
            recall = [torch.stack(recall_list[k]).mean().to(origin_decive) for k in self.k_list]
            ndcg = [torch.stack(ndcg_list[k]).mean().to(origin_decive) for k in self.k_list]

        model.to(origin_decive)
        for i, k in enumerate(self.k_list):
            wandb.log({f"precision@{k}": precision[i], f"recall@{k}": recall[i], f"ndcg@{k}": ndcg[i],'comm_round': comm_round})
        return [precision, recall, ndcg]

    def get_user_record(self, data):
        user_history_dict = dict()
        for interaction in data:
            user = int(interaction[0][0])
            item = int(interaction[0][1])
            label = int(interaction[1])
            if label == 1:
                if user not in user_history_dict:
                    user_history_dict[user] = set()
                user_history_dict[user].add(item)
        return user_history_dict

    def ndcg_at_k_batch(self, hits, k,device='cuda'):
        """
        calculate NDCG@k
        hits: array, element is binary (0 / 1), 2-dim
        """
        if hits.ndim == 1:
            hits = hits.reshape(1, -1)
        hits_k = hits[:, :k]
        dcg = ((2 ** hits_k - 1) / torch.log2(torch.arange(2, k + 2, device=device))).sum(axis=1)
        sorted_hits_k, _ = hits_k.sort(descending=True)
        idcg = ((2 ** sorted_hits_k - 1) / torch.log2(torch.arange(2, k + 2, device=device))).sum(axis=1)
        idcg[idcg == 0] = np.inf
        ndcg = (dcg / idcg)
        return ndcg
