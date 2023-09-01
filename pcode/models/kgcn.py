import random
from collections import defaultdict

import torch
import torch.nn.functional as F

__all__ = ["kgcn", "kgcn_aggregate", "kgcn_kg"]

from torch import nn

from pcode.utils.auto_distributed import recv_list, send_list


class Aggregator(torch.nn.Module):
    '''
    Aggregator class
    Mode in ['sum', 'concat', 'neighbor']
    '''

    def __init__(self, batch_size, dim, aggregator):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim  # 16
        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        self.aggregator = aggregator

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, act):
        batch_size = user_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        if self.aggregator == 'sum':
            output = (self_vectors + neighbors_agg).reshape((-1, self.dim))

        elif self.aggregator == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            output = output.reshape((-1, 2 * self.dim))

        else:
            output = neighbors_agg.reshape((-1, self.dim))

        output = self.weights(output)
        return act(output.reshape((self.batch_size, -1, self.dim)))

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        '''
        This aims to aggregate neighbor vectors
        '''
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.reshape((self.batch_size, 1, 1, self.dim))

        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim=-1)
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)

        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim=-1)

        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_scores_normalized * neighbor_vectors).sum(dim=2)

        return neighbors_aggregated


class KGCN(torch.nn.Module):
    def __init__(self, num_usr, num_ent, num_rel, kg, args, device):
        super(KGCN, self).__init__()
        self.num_usr = num_usr
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size  # 采样邻居个数
        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)  # 相加然后过线性层
        self.init = torch.nn.init.xavier_normal_
        self._gen_adj()  # 对KG中的每一个head,固定采样n_neighbor个邻居节点和关系

        self.usr = torch.zeros(num_usr, args.dim).to(self.device)
        self.ent = torch.zeros(num_ent, args.dim).to(self.device)
        self.rel = torch.zeros(num_rel, args.dim).to(self.device)
        self.init(self.usr)
        self.init(self.ent)
        self.init(self.rel)

        self._init_trained_tracker()

    def _init_trained_tracker(self):
        self.trained = defaultdict(list)
        self.trained['user_embeddings_grad'] = 0
        self.trained['entities_embeddings_grad'] = 0
        self.trained['relations_embeddings_grad'] = 0

    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)

        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)

            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])

    def forward(self, data_batch, entities=None, relations=None):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        (u, v) = data_batch
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        u = u.reshape((-1, 1))
        v = v.reshape((-1, 1))

        self.usr.requires_grad = True
        self.ent.requires_grad = True
        self.rel.requires_grad = True

        # [batch_size, dim]
        user_embeddings = self.usr[u].squeeze(dim=1)
        if entities == None and relations == None:
            entities, relations = self._get_neighbors(v)  # 对每一个user-item的item,取item的n_iter层邻居

        item_embeddings = self._aggregate(user_embeddings, entities, relations)  # 单层加权求和

        scores = (user_embeddings * item_embeddings).sum(dim=1)
        return torch.sigmoid(scores)

    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v]
        relations = []

        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).reshape((self.batch_size, -1)).to(
                self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).reshape((self.batch_size, -1)).to(
                self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent[entity] for entity in entities]
        relation_vectors = [self.rel[relation] for relation in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].reshape((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].reshape((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].reshape((self.batch_size, self.dim))

    def request_neighbors(self, user_id, item_id):
        with torch.no_grad():
            # user_embeddings
            [user_embeddings] = recv_list(0)
            self.user_id = user_id
            self.usr[user_id] = user_embeddings

            # item_embeddings
            self.neighbor_entities, self.neighbor_relations = self.get_neighbor_id(item_id)
            send_list([self.neighbor_entities, self.neighbor_relations], 0)
            entities_embeddings, relations_embeddings = recv_list(0)
            self.ent[self.neighbor_entities] = entities_embeddings
            self.rel[self.neighbor_relations] = relations_embeddings

    def get_neighbor_id(self, item_id):
        neighbor_entities = torch.flatten(torch.LongTensor(self.adj_ent[item_id])).unique()
        neighbor_relations = torch.flatten(torch.LongTensor(self.adj_rel[item_id])).unique()
        return neighbor_entities, neighbor_relations

    def distribute_neighbors(self, selected_client_ids):
        with torch.no_grad():
            self.trained['user_id'] += [torch.LongTensor([id - 1]) for id in selected_client_ids]
            # user_embeddings
            user_embeddings = [self.usr[torch.tensor(id - 1).to(self.device)] for id in selected_client_ids]
            for worker, embeddings in enumerate(user_embeddings, 1):
                send_list([user_embeddings[worker - 1]], worker)

            # item_embeddings

            for worker, embeddings in enumerate(user_embeddings, 1):
                [entities, relations] = recv_list(worker)
                self.trained['entities_id'].append(entities)
                self.trained['relations_id'].append(relations)
                send_list([self.ent[entities], self.rel[relations]], worker)

    def upload_selected(self):
        with torch.no_grad():
            user_embeddings_grad = self.usr.grad[self.user_id]
            entities_embeddings_grad = self.ent.grad[self.neighbor_entities]
            relations_embeddings_grad = self.rel.grad[self.neighbor_relations]
            send_list([user_embeddings_grad, entities_embeddings_grad, relations_embeddings_grad], 0)

    def upload(self):
        with torch.no_grad():
            user_embeddings_grad = self.usr.grad
            entities_embeddings_grad = self.ent.grad
            relations_embeddings_grad = self.rel.grad
            send_list([user_embeddings_grad, entities_embeddings_grad, relations_embeddings_grad], 0)

    def receive_selected(self, selected_client_ids):
        with torch.no_grad():
            for worker, selected_client_id in enumerate(selected_client_ids, 1):
                [user_embeddings_grad, entities_embeddings_grad, relations_embeddings_grad] = recv_list(worker,
                                                                                                        store_device=torch.device(
                                                                                                            'cpu'))
                self.trained['user_embeddings_grad'].append(user_embeddings_grad)
                self.trained['entities_embeddings_grad'].append(entities_embeddings_grad)
                self.trained['relations_embeddings_grad'].append(relations_embeddings_grad)

    def receive(self, selected_client_ids):
        with torch.no_grad():
            for worker, selected_client_id in enumerate(selected_client_ids, 1):
                [user_embeddings_grad, entities_embeddings_grad, relations_embeddings_grad] = recv_list(worker,
                                                                                                        store_device=torch.device(
                                                                                                            'cpu'))
                self.trained['user_embeddings_grad'] += user_embeddings_grad
                self.trained['entities_embeddings_grad'] += entities_embeddings_grad
                self.trained['relations_embeddings_grad'] += relations_embeddings_grad

    def update_selected(self):
        with torch.no_grad():
            device = torch.device('cpu')
            self.trained['user_id'] = torch.concat(self.trained['user_id']).to(device)
            self.trained['entities_id'] = torch.concat(self.trained['entities_id']).to(device)
            self.trained['relations_id'] = torch.concat(self.trained['relations_id']).to(device)
            self.trained['user_embeddings_grad'] = torch.vstack(self.trained['user_embeddings_grad']).to(device)
            self.trained['entities_embeddings_grad'] = torch.vstack(self.trained['entities_embeddings_grad']).to(device)
            self.trained['relations_embeddings_grad'] = torch.vstack(self.trained['relations_embeddings_grad']).to(
                device)
            self.trained_usr = torch.zeros_like(self.usr, device='cpu')
            self.trained_ent = torch.zeros_like(self.ent, device='cpu')
            self.trained_rel = torch.zeros_like(self.rel, device='cpu')

            self.trained_usr.scatter_reduce_(0, self.trained['user_id'].expand(self.trained_usr.shape[1],
                                                                               len(self.trained['user_id'])).T,
                                             self.trained['user_embeddings_grad'], "sum", include_self=False)
            self.trained_ent.scatter_reduce_(0, self.trained['entities_id'].expand(self.trained_ent.shape[1],
                                                                                   len(self.trained['entities_id'])).T,
                                             self.trained['entities_embeddings_grad'], "sum", include_self=False)
            self.trained_rel.scatter_reduce_(0, self.trained['relations_id'].expand(self.trained_rel.shape[1],
                                                                                    len(self.trained[
                                                                                            'relations_id'])).T,
                                             self.trained['relations_embeddings_grad'], "sum", include_self=False)
            batch_size = len(self.trained['user_id'])
            self.trained_usr = self.trained_usr.to(self.device) / batch_size
            self.trained_ent = self.trained_ent.to(self.device) / batch_size
            self.trained_rel = self.trained_rel.to(self.device) / batch_size
            self.usr.grad = torch.zeros_like(self.trained_usr)
            self.ent.grad = torch.zeros_like(self.trained_ent)
            self.rel.grad = torch.zeros_like(self.trained_rel)
            self.usr.grad = torch.where(self.trained_usr == self.usr.grad, self.usr.grad, self.trained_usr)
            self.ent.grad = torch.where(self.trained_ent == self.ent.grad, self.ent.grad, self.trained_ent)
            self.rel.grad = torch.where(self.trained_rel == self.rel.grad, self.rel.grad, self.trained_rel)

            self._init_trained_tracker()

    def update(self):
        with torch.no_grad():
            device = torch.device('cpu')
            self.trained['user_id'] = torch.concat(self.trained['user_id']).to(device)
            self.trained['entities_id'] = torch.concat(self.trained['entities_id']).to(device)
            self.trained['relations_id'] = torch.concat(self.trained['relations_id']).to(device)
            batch_size = len(self.trained['user_id'])
            self.usr.grad = self.trained['user_embeddings_grad'].to(self.device) / batch_size
            self.ent.grad = self.trained['entities_embeddings_grad'].to(self.device) / batch_size
            self.rel.grad = self.trained['relations_embeddings_grad'].to(self.device) / batch_size
            self._init_trained_tracker()


class KGCN_E(torch.nn.Module):
    def __init__(self, num_usr, num_ent, num_rel, kg, args, device):
        super(KGCN_E, self).__init__()
        self.num_usr = num_usr
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size  # 采样邻居个数
        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)  # 相加然后过线性层
        self._gen_adj()  # 对KG中的每一个head,固定采样n_neighbor个邻居节点和关系
        self.init = torch.nn.init.xavier_normal_

        self.usr = nn.Embedding(num_usr, args.dim)
        self.ent = nn.Embedding(num_ent, args.dim)
        self.rel = nn.Embedding(num_rel, args.dim)
        self.init(self.usr.weight)
        self.init(self.ent.weight)
        self.init(self.rel.weight)

        self.trained = defaultdict(list)

    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)

        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)

            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])

    def forward(self, data_batch, entities=None, relations=None):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        (u, v) = data_batch
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        u = u.reshape((-1, 1))
        v = v.reshape((-1, 1))

        self.usr.requires_grad = True
        self.ent.requires_grad = True
        self.rel.requires_grad = True

        # [batch_size, dim]
        user_embeddings = self.usr(u).squeeze(dim=1)
        if entities == None and relations == None:
            entities, relations = self._get_neighbors(v)  # 对每一个user-item的item,取item的n_iter层邻居

        item_embeddings = self._aggregate(user_embeddings, entities, relations)  # 单层加权求和

        scores = (user_embeddings * item_embeddings).sum(dim=1)

        return torch.sigmoid(scores)

    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v]
        relations = []

        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).reshape((self.batch_size, -1)).to(
                self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).reshape((self.batch_size, -1)).to(
                self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].reshape((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].reshape((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].reshape((self.batch_size, self.dim))

    def request_neighbors(self, user_id, item_id):
        with torch.no_grad():
            # user_embeddings
            [user_embeddings] = recv_list(0)
            self.user_id = user_id
            self.usr.weight[user_id] = user_embeddings

            # item_embeddings
            self.neighbor_entities, self.neighbor_relations = self.get_neighbor_id(item_id)
            send_list([self.neighbor_entities, self.neighbor_relations], 0)
            entities_embeddings, relations_embeddings = recv_list(0)
            self.ent.weight[self.neighbor_entities] = entities_embeddings
            self.rel.weight[self.neighbor_relations] = relations_embeddings

    def get_neighbor_id(self, item_id):
        neighbor_entities = torch.flatten(torch.LongTensor(self.adj_ent[item_id]))
        neighbor_entities = torch.concat([neighbor_entities, item_id]).unique()
        neighbor_relations = torch.flatten(torch.LongTensor(self.adj_rel[item_id])).unique()
        return neighbor_entities, neighbor_relations

    def distribute_neighbors(self, selected_client_ids):
        with torch.no_grad():
            self.trained['user_id'] += [torch.LongTensor([id - 1]) for id in selected_client_ids]
            # user_embeddings
            user_embeddings = [self.usr(torch.tensor(id - 1)) for id in selected_client_ids]
            for worker, embeddings in enumerate(user_embeddings, 1):
                send_list([user_embeddings[worker - 1]], worker)

            # item_embeddings

            for worker, embeddings in enumerate(user_embeddings, 1):
                [entities, relations] = recv_list(worker)
                self.trained['entities_id'].append(entities)
                self.trained['relations_id'].append(relations)
                send_list([self.ent(entities), self.rel(relations)], worker)

    def upload(self):
        with torch.no_grad():
            user_embeddings_grad = self.usr.weight.grad[self.user_id]
            entities_embeddings_grad = self.ent.weight.grad[self.neighbor_entities]
            relations_embeddings_grad = self.rel.weight.grad[self.neighbor_relations]
            send_list([user_embeddings_grad, entities_embeddings_grad, relations_embeddings_grad], 0)

    def receive(self, selected_client_ids):
        with torch.no_grad():
            for worker, selected_client_id in enumerate(selected_client_ids, 1):
                [user_embeddings_grad, entities_embeddings_grad, relations_embeddings_grad] = recv_list(worker,
                                                                                                        store_device=torch.device(
                                                                                                            'cpu'))
                self.trained['user_embeddings_grad'].append(user_embeddings_grad)
                self.trained['entities_embeddings_grad'].append(entities_embeddings_grad)
                self.trained['relations_embeddings_grad'].append(relations_embeddings_grad)

    def update(self):
        with torch.no_grad():
            device = torch.device('cpu')
            self.trained['user_id'] = torch.concat(self.trained['user_id']).to(device)
            self.trained['entities_id'] = torch.concat(self.trained['entities_id']).to(device)
            self.trained['relations_id'] = torch.concat(self.trained['relations_id']).to(device)
            self.trained['user_embeddings_grad'] = torch.vstack(self.trained['user_embeddings_grad']).to(device)
            self.trained['entities_embeddings_grad'] = torch.vstack(self.trained['entities_embeddings_grad']).to(device)
            self.trained['relations_embeddings_grad'] = torch.vstack(self.trained['relations_embeddings_grad']).to(
                device)
            self.trained_usr = torch.zeros_like(self.usr.weight, device='cpu')
            self.trained_ent = torch.zeros_like(self.ent.weight, device='cpu')
            self.trained_rel = torch.zeros_like(self.rel.weight, device='cpu')

            self.trained_usr.scatter_reduce_(0, self.trained['user_id'].expand(self.trained_usr.shape[1],
                                                                               len(self.trained['user_id'])).T,
                                             self.trained['user_embeddings_grad'], "sum", include_self=False)
            self.trained_ent.scatter_reduce_(0, self.trained['entities_id'].expand(self.trained_ent.shape[1],
                                                                                   len(self.trained['entities_id'])).T,
                                             self.trained['entities_embeddings_grad'], "sum", include_self=False)
            self.trained_rel.scatter_reduce_(0, self.trained['relations_id'].expand(self.trained_rel.shape[1],
                                                                                    len(self.trained[
                                                                                            'relations_id'])).T,
                                             self.trained['relations_embeddings_grad'], "sum", include_self=False)
            batch_size = len(self.trained['user_id'])
            self.trained_usr = (self.trained_usr / batch_size).to(device)
            self.trained_ent = (self.trained_ent / batch_size).to(device)
            self.trained_rel = (self.trained_rel / batch_size).to(device)
            self.usr.weight.grad = torch.zeros_like(self.trained_usr)
            self.ent.weight.grad = torch.zeros_like(self.trained_ent)
            self.rel.weight.grad = torch.zeros_like(self.trained_rel)
            self.usr.weight.grad = torch.where(self.trained_usr == self.usr.weight.grad, self.usr.weight.grad,
                                               self.trained_usr)
            self.ent.weight.grad = torch.where(self.trained_ent == self.ent.weight.grad, self.ent.weight.grad,
                                               self.trained_ent)
            self.rel.weight.grad = torch.where(self.trained_rel == self.rel.weight.grad, self.rel.weight.grad,
                                               self.trained_rel)

            self.trained = defaultdict(list)

    def parameters(self, embeddings=False):
        for name, param in self.named_parameters():
            if embeddings or 'aggregator' in name:
                yield param


class KGCN_kg(torch.nn.Module):
    def __init__(self, num_usr, num_ent, num_rel, kg, args, device):
        super(KGCN_kg, self).__init__()
        self.num_usr = num_usr
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size  # 采样邻居个数
        self.kg = kg
        self.device = device
        self.aggregator = KGCN_aggregator(args.batch_size, args.dim, args.neighbor_sample_size, args.aggregator,
                                          args.n_iter)
        self._gen_adj()  # 对KG中的每一个head,固定采样n_neighbor个邻居节点和关系
        self.init = torch.nn.init.xavier_normal_
        self.neighbor_embeddings_dict = {}
        self.id_map = {}

        self.usr = nn.Embedding(num_usr, args.dim)
        self.ent = nn.Embedding(num_ent, args.dim)
        self.rel = nn.Embedding(num_rel, args.dim)
        self.init(self.usr.weight)
        self.init(self.ent.weight)
        self.init(self.rel.weight)

        self.trained = defaultdict(list)

    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        self.register_buffer('adj_ent', torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long))
        self.register_buffer('adj_rel', torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long))

        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)

            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])

    def _get_embeddings(self, user_id, dataset,batch_size, request_by_client=False):
        with torch.no_grad():
            if request_by_client:
                pass
            else:
                if not hasattr(self, "dataset_dict"):
                    self.dataset_dict = {
                        idx: [torch.tensor(df["itemID"].values), torch.tensor(df["label"].values, dtype=torch.float)]
                        for idx, df in dataset.df.groupby(["userID"])}
                item_ids = self.dataset_dict[user_id][0]
                target = self.dataset_dict[user_id][1]
            if batch_size == None or batch_size>item_ids.size(0):
                self.batch_size = item_ids.size(0)
            else:
                idx = torch.randperm(item_ids.size(0))
                item_ids = item_ids[idx[:batch_size]]
                target = target[idx[:batch_size]]
                self.batch_size = batch_size
            # change to [batch_size, 1]
            item_ids = item_ids.clone().reshape((-1, 1))
            # [batch_size, dim]

            user_embeddings = self.usr(torch.tensor([user_id]))

            entities, relations = self._get_neighbors(item_ids)  # 对每一个user-item的item,取item的n_iter层邻居

            entity_set = torch.concat([entity.unique() for entity in entities]).unique()
            entity_map = {int(entity): i for i, entity in enumerate(entity_set)}
            entities = [entity.map_(entity, lambda x, y: entity_map[x]) for entity in entities]
            entity_embeddings = torch.vstack([self.ent(entity) for entity in entity_set])

            relation_set = torch.concat([relation.unique() for relation in relations]).unique()
            relation_map = {int(relation): i for i, relation in enumerate(relation_set)}
            relations = [relation.map_(relation, lambda x, y: relation_map[x]) for relation in relations]
            relation_embeddings = torch.vstack([self.rel(relation) for relation in relation_set])

            self.id_map[user_id] = [list(entity_map.keys()), list(relation_map.keys())]
            return [user_embeddings, entities, entity_embeddings,
                    relations, relation_embeddings, target]

    def _all_embeddings(self, user_id, dataset, request_by_client=False):
        with torch.no_grad():
            if request_by_client:
                pass
            else:
                if not hasattr(self, "dataset_dict"):
                    self.dataset_dict = {idx: [torch.tensor(df["itemID"].values),
                                               torch.tensor(df["label"].values, dtype=torch.float)]
                                         for idx, df in dataset.df.groupby(["userID"])}
                item_ids = self.dataset_dict[user_id][0]
                target = self.dataset_dict[user_id][1]
            self.batch_size = item_ids.size(0)
            # change to [batch_size, 1]
            item_ids = item_ids.reshape((-1, 1))
            # [batch_size, dim]

            user_embeddings = self.usr(torch.tensor([user_id]))
            entities, relations = self._get_neighbors(item_ids)
            return [self.usr.weight.clone(), entities, self.ent.weight.clone(),
                    relations, self.rel.weight.clone(), target]

    def forward(self, data_batch):
        if len(data_batch) == 1:
            pass
        else:
            user_ids, item_ids = data_batch
            # change to [batch_size, 1]
            self.batch_size = item_ids.size(0)
            item_ids = item_ids.reshape((-1, 1))
            # [batch_size, dim]
            entities, relations = self._get_neighbors(item_ids)
            return self.aggregator(user_ids, self.usr.weight, entities, self.ent.weight, relations, self.rel.weight)

    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v]
        relations = []

        for h in range(self.n_iter):
            neighbor_entities = self.adj_ent[entities[h]].reshape((self.batch_size, -1))
            neighbor_relations = self.adj_rel[entities[h]].reshape((self.batch_size, -1))
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    def recode_grad(self, flatten_local_models):
        with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                param.grad=torch.zeros_like(param)
            for user_id, grad in flatten_local_models.items():
                usr_grad, ent_grad, rel_grad=grad['embeddings_grad']
                model_grad= grad['model_grad']
                self.usr.weight.grad[user_id] += usr_grad[0]
                self.ent.weight.grad[self.id_map[user_id][0]] += ent_grad
                self.rel.weight.grad[self.id_map[user_id][1]] += rel_grad
                for i, param in enumerate(self.aggregator.parameters()):
                    param.grad += model_grad[i]
            for param in self.parameters():
                param.grad/=len(flatten_local_models)
    def recode_grad_by_trainning_num(self, flatten_local_models):
        with torch.no_grad():
            for i, param in enumerate(self.parameters()):
                param.grad = torch.zeros_like(param)
            totle_interactions = 0
            usr_count = torch.zeros(self.num_usr)
            ent_count = torch.zeros(self.num_ent)
            rel_count = torch.zeros(self.num_rel)
            for user_id, grad in flatten_local_models.items():
                num= len(self.dataset_dict[user_id][0])
                totle_interactions+=num

                model_grad= grad['model_grad']
                usr_grad, ent_grad, rel_grad = grad['embeddings_grad']
                self.usr.weight.grad[user_id] += usr_grad[0]* num
                self.ent.weight.grad[self.id_map[user_id][0]] += ent_grad* num
                self.rel.weight.grad[self.id_map[user_id][1]] += rel_grad* num
                usr_count[user_id]+=num
                ent_count[self.id_map[user_id][0]]+=num
                rel_count[self.id_map[user_id][1]]+=num
                for i,param in enumerate(self.aggregator.parameters()):
                    param.grad+= model_grad[i]*num
            usr_count[usr_count == 0] = 1  # 没有交互的给1，防止零除
            ent_count[ent_count == 0] = 1  # 没有交互的给1，防止零除
            rel_count[rel_count == 0] = 1  # 没有交互的给1，防止零除
            self.usr.weight.grad/= usr_count.reshape([-1, 1])
            self.ent.weight.grad/= ent_count.reshape([-1, 1])
            self.rel.weight.grad/= rel_count.reshape([-1, 1])
            for i, param in enumerate(self.aggregator.parameters()):
                param.grad /= totle_interactions


    def recode_all(self, grads):
        for (usr_grad, ent_grad, rel_grad) in grads:
            if usr_grad != None:
                self.usr_grad += usr_grad
                self.ent_grad += ent_grad
                self.rel_grad += rel_grad


class KGCN_aggregator(torch.nn.Module):
    def __init__(self, batch_size, dim, n_neighbor, aggregator, n_iter):
        super(KGCN_aggregator, self).__init__()
        self.batch_size = None
        self.dim = dim
        self.n_neighbor = n_neighbor
        self.n_iter = n_iter
        self.aggregator = Aggregator(batch_size, dim, aggregator)

    def forward(self, usr_id, usr_embed, ent_id, ent_embed, rel_id, rel_embed):
        ent_embed.requires_grad = True
        rel_embed.requires_grad = True
        usr_embed.requires_grad = True
        self.batch_size = len(ent_id[0])
        if usr_id is None:
            usr_id = [0] * self.batch_size
        user_embeddings = usr_embed[usr_id]
        entities_embeddings = [ent_embed[entity] for entity in ent_id]
        relations_embeddings = [rel_embed[relation] for relation in rel_id]
        item_embeddings = self._aggregate(user_embeddings, entities_embeddings, relations_embeddings)  # 单层加权求和

        scores = (user_embeddings * item_embeddings).sum(dim=1)
        return torch.sigmoid(scores)

    def _aggregate(self, user_embeddings, entity_embeddings, relation_embeddings):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_embeddings[hop],
                    neighbor_vectors=entity_embeddings[hop + 1].reshape(
                        (self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_embeddings[hop].reshape(
                        (self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_embeddings = entity_vectors_next_iter

        return entity_embeddings[0].reshape((self.batch_size, self.dim))


def kgcn(conf):
    if hasattr(conf, "kg"):
        kg, num_user, num_entity, num_relation = conf.kg
    else:
        kg, num_user, num_entity, num_relation = None, 1872, 9366, 60
    device = conf.device
    conf.n_iter = 1
    conf.batch_size = 32
    conf.dim = 16
    conf.neighbor_sample_size = 8
    conf.aggregator = "sum"

    return KGCN_E(num_user, num_entity, num_relation, kg, conf, device)


def kgcn_aggregate(conf):
    return KGCN_aggregator(batch_size=conf.batch_size, dim=conf.dim, n_neighbor=conf.neighbor_sample_size, aggregator=conf.aggregator, n_iter=conf.n_iter)


def kgcn_kg(conf):
    if hasattr(conf, "kg"):
        kg, num_user, num_entity, num_relation = conf.kg
    else:
        kg, num_user, num_entity, num_relation = None, 1872, 9366, 60
    device = conf.device

    return KGCN_kg(num_user, num_entity, num_relation, kg, conf, device)
