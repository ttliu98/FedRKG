import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils import data


class RecommendationDS(data.Dataset):
    '''
    Data Loader class which makes dataset for training / knowledge graph dictionary
    '''

    def __init__(self, data, train=True):
        self.preprocessed_data_path = Path(f'./data/{data}/preprocessed_data')
        if not self.preprocessed_data_path.exists():  # music

            self.cfg = {
                'movie': {
                    'item2id_path': 'data/movie/item_index2entity_id.txt',
                    'kg_path': 'data/movie/kg.txt',
                    'rating_path': 'data/movie/ratings.csv',
                    'rating_sep': ',',
                    'threshold': 4.0
                },
                'music': {
                    'item2id_path': 'data/music/item_index2entity_id.txt',
                    'kg_path': 'data/music/kg.txt',
                    'rating_path': 'data/music/user_artists.dat',
                    'rating_sep': '\t',
                    'threshold': 0.0
                },
                'book': {
                    'item2id_path': 'data/book/item_index2entity_id_rehashed.txt',
                    'kg_path': 'data/book/kg_rehashed.txt',
                    'rating_path': 'data/book/BX-Book-Ratings.csv',
                    'rating_sep': ';',
                    'threshold': 0.0
                }
            }
            self.data = data
            self.test_ratio = 0.2

            df_item2id = pd.read_csv(self.cfg[data]['item2id_path'], sep='\t', header=None, names=['item', 'id'])
            df_kg = pd.read_csv(self.cfg[data]['kg_path'], sep='\t', header=None, names=['head', 'relation', 'tail'])
            df_rating = pd.read_csv(self.cfg[data]['rating_path'], sep=self.cfg[data]['rating_sep'],
                                    names=['userID', 'itemID', 'rating'], skiprows=1)

            # df_rating['itemID'] and df_item2id['item'] both represents old entity ID
            df_rating = df_rating[df_rating['itemID'].isin(df_item2id['item'])]  # 只取item2id里存在的item
            df_rating= df_rating.groupby('userID').filter(lambda x: len(x) > 10)
            df_rating.reset_index(inplace=True, drop=True)


            self.df_item2id = df_item2id  # item和id的对应关系
            self.df_kg = df_kg  # 知识图谱
            self.df_rating = df_rating  # 只包含对应关系item的购买记录 pd(user,item,rating)


            self.user_encoder = LabelEncoder()
            self.entity_encoder = LabelEncoder()
            self.relation_encoder = LabelEncoder()

            self._encoding()

            kg = self._construct_kg()  # {head: (relation,tails)} 无向图，正反同关系
            df_dataset = self._build_dataset()
            train_set, test_set, _, _ = train_test_split(df_dataset, df_dataset['label'], test_size=self.test_ratio,
                                                         shuffle=False, random_state=999)
            num_user, num_entity, num_relation = self.get_num()

            train_userIDs = set(train_set['userID'])
            test_userIDs = set(test_set['userID'])
            userIDs_to_move = test_userIDs - train_userIDs
            rows_to_move = test_set[test_set['userID'].isin(userIDs_to_move)]
            train_set = train_set.append(rows_to_move)
            test_set = test_set[~test_set['userID'].isin(userIDs_to_move)]

            torch.save(
                {'train_set': train_set, 'test_set': test_set, 'kg': kg, 'num_user': num_user, 'num_entity': num_entity,
                 'num_relation': num_relation},
                self.preprocessed_data_path)
        else:
            preprocessed_data = torch.load(self.preprocessed_data_path)
            train_set, test_set = preprocessed_data['train_set'], preprocessed_data['test_set']


        self.df = train_set if train else test_set
        self.idx = self.df.index
        self.index = defaultdict(list)
        self.user_num = self.df.userID.max() + 1
        for user_id in range(self.user_num):
            self.index[user_id] = self.df[self.df.userID == user_id].index

    def _encoding(self):
        '''
        Fit each label encoder and encode knowledge graph
        '''
        self.user_encoder.fit(self.df_rating['userID'])
        # df_item2id['id'] and df_kg[['head', 'tail']] represents new entity ID
        self.entity_encoder.fit(
            pd.concat([self.df_kg['head'], self.df_kg['tail']]))  # id是item的紧凑表示, 必定在entitiy里

        self.relation_encoder.fit(self.df_kg['relation'])

        # encode df_kg
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['head'])
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['tail'])
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['relation'])

    def _build_dataset(self):
        '''
        Build dataset for training (rating data)
        It contains negative sampling process
        '''
        print('Build dataset dataframe ...', end=' ')
        # df_rating update
        df_dataset = pd.DataFrame()
        df_dataset['userID'] = self.user_encoder.transform(self.df_rating['userID'])

        # update to new id
        item2id_dict = dict(zip(self.df_item2id['item'], self.df_item2id['id']))
        self.df_rating['itemID'] = self.df_rating['itemID'].apply(lambda x: item2id_dict[x])  # item 映射为 entity id
        df_dataset['itemID'] = self.entity_encoder.transform(self.df_rating['itemID'])  # 紧凑表示

        df_dataset['label'] = self.df_rating['rating'].apply(
            lambda x: 0 if x < self.cfg[self.data]['threshold'] else 1)  # label二值化[0,1]

        # negative sampling
        df_dataset = df_dataset[df_dataset['label'] == 1]  # 只取rating大于阈值的正样本
        # df_dataset requires columns to have new entity ID
        full_item_set = set(range(len(self.entity_encoder.classes_)))
        user_list = []
        item_list = []
        label_list = []
        for user, group in df_dataset.groupby(['userID']):
            item_set = set(group['itemID'])
            negative_set = full_item_set - item_set
            negative_sampled = random.sample(negative_set, len(item_set))  # 采样和正样本一样多的负样本
            user_list.extend([user] * len(negative_sampled))
            item_list.extend(negative_sampled)
            label_list.extend([0] * len(negative_sampled))
        negative = pd.DataFrame({'userID': user_list, 'itemID': item_list, 'label': label_list})  # 负样本label为0
        df_dataset = pd.concat([df_dataset, negative])

        df_dataset = df_dataset.sample(frac=1, replace=False, random_state=999)  # 不放回抽样1,相当于打乱
        df_dataset.reset_index(inplace=True, drop=True)
        print('Done')
        return df_dataset

    def _construct_kg(self):
        '''
        Construct knowledge graph
        Knowledge graph is dictionary form
        'head': [(relation, tail), ...]
        '''
        print('Construct knowledge graph ...', end=' ')
        kg = dict()
        for i in range(len(self.df_kg)):
            head = self.df_kg.iloc[i]['head']
            relation = self.df_kg.iloc[i]['relation']
            tail = self.df_kg.iloc[i]['tail']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
            if tail in kg:
                kg[tail].append((relation, head))
            else:
                kg[tail] = [(relation, head)]
        print('Done')
        return kg

    def get_kg(self):
        preprocessed_data = torch.load(self.preprocessed_data_path)
        return preprocessed_data['kg'], preprocessed_data['num_user'], preprocessed_data['num_entity'], \
        preprocessed_data['num_relation']

    def get_encoders(self):
        return (self.user_encoder, self.entity_encoder, self.relation_encoder)

    def get_num(self):
        return (len(self.user_encoder.classes_), len(self.entity_encoder.classes_), len(self.relation_encoder.classes_))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        user_id = np.array(self.df.loc[self.idx[idx]]['userID'])
        item_id = np.array(self.df.loc[self.idx[idx]]['itemID'])
        label = np.array(self.df.loc[self.idx[idx]]['label'], dtype=np.float32)
        return (user_id, item_id), label

    def set_user(self, user_id):
        if isinstance(user_id, int):
            if user_id == -1:
                self.idx = self.df.index
            else:
                self.idx = self.index[user_id]
        else:
            self.idx = self.index[user_id[0]]
            for id in user_id[1:]:
                self.idx = self.idx.append(self.index[id])

        return self

    def get(self, attr):
        return self.df.loc[self.idx][attr].values
