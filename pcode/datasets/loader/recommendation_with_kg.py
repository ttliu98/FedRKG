import torch


class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.idx = self.df.index
        self.index = defaultdict(list)
        self.user_num = self.df.userID.max() + 1
        for user_id in range(self.user_num):
            self.index[user_id] = self.df[self.df.userID == user_id].index

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
