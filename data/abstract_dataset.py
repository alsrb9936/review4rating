import random
from collections import defaultdict
from torch.utils.data import Dataset


class RecDataset(Dataset):
    def __init__(self, df, configs, split="train"):
        self.df = df.reset_index(drop=True)
        self.configs = configs
        self.split = split

        # get_dataloader()에서 이미 전체 split 기준으로 넣어줌
        self.num_users = int(configs.get("num_users", self.df["user_id"].max() + 1))
        self.num_items = int(configs.get("num_items", self.df["item_id"].max() + 1))

        # train에서 negative sampling용
        self.user_pos_items = defaultdict(set)
        if split == "train":
            for row in self.df.itertuples(index=False):
                self.user_pos_items[int(row.user_id)].add(int(row.item_id))

        # evaluator 호환용
        self.user_history_lists = {}
        self.valid_user_pos_lists = {}
        self.test_user_pos_lists = {}
        self.valid_users = []
        self.test_users = []
        self.eval_users = []

    def __len__(self):
        if self.split == "train":
            return len(self.df)
        return len(self.eval_users)

    def __getitem__(self, idx):
        raise NotImplementedError

    def _setup_evaluation(self, train_df, valid_df, test_df):
        pass

    def _sample_negative(self, user_id):
        pos_items = self.user_pos_items[user_id]
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in pos_items:
                return neg_item