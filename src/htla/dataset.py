import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

BIG5_LABELS  = ["agreeableness_label", "openness_label",
                 "conscientiousness_label", "extraversion_label",
                 "neuroticism_label"]
BERT_MODEL   = "bert-base-uncased"
MAX_TOKENS   = 128
MAX_COMMENTS = 50

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)


class PANDORADataset(Dataset):
    def __init__(self, author_list, profiles_df, comments_df,
                 max_comments=MAX_COMMENTS, max_tokens=MAX_TOKENS):
        self.max_comments = max_comments
        self.max_tokens   = max_tokens
        self.profiles     = profiles_df[
            profiles_df["author"].isin(author_list)
        ].reset_index(drop=True)

        print("Building comment index...")
        self.comments_by_author = (
            comments_df[comments_df["author"].isin(author_list)]
            .groupby("author")["body"]
            .apply(list)
            .to_dict()
        )
        print(f"  Done. {len(self.comments_by_author)} users indexed.")

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        row    = self.profiles.iloc[idx]
        author = row["author"]
        labels = torch.tensor(
            row[BIG5_LABELS].values.astype(float), dtype=torch.float
        )
        user_comments       = self.comments_by_author.get(author, [""])
        input_ids_list      = []
        attention_mask_list = []

        for comment in user_comments[:self.max_comments]:
            encoded = tokenizer(
                str(comment),
                max_length=self.max_tokens,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids_list.append(encoded["input_ids"].squeeze(0))
            attention_mask_list.append(encoded["attention_mask"].squeeze(0))

        num_comments = len(input_ids_list)
        while len(input_ids_list) < self.max_comments:
            input_ids_list.append(torch.zeros(self.max_tokens, dtype=torch.long))
            attention_mask_list.append(torch.zeros(self.max_tokens, dtype=torch.long))

        comment_mask               = torch.zeros(self.max_comments, dtype=torch.bool)
        comment_mask[:num_comments] = True

        return {
            "input_ids":       torch.stack(input_ids_list),
            "attention_masks": torch.stack(attention_mask_list),
            "comment_mask":    comment_mask,
            "labels":          labels
        }