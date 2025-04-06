import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import datasets
from collections import defaultdict
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm


# 1. 数据集处理
class Conll2003NERDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512, label_all_tokens=True):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens
        self.label_map = {label: i for i, label in enumerate(self.dataset.features["ner_tags"].feature.names)}
        self.ignore_index = -100

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        tokens = item["tokens"]
        labels = [self.label_map[label] for label in item["ner_tags"]]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        word_ids = encoding.word_ids()
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(self.ignore_index)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(labels[word_idx] if self.label_all_tokens else self.ignore_index)
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "original_tokens": tokens,  # 保留原始token用于评估
            "word_ids": word_ids  # 保留word_ids用于评估
        }


# 2. 模型架构
class NERModel(nn.Module):
    def __init__(self, num_labels, model_name="intfloat/multilingual-e5-small", margin=1.0):
        super().__init__()
        self.encoder = AutoTokenizer.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.margin = margin
        self.num_labels = num_labels
        self.ignore_index = -100

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss = self.compute_loss(logits, sequence_output, labels, attention_mask)
            return logits, loss
        return logits

    def compute_loss(self, logits, embeddings, labels, attention_mask):
        ce_loss = F.cross_entropy(
            logits.view(-1, self.num_labels),
            labels.view(-1),
            ignore_index=self.ignore_index
        )
        quadruplet_loss = self.quadruplet_loss(embeddings, labels, attention_mask)
        context_loss = self.context_loss(embeddings, labels)
        return ce_loss + 0.5 * quadruplet_loss + 0.1 * context_loss  # 加权求和

    def quadruplet_loss(self, embeddings, labels, attention_mask):
        batch_size, seq_len, _ = embeddings.shape
        loss = 0.0
        valid_count = 0

        # 创建实体索引映射
        entity_indices = defaultdict(list)
        for i in range(batch_size):
            for j in range(seq_len):
                if attention_mask[i, j] and labels[i, j] > 0:  # 忽略padding和非实体
                    entity_indices[labels[i, j].item()].append((i, j))

        # 采样四元组
        for entity_type, indices in entity_indices.items():
            if len(indices) < 2:
                continue

            # 随机选择锚点和正样本
            anchor_idx, positive_idx = np.random.choice(len(indices), 2, replace=False)
            anchor = embeddings[indices[anchor_idx]]
            positive = embeddings[indices[positive_idx]]

            # 选择负样本(不同实体类型)
            other_types = [et for et in entity_indices.keys() if et != entity_type]
            if not other_types:
                continue

            neg_type = np.random.choice(other_types)
            neg_idx = np.random.randint(len(entity_indices[neg_type]))
            negative = embeddings[entity_indices[neg_type][neg_idx]]

            # 选择远负样本(非实体)
            non_entity_indices = [
                (i, j) for i in range(batch_size)
                for j in range(seq_len)
                if attention_mask[i, j] and labels[i, j] == 0
            ]
            if not non_entity_indices:
                continue

            distant_neg_idx = np.random.randint(len(non_entity_indices))
            distant_negative = embeddings[non_entity_indices[distant_neg_idx]]

            # 计算四元组损失
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)
            distant_neg_dist = F.pairwise_distance(anchor, distant_negative)

            loss += F.relu(pos_dist - neg_dist + self.margin) + \
                    F.relu(pos_dist - distant_neg_dist + 2 * self.margin)
            valid_count += 1

        return loss / valid_count if valid_count > 0 else torch.tensor(0.0)

    def context_loss(self, embeddings, labels):
        batch_size, seq_len, _ = embeddings.shape
        loss = 0.0
        valid_pairs = 0

        for i in range(batch_size):
            for j in range(seq_len - 1):
                if labels[i, j] != self.ignore_index and labels[i, j] == labels[i, j + 1] and labels[i, j] > 0:
                    loss += F.mse_loss(embeddings[i, j], embeddings[i, j + 1])
                    valid_pairs += 1

        return loss / valid_pairs if valid_pairs > 0 else torch.tensor(0.0)


# 3. 评估指标计算
class NEREvaluator:
    def __init__(self, label_map):
        self.label_map = label_map
        self.id2label = {v: k for k, v in label_map.items()}

    def align_predictions(self, predictions, label_ids, word_ids, original_tokens):
        """将子词预测对齐到原始单词"""
        preds = np.argmax(predictions, axis=2)
        batch_preds, batch_labels = [], []

        for i in range(len(preds)):
            word_preds, word_labels = [], []
            current_word = None

            for j in range(len(preds[i])):
                word_idx = word_ids[i][j]
                if word_idx is None:
                    continue

                if word_idx != current_word:
                    current_word = word_idx
                    word_preds.append(self.id2label[preds[i][j]])
                    word_labels.append(self.id2label[label_ids[i][j]] if label_ids[i][j] != self.ignore_index else "O")

            batch_preds.append(word_preds)
            batch_labels.append(word_labels)

        return batch_preds, batch_labels

    def compute_metrics(self, predictions, labels, word_ids, original_tokens):
        preds_list, labels_list = self.align_predictions(predictions, labels, word_ids, original_tokens)

        return {
            "precision": precision_score(labels_list, preds_list),
            "recall": recall_score(labels_list, preds_list),
            "f1": f1_score(labels_list, preds_list),
            "report": classification_report(labels_list, preds_list)
        }


# 4. 训练和评估流程
def train_and_evaluate():
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
    dataset = datasets.load_dataset("conll2003", trust_remote_code=True)

    # 准备数据
    train_dataset = Conll2003NERDataset(dataset["train"], tokenizer)
    val_dataset = Conll2003NERDataset(dataset["validation"], tokenizer)
    test_dataset = Conll2003NERDataset(dataset["test"], tokenizer)

    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
            "original_tokens": [x["original_tokens"] for x in batch],
            "word_ids": [x["word_ids"] for x in batch]
        }

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

    # 初始化模型和评估器
    model = NERModel(num_labels=len(train_dataset.label_map)).to(device)
    evaluator = NEREvaluator(train_dataset.label_map)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 训练循环
    best_f1 = 0
    for epoch in range(5):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()
            _, loss = model(**batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} Train Loss: {epoch_loss / len(train_loader):.4f}")

        # 验证
        val_metrics = evaluate(model, val_loader, evaluator, device)
        print(f"Validation Metrics:\n{val_metrics['report']}")

        # 保存最佳模型
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), "best_model.pt")

    # 最终测试
    model.load_state_dict(torch.load("best_model.pt"))
    test_metrics = evaluate(model, test_loader, evaluator, device)
    print(f"\nFinal Test Metrics:\n{test_metrics['report']}")


def evaluate(model, dataloader, evaluator, device):
    model.eval()
    all_preds, all_labels, all_word_ids, all_tokens = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            logits = model(batch_inputs["input_ids"], batch_inputs["attention_mask"])
            all_preds.append(logits.detach().cpu().numpy())
            all_labels.append(batch_inputs["labels"].detach().cpu().numpy())
            all_word_ids.extend(batch_inputs["word_ids"])
            all_tokens.extend(batch_inputs["original_tokens"])

    # 合并所有batch结果
    predictions = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return evaluator.compute_metrics(predictions, labels, all_word_ids, all_tokens)


if __name__ == "__main__":
    train_and_evaluate()
