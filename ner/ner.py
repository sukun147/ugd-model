import json

import numpy as np
import torch
from seqeval.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline, AutoConfig


# ==================== 数据加载部分 ====================
class NERDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, max_length=128, split='train'):
        self.full_data = self.load_data(json_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 划分训练集和测试集（8:2比例）
        train_size = int(0.8 * len(self.full_data))
        test_size = len(self.full_data) - train_size
        self.train_data, self.test_data = random_split(
            self.full_data,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.data = self.train_data if split == 'train' else self.test_data
        self.label2id = self._create_label_mapping(self.full_data)
        self.id2label = {v: k for k, v in self.label2id.items()}

    def _create_label_mapping(self, data):
        labels = set()
        for item in data:
            for ann in item["annotations"]:
                labels.add(ann["label"])
        label2id = {"O": 0}
        for label in sorted(labels):
            label2id[f"B-{label}"] = len(label2id)
            label2id[f"I-{label}"] = len(label2id)
        return label2id

    def load_data(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        annotations = item["annotations"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = torch.zeros(self.max_length, dtype=torch.long)
        for ann in annotations:
            start = ann["start_offset"]
            end = ann["end_offset"]
            entity = ann["entity"]
            label = ann["label"]

            char_to_token = self.tokenizer(text, return_offsets_mapping=True).char_to_token
            token_start = char_to_token(start)
            token_end = char_to_token(end - 1) if end > start else token_start

            if token_start is not None and token_end is not None:
                labels[token_start] = self.label2id[f"B-{label}"]
                for i in range(token_start + 1, token_end + 1):
                    if i < self.max_length:
                        labels[i] = self.label2id[f"I-{label}"]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "text": text,
            "annotations": annotations
        }


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    texts = [item["text"] for item in batch]
    annotations = [item["annotations"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "texts": texts,
        "annotations": annotations
    }


# ==================== 模型架构部分 ====================
class EntityRecognizer(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        return logits


class QuadrupletLoss(nn.Module):
    def __init__(self, margin1=1.0, margin2=0.5):
        super().__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, anchor, positive, negative1, negative2):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist1 = torch.sum((anchor - negative1) ** 2, dim=1)
        neg_dist2 = torch.sum((anchor - negative2) ** 2, dim=1)

        loss1 = torch.relu(pos_dist - neg_dist1 + self.margin1)
        loss2 = torch.relu(pos_dist - neg_dist2 + self.margin2)
        loss = (loss1 + loss2).mean()
        return loss


class ContextualContinuityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, labels):
        loss = 0.0
        batch_size, seq_len, _ = embeddings.shape

        for i in range(batch_size):
            for j in range(seq_len - 1):
                if labels[i, j] != 0 and labels[i, j] == labels[i, j + 1]:
                    loss += torch.norm(embeddings[i, j] - embeddings[i, j + 1], p=2)

        return loss / (batch_size * seq_len)


class LLMCC(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.recognizer = EntityRecognizer(model_name, num_labels)
        self.contrastive_loss = QuadrupletLoss()
        self.context_loss = ContextualContinuityLoss()

    def forward(self,
                input_ids,
                attention_mask,
                labels,
                anchor_positions,
                positive_positions,
                negative1_positions,
                negative2_positions):
        # 实体识别
        logits = self.recognizer(input_ids, attention_mask)

        # 获取对比学习所需的嵌入
        embeddings = self.recognizer.encoder(input_ids, attention_mask).last_hidden_state

        # 收集对比学习的样本
        anchor_emb = embeddings[torch.arange(embeddings.size(0)), anchor_positions]
        positive_emb = embeddings[torch.arange(embeddings.size(0)), positive_positions]
        negative1_emb = embeddings[torch.arange(embeddings.size(0)), negative1_positions]
        negative2_emb = embeddings[torch.arange(embeddings.size(0)), negative2_positions]

        # 计算各种损失
        ce_loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        contrastive_loss = self.contrastive_loss(anchor_emb, positive_emb, negative1_emb, negative2_emb)
        context_loss = self.context_loss(embeddings, labels)

        total_loss = ce_loss + contrastive_loss + context_loss

        return {
            "logits": logits,
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "contrastive_loss": contrastive_loss,
            "context_loss": context_loss,
            "embeddings": embeddings
        }


# ==================== 实体预测增强部分 ====================
class EntityPredictionEnhancer:
    def __init__(self, model_path, entity_recognizer, tokenizer, id2label):
        self.llm = pipeline(
            "text-generation",
            model=model_path
        )
        self.recognizer = entity_recognizer
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}

    def enhance_predictions(self, dataloader, device, k=3):
        """批量增强预测结果"""
        enhanced_results = []

        for batch in tqdm(dataloader, desc="Enhancing predictions"):
            texts = batch["texts"]

            # 初始预测
            initial_entities_list = self._batch_predict_entities(batch, device)

            # 对每个文本进行增强
            for text, initial_entities in zip(texts, initial_entities_list):
                # 生成验证提示
                prompt = self._create_verification_prompt(text, initial_entities)

                # 使用LLM验证
                llm_response = self.llm(
                    prompt,
                    max_length=1024,
                    num_return_sequences=1,
                    temperature=0.7
                )[0]["generated_text"]

                # 解析LLM响应
                verified_entities = self._parse_llm_response(llm_response)

                enhanced_results.append({
                    "text": text,
                    "initial_entities": initial_entities,
                    "enhanced_entities": verified_entities
                })

        return enhanced_results

    def _batch_predict_entities(self, batch, device):
        """批量预测实体"""
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            logits = self.recognizer(input_ids, attention_mask).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        # 转换为实体列表
        entities_list = []
        for i in range(len(batch["texts"])):
            text = batch["texts"][i]
            pred = preds[i]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())

            entities = []
            current_entity = None

            for j, (token, pred_id) in enumerate(zip(tokens, pred)):
                label = self.id2label.get(pred_id, "O")

                if label.startswith("B-"):
                    if current_entity is not None:
                        entities.append(current_entity)
                    current_entity = {
                        "entity": token,
                        "label": label[2:],
                        "start": j,
                        "end": j
                    }
                elif label.startswith("I-"):
                    if current_entity is not None and current_entity["label"] == label[2:]:
                        current_entity["entity"] += token.replace("##", "")
                        current_entity["end"] = j
                    else:
                        if current_entity is not None:
                            entities.append(current_entity)
                        current_entity = None
                else:
                    if current_entity is not None:
                        entities.append(current_entity)
                    current_entity = None

            if current_entity is not None:
                entities.append(current_entity)

            # 清理实体文本
            for entity in entities:
                entity["entity"] = entity["entity"].replace("##", "").replace("[CLS]", "").replace("[SEP]", "").strip()

            entities_list.append(entities)

        return entities_list

    def _create_verification_prompt(self, text, entities):
        prompt = f"""你是一个专业的NLP专家，负责验证命名实体识别结果。请仔细检查以下文本和预测的实体：

文本: "{text}"

预测的实体:
"""
        for i, entity in enumerate(entities, 1):
            prompt += f"{i}. {entity['entity']} (类型: {entity['label']}, 位置: {entity['start']}-{entity['end']})\n"

        prompt += """
请按照以下步骤进行验证:
1. 仔细阅读文本，理解上下文
2. 检查每个预测的实体是否正确识别和分类
3. 指出任何错误的实体（包括错误的边界或错误的类型）
4. 添加任何遗漏的实体
5. 提供修正后的实体列表，格式为: [实体文本] (类型: [实体类型], 位置: [开始]-[结束])

修正后的实体列表:
"""
        return prompt

    def _parse_llm_response(self, response):
        lines = response.split("\n")
        entities = []

        for line in lines:
            if "(" in line and ")" in line and "类型:" in line and "位置:" in line:
                try:
                    entity_part = line.split("(")[0].strip()
                    type_part = line.split("类型:")[1].split(",")[0].strip()
                    pos_part = line.split("位置:")[1].split(")")[0].strip()
                    start, end = map(int, pos_part.split("-"))

                    entities.append({
                        "entity": entity_part,
                        "label": type_part,
                        "start": start,
                        "end": end
                    })
                except:
                    continue

        return entities


# ==================== 训练和评估部分 ====================
def compute_metrics(predictions, labels, id2label):
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    seqeval_report = classification_report(true_labels, true_predictions, output_dict=True)

    flat_true_labels = [l for sublist in true_labels for l in sublist]
    flat_pred_labels = [l for sublist in true_predictions for l in sublist]

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        flat_true_labels, flat_pred_labels, average='micro', zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        flat_true_labels, flat_pred_labels, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        flat_true_labels, flat_pred_labels, average='weighted', zero_division=0
    )

    return {
        "seqeval_report": seqeval_report,
        "micro_avg": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1
        },
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1
        }
    }


def evaluate(model, dataloader, device, id2label, enhancer=None):
    model.eval()
    losses = []
    all_predictions = []
    all_labels = []
    enhanced_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 随机选择对比学习的样本位置
            batch_size, seq_len = batch["labels"].shape
            anchor_positions = torch.randint(0, seq_len, (batch_size,)).to(device)
            positive_positions = torch.randint(0, seq_len, (batch_size,)).to(device)
            negative1_positions = torch.randint(0, seq_len, (batch_size,)).to(device)
            negative2_positions = torch.randint(0, seq_len, (batch_size,)).to(device)

            # 确保对比学习的样本满足条件
            for i in range(batch_size):
                # 确保anchor是实体
                while batch["labels"][i, anchor_positions[i]] == 0:
                    anchor_positions[i] = torch.randint(0, seq_len, (1,)).item()

                # 确保positive与anchor同类
                while batch["labels"][i, positive_positions[i]] != batch["labels"][i, anchor_positions[i]]:
                    positive_positions[i] = torch.randint(0, seq_len, (1,)).item()

                # 确保negative1与anchor不同类且是实体
                while batch["labels"][i, negative1_positions[i]] == batch["labels"][i, anchor_positions[i]] or \
                        batch["labels"][i, negative1_positions[i]] == 0:
                    negative1_positions[i] = torch.randint(0, seq_len, (1,)).item()

                # 确保negative2是非实体
                while batch["labels"][i, negative2_positions[i]] != 0:
                    negative2_positions[i] = torch.randint(0, seq_len, (1,)).item()

            # 前向传播
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                anchor_positions=anchor_positions,
                positive_positions=positive_positions,
                negative1_positions=negative1_positions,
                negative2_positions=negative2_positions
            )

            losses.append(outputs["total_loss"].item())
            predictions = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)

            # 如果需要增强预测
            if enhancer is not None:
                enhanced_batch = enhancer.enhance_predictions([batch], device)
                enhanced_results.extend(enhanced_batch)

    metrics = compute_metrics(all_predictions, all_labels, id2label)
    avg_loss = np.mean(losses)

    result = {
        "loss": avg_loss,
        "metrics": metrics
    }

    if enhancer is not None:
        result["enhanced_results"] = enhanced_results

    return result


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化tokenizer和数据集
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    train_dataset = NERDataset(config["data_path"], tokenizer, config["max_length"], split='train')
    test_dataset = NERDataset(config["data_path"], tokenizer, config["max_length"], split='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    # 初始化模型
    model = LLMCC(config["model_path"], len(train_dataset.label2id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"] * len(train_loader),
        eta_min=1e-6
    )

    # 初始化实体预测增强器
    enhancer = EntityPredictionEnhancer(
        config["model_path"],
        model.recognizer,
        tokenizer,
        train_dataset.id2label
    ) if config["use_enhancer"] else None

    best_f1 = 0.0
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()

            # 随机选择对比学习的样本位置
            batch_size, seq_len = batch["labels"].shape
            anchor_positions = torch.randint(0, seq_len, (batch_size,)).to(device)
            positive_positions = torch.randint(0, seq_len, (batch_size,)).to(device)
            negative1_positions = torch.randint(0, seq_len, (batch_size,)).to(device)
            negative2_positions = torch.randint(0, seq_len, (batch_size,)).to(device)

            # 确保对比学习的样本满足条件
            for i in range(batch_size):
                while batch["labels"][i, anchor_positions[i]] == 0:
                    anchor_positions[i] = torch.randint(0, seq_len, (1,)).item()

                while batch["labels"][i, positive_positions[i]] != batch["labels"][i, anchor_positions[i]]:
                    positive_positions[i] = torch.randint(0, seq_len, (1,)).item()

                while batch["labels"][i, negative1_positions[i]] == batch["labels"][i, anchor_positions[i]] or \
                        batch["labels"][i, negative1_positions[i]] == 0:
                    negative1_positions[i] = torch.randint(0, seq_len, (1,)).item()

                while batch["labels"][i, negative2_positions[i]] != 0:
                    negative2_positions[i] = torch.randint(0, seq_len, (1,)).item()

            # 前向传播
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                anchor_positions=anchor_positions,
                positive_positions=positive_positions,
                negative1_positions=negative1_positions,
                negative2_positions=negative2_positions
            )

            # 反向传播
            outputs["total_loss"].backward()
            optimizer.step()
            scheduler.step()

            train_loss += outputs["total_loss"].item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # 评估阶段
        eval_results = evaluate(model, test_loader, device, train_dataset.id2label, enhancer)

        # 打印评估结果
        print("\nEvaluation Results:")
        print(f"Test Loss: {eval_results['loss']:.4f}")
        print("\nMicro Average:")
        print(f"Precision: {eval_results['metrics']['micro_avg']['precision']:.4f}")
        print(f"Recall: {eval_results['metrics']['micro_avg']['recall']:.4f}")
        print(f"F1 Score: {eval_results['metrics']['micro_avg']['f1']:.4f}")

        # 保存最佳模型
        current_f1 = eval_results["metrics"]["micro_avg"]["f1"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), config["save_path"])
            print("\nSaved best model!")

        # 如果有增强结果，打印示例
        if enhancer is not None and "enhanced_results" in eval_results:
            print("\nEnhanced Prediction Example:")
            example = eval_results["enhanced_results"][0]
            print(f"Text: {example['text']}")
            print("Initial Entities:")
            for ent in example["initial_entities"]:
                print(f"  - {ent['entity']} ({ent['label']})")
            print("Enhanced Entities:")
            for ent in example["enhanced_entities"]:
                print(f"  - {ent['entity']} ({ent['label']})")

    print(f"\nTraining complete. Best F1 Score: {best_f1:.4f}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    train_model({
        "model_path": "D:/Code/model/Qwen2.5-7B-MedChatZH-LoRA-SFT-GPTQ-Int4",
        "data_path": "data/data.json",  # 数据文件路径
        "max_length": 1024,  # 最大序列长度
        "batch_size": 8,  # 批大小
        "lr": 2e-5,  # 学习率
        "epochs": 5,  # 训练轮数
        "save_path": "best_model.pt",  # 模型保存路径
        "use_enhancer": False  # 是否使用实体预测增强
    })
