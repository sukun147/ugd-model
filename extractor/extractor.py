import random

import numpy as np
import torch
import torch.nn as nn
from py2neo import Graph


class KnowledgeExtractor:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, embedding_model_name="intfloat/multilingual-e5-small",
                 device="cuda"):
        """
        初始化知识提取器

        参数:
            neo4j_uri: Neo4j数据库连接URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            embedding_model_name: 嵌入模型名称
        """
        # 设置设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 连接Neo4j知识图谱
        self.kg = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # 初始化嵌入模型
        # self.embedding_model = SentenceTransformer(embedding_model_name).to(self.device)
        # self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # 初始化策略网络
        # self.policy_net = PolicyNetwork(self.embedding_dim * 2, self.embedding_dim).to(self.device)  # 状态是当前实体和目标实体的组合

        # 优化器
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        # 经验回放
        # self.replay_buffer = deque(maxlen=10000)

        # GRPO特有参数
        # self.gamma = 0.99  # 折扣因子
        # self.epsilon = 0.1  # 探索率
        # self.max_path_len = 10  # 最大路径长度
        # self.batch_size = 1  # 训练批大小
        # self.eta = 0.2  # 信任区域半径
        # self.max_kl = 0.01  # KL散度约束
        # self.gae_lambda = 0.95  # GAE参数

    def _get_entity_embedding(self, entity_name):
        """
        获取实体的嵌入表示
        """
        return self.embedding_model.encode("query: " + entity_name, convert_to_tensor=True,
                                           normalize_embeddings=True).to(self.device)

    def extract_subgraph(self, source_entities, target_entities, max_hops=2):
        """
        提取源实体和目标实体周围的子图（启发式方法）

        参数:
            source_entities: 源实体列表
            target_entities: 目标实体列表
            max_hops: 最大跳数

        返回:
            子图中的所有实体和关系
        """
        # 获取所有相关实体
        all_entities = set(source_entities + target_entities)

        # 扩展源实体周围的子图
        for entity in source_entities:
            query = f"""
            MATCH (e)-[r*1..{max_hops}]-(neighbor)
            WHERE e.name = '{entity}'
            RETURN e, r, neighbor
            """
            result = self.kg.run(query).data()
            for item in result:
                if 'name' in item['neighbor']:
                    all_entities.add(item['neighbor']['name'])

        # 扩展目标实体周围的子图
        for entity in target_entities:
            query = f"""
            MATCH (e)-[r*1..{max_hops}]-(neighbor)
            WHERE e.name = '{entity}'
            RETURN e, r, neighbor
            """
            result = self.kg.run(query).data()
            for item in result:
                if 'name' in item['neighbor']:
                    all_entities.add(item['neighbor']['name'])

        # 获取子图中所有实体之间的关系
        triples = []
        for entity in all_entities:
            query = f"""
            MATCH (e)-[r]->(neighbor)
            WHERE e.name = '{entity}' AND neighbor.name IN {list(all_entities)}
            RETURN e.name, type(r), neighbor.name
            """
            result = self.kg.run(query).data()
            for item in result:
                triples.append((item['e.name'], item['type(r)'], item['neighbor.name']))

        return list(all_entities), triples

    def _get_neighbors(self, entity_name):
        """
        获取实体的所有邻居节点
        """
        query = f"""
        MATCH (e)-[r]->(neighbor)
        WHERE e.name = '{entity_name}'
        RETURN type(r) as relation, neighbor.name as neighbor
        """
        result = self.kg.run(query).data()
        return [(item['relation'], item['neighbor']) for item in result]

    def _get_state(self, current_entity, target_entity):
        current_embedding = self._get_entity_embedding(current_entity)  # [embedding_dim]
        target_embedding = self._get_entity_embedding(target_entity)  # [embedding_dim]
        return torch.cat([current_embedding, target_embedding - current_embedding])

    def train(self, train_questions, num_epochs=100):
        """训练主循环"""
        for epoch in range(num_epochs):
            epoch_rewards = []
            for question in train_questions:
                # 1. 提取实体
                source_entities = question['source_entities']
                target_entities = question['target_entities']

                # 2. 收集经验
                paths, rewards = self._collect_experience(
                    question['text'], source_entities, target_entities
                )
                epoch_rewards.extend(rewards)

                # 3. 策略更新
                if len(self.replay_buffer) >= self.batch_size:
                    loss = self._update_policy()
                    print(f'Epoch {epoch + 1} | Avg Reward: {np.mean(epoch_rewards):.2f} | Loss: {loss:.4f}')

    def _collect_experience(self, question, sources, targets):
        """使用ε-greedy策略收集轨迹"""
        paths, rewards = [], []
        for source in sources:
            for target in targets:
                # 轨迹生成
                path = self._rollout_path(source, target)
                if not path:
                    continue

                # 计算三部分奖励（论文式2-4）
                reward = self.compute_rewards(question, path, target)

                # 存储经验
                self.replay_buffer.append((source, target, path, reward))
                paths.append(path)
                rewards.append(reward.cpu())
        return paths, rewards

    def _rollout_path(self, source, target):
        """执行轨迹生成"""
        path = []
        current = source

        for _ in range(self.max_path_len):
            neighbors = self._get_neighbors(current)
            if not neighbors:
                break

            # 状态表示（论文式1）
            state = self._get_state(current, target)

            # ε-greedy动作选择
            if random.random() < self.epsilon:
                relation, next_entity = random.choice(neighbors)
            else:
                neighbor_embs = torch.stack([
                    self._get_entity_embedding(n[1]) for n in neighbors
                ]).to(self.device)
                action_probs = self.policy_net(
                    state.unsqueeze(0), neighbor_embs.unsqueeze(0)
                )
                relation, next_entity = neighbors[torch.argmax(action_probs).item()]

            path.append((current, relation, next_entity))
            current = next_entity

            if next_entity == target:
                break
        return path

    def _update_policy(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, masks = [], [], [], []

        # 准备batch数据
        for (source, target, path, reward) in batch:
            current = path[0][0]
            state = self._get_state(current, target)
            neighbors = self._get_neighbors(current)
            action_idx = [n[1] for n in neighbors].index(path[0][2])

            states.append(state)
            actions.append(action_idx)
            rewards.append(reward)
            masks.append(1 if path[-1][1] == target else 0)  # 是否到达目标

        # 转换为张量
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        masks = torch.tensor(masks).to(self.device)

        # 获取邻居嵌入（带padding处理）
        neighbor_embs = []
        max_neighbors = max(len(self._get_neighbors(p[0][0])) for _, _, p, _ in batch)
        for _, _, path, _ in batch:
            neighbors = self._get_neighbors(path[0][0])
            embeds = torch.stack([self._get_entity_embedding(n[1]) for n in neighbors])
            padding = torch.zeros(max_neighbors - len(neighbors), self.embedding_dim).to(self.device)
            neighbor_embs.append(torch.cat([embeds, padding]))
        neighbor_embs = torch.stack(neighbor_embs)

        # 计算旧策略概率
        with torch.no_grad():
            old_probs = self.policy_net(states, neighbor_embs).gather(1, actions.unsqueeze(1))

        # 计算GAE优势（论文式5）
        values = rewards  # 简化版价值估计
        advantages = self._compute_gae(rewards, values, masks)

        # GRPO多步更新（含KL散度约束）
        loss_history = []
        for _ in range(3):  # 多步更新
            new_probs = self.policy_net(states, neighbor_embs).gather(1, actions.unsqueeze(1))
            ratio = (new_probs / old_probs).squeeze()

            # KL散度约束
            kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).mean()
            if kl > self.max_kl:
                break

            # 带clip的损失函数
            loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - self.eta, 1 + self.eta) * advantages
            ).mean() + 0.1 * kl  # KL正则化系数

            # 梯度更新
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()

            loss_history.append(loss.item())

        return np.mean(loss_history)

    def compute_rewards(self, question_context, path, target_entity):
        """
        计算路径的奖励
        """
        # 1. 可达性奖励
        reach_reward = 1.0 if path[-1][2] == target_entity else -1.0
        # 2. 上下文相关性奖励
        path_text = " ".join([f"{h} {r} {t}" for h, r, t in path])
        path_embedding = self.embedding_model.encode("query: " + path_text, convert_to_tensor=True,
                                                     normalize_embeddings=True).to(self.device)
        context_embedding = self.embedding_model.encode("query: " + question_context, convert_to_tensor=True,
                                                        normalize_embeddings=True).to(self.device)
        context_reward = (torch.cosine_similarity(path_embedding, context_embedding, dim=0) + 1) / 2  # 映射到[0,1]
        # 3. 简洁性奖励
        conciseness_reward = 1.0 / len(path)
        # 综合奖励
        return reach_reward * 0.5 + context_reward * 0.3 + conciseness_reward * 0.2

    def _compute_gae(self, rewards, values, masks):
        """计算GAE优势"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * masks[t] - values[t] if t < len(rewards) - 1 else 0
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * masks[t] * last_advantage
        return advantages


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, neighbor_dim):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.neighbor_net = nn.Sequential(
            nn.Linear(256 + neighbor_dim, 128),
            nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.out = nn.Linear(128, 1)

    def forward(self, state, neighbor_embeddings):
        """
        参数:
            state: [batch_size, state_dim]
            neighbor_embeddings: [batch_size, max_neighbors, neighbor_dim]
        """
        # 处理状态特征
        state_feat = self.state_net(state)
        state_feat = state_feat.unsqueeze(1).expand(-1, neighbor_embeddings.size(1), -1)

        # 拼接特征
        combined = torch.cat([state_feat, neighbor_embeddings], dim=-1)
        neighbor_feat = self.neighbor_net(combined)

        # 注意力机制
        attn_out, _ = self.attention(
            neighbor_feat.transpose(0, 1),
            neighbor_feat.transpose(0, 1),
            neighbor_feat.transpose(0, 1)
        )
        scores = self.out(attn_out.transpose(0, 1)).squeeze(-1)

        # 处理padding
        mask = (neighbor_embeddings.abs().sum(-1) != 0).float()
        scores = scores - (1 - mask) * 1e9
        return torch.softmax(scores, dim=-1)


if __name__ == '__main__':
    # 初始化知识提取器
    extractor = KnowledgeExtractor(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="sukun031015",
        embedding_model_name="intfloat/multilingual-e5-small"
    )

    source_entities = ["生姜"]
    target_entities = []

    subgraph_entities, subgraph_triples = extractor.extract_subgraph(
        source_entities, target_entities, max_hops=1
    )
    print(subgraph_triples)

    # 存在梯度消失问题，策略没有得到训练
    # train_data = [
    #     {
    #         "text": "咳嗽使用什么中药化痰止咳？",
    #         "source_entities": ["咳嗽"],
    #         "target_entities": ["化痰止咳"]
    #     },
    #     {
    #         "text": "头晕使用什么中药清心？",
    #         "source_entities": ["头晕"],
    #         "target_entities": ["清心"]
    #     }
    # ]
    #
    # extractor.train(train_data, num_epochs=50)
