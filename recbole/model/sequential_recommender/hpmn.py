import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_, xavier_normal_, constant_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import RegLoss, BPRLoss

class HPMN(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # load parameters info
        self.embedding_size = 32
        self.dropout_prob = 0.2
        # load dataset info
        self.n_users = dataset.user_num

        self.hidden_size = 32
        self.loss = nn.CrossEntropyLoss(reduction='none')
        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        self.user_num_layers = [2, 2, 2, 5, 5, 1]
        self.item_num_layers = [2, 2, 2, 3, 1]

        self.gru_layers = nn.ModuleList((
            nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            bias=False,
            batch_first=True),
            nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            bias=False,
            batch_first=True),
            nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            bias=False,
            batch_first=True)
        )) 
        self.linears = nn.Sequential(
            nn.Linear(4*32, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )
        self.query_fc = nn.Linear(32, 32)
        self.query_weight = nn.Parameter(torch.zeros((self.hidden_size, self.hidden_size)))
        self.out_layers = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Linear(128, 200),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(200, 80),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(80, 32),
            nn.Sigmoid()
        )
        self.loss_fct = nn.CrossEntropyLoss()

    def get_covreg(self, memory):
        mean = torch.mean(memory, dim=2, keepdim=True)
        C = memory - mean
        C = torch.matmul(C, torch.transpose(C, 2, 1)) / memory.shape[2]
        C_diag = torch.diagonal(C)
        C_diag = torch.diag_embed(C_diag)
        C = C - C_diag
        norm = torch.norm(C, p='fro', dim=[1, 2])
        return torch.sum(norm)


    def build_memory(self, embs, li_layer, max_len):
        memory = []
        for i in range(3):
            outputs, states = self.gru_layers[i](embs)
            memory.append(states.unsqueeze(1))

            max_len /= li_layer[i]
            max_len = int(max_len)
            outputs = torch.reshape(outputs,
                                 [-1, max_len, li_layer[i], self.hidden_size])
            embs = torch.reshape(outputs[:,:,li_layer[i] - 1],[-1, max_len, self.hidden_size])
        memory = torch.cat(memory, axis=2).squeeze(0)
        # loss = self.get_covreg(memory)
        return memory, None

    def attention(self, key, value, query, k):
        k = key.shape[1]
        queries = torch.tile(torch.unsqueeze(query, 1), [1, k, 1])  # [B, T, Dk]
        inp = torch.cat([queries, key.expand(queries.shape), queries - key, queries * key], dim=-1)

        score = self.linears(inp)
        score = torch.softmax(torch.reshape(score, [-1, k]), dim=-1)  # [B, T]

        atten_output = torch.multiply(value, torch.unsqueeze(score, 2))
        atten_output_sum = torch.sum(atten_output, dim=1)

        return atten_output_sum, score

    def query_memory(self, query, memory, k=3):
        query = self.query_fc(query)
        weights = []
        for _ in range(4):
            read, weight = self.attention(memory, memory, query, k)
            query = torch.matmul(query, self.query_weight) + read
            weights.append(weight)

        return query, weights[0]

    def build_fc_net(self, inp):
        out = self.out_layers(inp)
        return out

    def forward(self, user_batch, seq_batch):
        users = self.user_embedding(user_batch).unsqueeze(1)
        seqs = self.item_embedding(seq_batch)
        zeros = torch.zeros_like(seqs[:, :23, :], dtype=torch.float32)
        uinp = torch.cat([zeros, users], axis=1)
        memory, umloss = self.build_memory(uinp, self.user_num_layers, 192)
        last = uinp[:, -2, :]
        query, self.user_weights = self.query_memory(
            last, memory)
        user_repre = torch.cat([query, last], axis=-1)

        zeros = torch.zeros_like(seqs[:, :192 - 184, :], dtype=torch.float32)
        iinp = torch.cat([zeros, seqs], axis=1)
        memory, imloss = self.build_memory(iinp, self.item_num_layers, 192)
        last = iinp[:, -1, :]
        query, self.item_weights = self.query_memory(
            last, memory)
        item_repre = torch.cat([query, last], axis=-1)

        repre = torch.cat([user_repre, item_repre], axis=-1)

        return self.build_fc_net(repre)


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        user_batch = interaction[self.USER_ID]
        seq_output = self.forward(user_batch, item_seq)
        pos_items = interaction[self.POS_ITEM_ID]
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        user_batch = interaction[self.USER_ID]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(user_batch, item_seq)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores


        pass
