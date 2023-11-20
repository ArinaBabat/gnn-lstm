import os    # функции для работы с операционной системой
import torch    # инструменты для работы с тензорами, автоматического дифференцирования, оптимизации и построения глубоких нейронных сетей
import ecole as ec
import numpy as np
import collections    # альтернативные типы контейнеров данных, дополнительные возможности по управлению коллекциями объектов
import random
from agent_model import GNNPolicy2_64_0, GNNPolicy2_64_1, GNNPolicy2_64_2, GNNPolicy2_64_3, GNNPolicy2_128_0, \
    GNNPolicy2_128_1, GNNPolicy2_128_2, GNNPolicy2_128_3, GNNPolicy2_256_0, GNNPolicy2_256_1, GNNPolicy2_256_2, \
    GNNPolicy2_256_3, GNNPolicy3_64_0, GNNPolicy3_64_1, GNNPolicy3_64_2, GNNPolicy3_64_3, GNNPolicy3_128_0, \
    GNNPolicy3_128_1, GNNPolicy3_128_2, GNNPolicy3_128_3, GNNPolicy3_256_0, GNNPolicy3_256_1, GNNPolicy3_256_2, \
    GNNPolicy3_256_3


class ObservationFunction(ec.observation.NodeBipartite):

    def __init__(self, problem):
        super().__init__()

    def seed(self, seed):
        pass


class Policy():

    def __init__(self, problem):
        self.rng = np.random.RandomState()

        self.device = f"cuda:0"
        self.problem = problem

        if problem == 'policy2_64_0':
            params_path = 'type0.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy2_64_1':
            params_path = 'type1.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy2_64_2':
            params_path = 'type2.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy2_64_3':
            params_path = 'type3.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)

        if problem == 'policy2_128_0':
            params_path = 'type4.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy2_128_1':
            params_path = 'type5.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy2_128_2':
            params_path = 'type6.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy2_128_3':
            params_path = 'type7.pkl'

            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy2_256_0':
            params_path = 'type8.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy2_256_1':
            params_path = 'type9.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy2_256_2':
            params_path = 'type10.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy2_256_3':
            params_path = 'type11.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)            


        if problem == 'policy3_64_0':
            params_path = 'type12.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy3_64_1':
            params_path = 'type13.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy3_64_2':
            params_path = 'type14.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy3_64_3':
            params_path = 'type15.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)

        if problem == 'policy3_128_0':
            params_path = 'type16.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy3_128_1':
            params_path = 'type17.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy3_128_2':
            params_path = 'type18.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy3_128_3':
            params_path = 'type19.pkl'
            
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy3_256_0':
            params_path = 'type20.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy3_256_1':
            params_path = 'type21.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy3_256_2':
            params_path = 'type22.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)
        if problem == 'policy3_256_3':
            params_path = 'type23.pkl'
            self.policy = GNNPolicy2_64_0().to(self.device)   

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def __call__(self, action_set, observation):
        if self.problem == 'load_balancing':
            return random.choice(action_set)
        else:
            variable_features = observation.column_features
            variable_features = np.delete(variable_features, 14, axis=1)
            variable_features = np.delete(variable_features, 13, axis=1)

            constraint_features = torch.FloatTensor(observation.row_features).to(self.device)
            edge_index = torch.LongTensor(observation.edge_features.indices.astype(np.int64)).to(self.device)
            edge_attr = torch.FloatTensor(np.expand_dims(observation.edge_features.values, axis=-1)).to(self.device)
            variable_features = torch.FloatTensor(variable_features).to(self.device)
            action_set = torch.LongTensor(np.array(action_set, dtype=np.int64)).to(self.device)

            logits = self.policy(constraint_features, edge_index, edge_attr, variable_features)
            logits = logits[action_set]
            action_idx = logits.argmax().item()
            action = action_set[action_idx]

            return action
