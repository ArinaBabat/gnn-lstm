import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np

# Проверяем доступность GPU и устанавливаем устройство
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PreNormException(Exception):
    pass

"""
PreNormLayer - класс, который наследуется от torch.nn.Module
 и реализует функциональность слоя предварительной нормализации.
"""
class PreNormLayer(torch.nn.Module):
    """
    Конструктор класса
    n_units - количество единиц входных данных
    shift - флаг, указывающий, должен ли выполняться сдвиг 
    scale - флаг, указывающий, должно ли выполняться масштабирование
    """
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        # Убеждаемся, что shift или scale установлены в True
        assert shift or scale
        # Регистрируем shift и scale в буфере
        self.register_buffer("shift", torch.zeros(n_units) if shift else None)
        self.register_buffer("scale", torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    """
    Метод, который выполняет прямой проход через слой.
    Он принимает входные данные input_ и возвращает результат
    преобразования. Если ожидаются обновления, метод вызывает исключение 
    """
    def forward(self, input_):
        # Если ожидаются обновления, вызываем исключение
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException
        # Добавляем shift, если он задан
        if self.shift is not None:
            input_ = input_ + self.shift
        # Умножаем на scale, если он задан
        if self.scale is not None:
            input_ = input_ * self.scale
        output = torch.matmul(input_, self.weight) + self.bias    # учёт весов
        return output
    """
    Метод для начала обновления статистики слоя.
    Он инициализирует переменные для оценки среднего и дисперсии.
    """
    def start_updates(self):
        # Инициализируем переменные для обновления статистики
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False
    """
    Метод для обновления статистики слоя в режиме онлайн.
    Он принимает входные данные input_ и вычисляет среднее и дисперсию. 
    """
    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        # Проверяем размерность входных данных
        assert (
            self.n_units == 1 or input_.shape[-1] == self.n_units
        ), f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."
        # Переформатируем входные данные, если необходимо
        input_ = input_.reshape(-1, self.n_units)
        # Вычисляем среднее значение по каждому признаку
        sample_avg = input_.mean(dim=0)
        # Вычисляем выборочную дисперсию по каждому признаку
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        # Вычисляем количество выборок
        sample_count = np.prod(input_.size()) / self.n_units
        # Вычисляем разницу между текущим средним и новыми значениями среднего
        delta = sample_avg - self.avg
        # Обновляем второй момент (ускоренная формула Welford'a)
        self.m2 = (
            self.var * self.count
            + sample_var * sample_count
            + delta ** 2 * self.count * sample_count / (self.count + sample_count)
        )

        self.count += sample_count    # Обновляем общее количество выборок
        self.avg += delta * sample_count / self.count    # Обновляем среднее значение
        self.var = self.m2 / self.count if self.count > 0 else 1    # Обновляем дисперсию (если count > 0)
    """
    Метод для завершения предварительного обучения слоя.
    Он фиксирует параметры слоя, основываясь на оценках среднего
    и дисперсии, и удаляет временные переменные.
    """
    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0    # Проверяем, что было хотя бы одно обновление
        if self.shift is not None:    # Обновляем сдвиг, если он задан
            self.shift = -self.avg

        if self.scale is not None:    # Обновляем масштаб, если он задан
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        # Удаляем временные переменные
        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False

#Класс, представляющий графовую свертку для двудольного графа.
class BipartiteGraphConvolution64(torch_geometric.nn.MessagePassing):
    """
    конструктор класса
    Вызывает конструктор родительского класса MessagePassing с аргументом "add"
    Инициализирует размерность встроенных признаков emb_size равной 64.
    Инициализирует модули для обработки признаков
    """
    def __init__(self):
        super().__init__("add")
        emb_size = 64
        # Инициализация модулей для обработки признаков

        self.feature_module_left = torch.nn.Sequential(    # Модуль для обработки признаков левой вершины
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(    # Модуль для обработки признаков ребра
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(    # Модуль для обработки признаков правой вершины
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(    # Модуль для финальной обработки признаков
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(PreNormLayer(1, shift=False))    # Модуль для постобработки признаков после свертки

        # output_layers
        self.output_module = torch.nn.Sequential(    # Модуль для выходных слоев
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    """
    Метод, который выполняет прямой проход через графовую свертку.
    Он принимает признаки левой вершины, индексы ребер, признаки ребер и признаки правой вершины.
    Метод вызывает метод propagate с передачей соответствующих аргументов
    и возвращает результат обработки выхода свертки и признаков правой вершины.
    """
    def forward(self, left_features, edge_indices, edge_features, right_features):    # Прямой проход через графовую свертку
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(    # Обработка выхода свертки и правой вершины
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    """
    Метод, который вычисляет сообщение между вершинами.
    Он принимает признаки вершины i, признаки вершины j и признаки ребра.
    Метод вычисляет выход, применяя последовательность модулей для обработки признаков
    к сумме признаков левой вершины, признаков ребра и признаков правой вершины.
    """
    def message(self, node_features_i, node_features_j, edge_features):    # Вычисление сообщения между вершинами
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


class BipartiteGraphConvolution128(torch_geometric.nn.MessagePassing):
    def __init__(self):
        super().__init__("add")
        emb_size = 128

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(PreNormLayer(1, shift=False))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output

class BipartiteGraphConvolution256(torch_geometric.nn.MessagePassing):
    def __init__(self):
        super().__init__("add")
        emb_size = 256

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(PreNormLayer(1, shift=False))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


class BaseModel(torch.nn.Module):
    """
    Our base model class, which implements pre-training methods.
    """
    """
    Инициализация предварительного обучения.
    Запускает обновление статистики для всех модулей PreNormLayer в модели.
    Проходит по всем модулям внутри модели с помощью self.modules().
    Если модуль является экземпляром класса PreNormLayer, вызывается метод start_updates(),
    который инициирует обновление статистики для этого модуля.
    """
    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()
    """
    Получение следующего модуля PreNormLayer для обновления статистики.
    Возвращает первый модуль, который ожидает обновления и уже получил обновления,
    и останавливает обновление статистики для этого модуля.
    Если таких модулей нет, возвращает None.
    """
    def pre_train_next(self):
        for module in self.modules():    #Проходит по всем модулям внутри модели 
            """
            Если модуль является экземпляром класса PreNormLayer, проверяется,
            ожидает ли модуль обновления и уже получил ли модуль обновления.
            """
            if (
                isinstance(module, PreNormLayer)
                and module.waiting_updates
                and module.received_updates
            ):
                """
                Если оба условия выполняются, вызывается метод stop_updates(),
                который останавливает обновление статистики для этого модуля,
                и модуль возвращается в качестве результата.
                """
                module.stop_updates()
                return module
        return None

    def pre_train(self, *args, **kwargs):
        """
        Метод предварительного обучения.
        Запускает прямой проход через модель с переданными аргументами, игнорируя градиенты.
        Если в процессе прямого прохода возникает исключение PreNormException, возвращает True.
        В противном случае, возвращает False.
        """
        try:
            with torch.no_grad():
                self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True


class GNNPolicy2_64_0(BaseModel):
    """
    Класс, который является подклассом BaseModel.
    Реализует модель графовой нейронной сети с политикой.
    """
    def __init__(self):
        super().__init__()
        emb_size = 64    # Размерность вложения
        cons_nfeats = 5    # Количество признаков для ограничений
        edge_nfeats = 1    # Количество признаков для ребер
        var_nfeats = 17    # Количество признаков для переменных

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(    # Вложение ограничений
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)    # Вложение ребер

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(    # Вложение переменных
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution64()    # Свертка от переменных к ограничениям
        self.conv_c_to_v = BipartiteGraphConvolution64()    # Свертка от ограничений к переменным

        self.output_module = torch.nn.Sequential(    # Выходной модуль
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )
        
    """
        Прямой проход через модель GNNPolicy
        Аргументы:
            constraint_features: Тензор признаков ограничений размером [num_constraints, cons_nfeats].
            edge_indices: Тензор индексов ребер размером [2, num_edges].
            edge_features: Тензор признаков ребер размером [num_edges, edge_nfeats].
            variable_features: Тензор признаков переменных размером [num_variables, var_nfeats].

        Возвращает:
            Выход модели размером [num_variables], представляющий политику.
        """
    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )
        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy2_64_1(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution64()
        self.conv_c_to_v = BipartiteGraphConvolution64()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy2_64_2(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution64()
        self.conv_c_to_v = BipartiteGraphConvolution64()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy2_64_3(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution64()
        self.conv_c_to_v = BipartiteGraphConvolution64()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy2_128_0(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 128
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution128()
        self.conv_c_to_v = BipartiteGraphConvolution128()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )




        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy2_128_1(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 128
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution128()
        self.conv_c_to_v = BipartiteGraphConvolution128()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy2_128_2(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 128
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution128()
        self.conv_c_to_v = BipartiteGraphConvolution128()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy2_128_3(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 128
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution128()
        self.conv_c_to_v = BipartiteGraphConvolution128()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy2_256_0(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 256
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution256()
        self.conv_c_to_v = BipartiteGraphConvolution256()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )




        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy2_256_1(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 256
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution256()
        self.conv_c_to_v = BipartiteGraphConvolution256()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy2_256_2(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 256
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution256()
        self.conv_c_to_v = BipartiteGraphConvolution256()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy2_256_3(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 256
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
        )
        
        self.conv_v_to_c = BipartiteGraphConvolution256()
        self.conv_c_to_v = BipartiteGraphConvolution256()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output


class GNNPolicy3_64_0(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution64()
        self.conv_c_to_v = BipartiteGraphConvolution64()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )




        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy3_64_1(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),            
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),            
        )

        self.conv_v_to_c = BipartiteGraphConvolution64()
        self.conv_c_to_v = BipartiteGraphConvolution64()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy3_64_2(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),            
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),            
        )

        self.conv_v_to_c = BipartiteGraphConvolution64()
        self.conv_c_to_v = BipartiteGraphConvolution64()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy3_64_3(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),            
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),            
        )

        self.conv_v_to_c = BipartiteGraphConvolution64()
        self.conv_c_to_v = BipartiteGraphConvolution64()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy3_128_0(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 128
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),            
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),            
        )

        self.conv_v_to_c = BipartiteGraphConvolution128()
        self.conv_c_to_v = BipartiteGraphConvolution128()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )




        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy3_128_1(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 128
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),            
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),            
        )

        self.conv_v_to_c = BipartiteGraphConvolution128()
        self.conv_c_to_v = BipartiteGraphConvolution128()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy3_128_2(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 128
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),            
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),            
        )

        self.conv_v_to_c = BipartiteGraphConvolution128()
        self.conv_c_to_v = BipartiteGraphConvolution128()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy3_128_3(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 128
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),            
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),            
        )

        self.conv_v_to_c = BipartiteGraphConvolution128()
        self.conv_c_to_v = BipartiteGraphConvolution128()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy3_256_0(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 256
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),            
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),            
        )

        self.conv_v_to_c = BipartiteGraphConvolution256()
        self.conv_c_to_v = BipartiteGraphConvolution256()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )




        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy3_256_1(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 256
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),            
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),            
        )

        self.conv_v_to_c = BipartiteGraphConvolution256()
        self.conv_c_to_v = BipartiteGraphConvolution256()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy3_256_2(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 256
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),            
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),            
        )

        self.conv_v_to_c = BipartiteGraphConvolution256()
        self.conv_c_to_v = BipartiteGraphConvolution256()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output

class GNNPolicy3_256_3(BaseModel):
    def __init__(self):
        super().__init__()
        emb_size = 256
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 17

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),            
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(PreNormLayer(edge_nfeats),)

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),            
        )
        
        self.conv_v_to_c = BipartiteGraphConvolution256()
        self.conv_c_to_v = BipartiteGraphConvolution256()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output



