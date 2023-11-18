import ecole
import pyscipopt


class DefaultInformationFunction:
    def before_reset(self, model):# Метод, который вызывается перед сбросом модели. 
        pass              # В данном случае, метод не выполняет никаких действий (pass)

    """
    Метод для извлечения информации о модели.
    Принимает два аргумента: model - объект модели SCIP,
    done - флаг, указывающий, завершена ли оптимизация.
    """
    def extract(self, model, done):
        m = model.as_pyscipopt() # представляет собой тот же экземпляр оптимизационной модели,
                                 # что и model, но преобразованный в формат, понятный библиотеке pyscipopt

        stage = m.getStage()    # Получает текущую стадию оптимизационной модели
              # (является ли модель еще необработанной, преобразованной, находится в стадии предварительной обработки и так далее)        
        sense = 1 if m.getObjectiveSense() == "minimize" else -1 # Определяет направление оптимизации (минимизация 1 или максимизация -1)

        primal_bound = sense * m.infinity()    # Инициализирует переменную primal_bound верхней границей значения целевой функции. 
                            # Значение m.infinity() используется для представления бесконечности в SCIP. 
                            # Значение sense используется для учета направления оптимизации.
        dual_bound = sense * -m.infinity()    # Инициализирует переменную dual_bound нижней границей значения целевой функции. 
                            # Аналогично, используется значение m.infinity(), умноженное на sense, чтобы учесть направление оптимизации.
        nlpiters = 0    # будет содержать количество итераций метода NLP, и в данном случае она начинается с нуля.
        nnodes = 0     # будет содержать количество узлов в дереве поиска, один из показателей хода процесса оптимизации.
        solvingtime = 0    # будет содержать общее время, затраченное на решение оптимизационной задачи
        status = m.getStatus()    # Получает текущий статус оптимизационной модели. 
                # Будет содержать информацию о том, завершено ли решение, произошла ли ошибка, было ли найдено оптимальное решение и так далее

        # Информация извлекается в зависимости от стадии оптимизации
        if stage >= pyscipopt.scip.PY_SCIP_STAGE.PROBLEM:
            primal_bound = m.getObjlimit()    # устанавливается в значение целевой функции, установленное в модели
            nnodes = m.getNNodes()    # Количество узлов в дереве поиска
            solvingtime = m.getSolvingTime()    # Время, затраченное на решение задачи

        if stage >= pyscipopt.scip.PY_SCIP_STAGE.TRANSFORMED:
            primal_bound = m.getPrimalbound()    # Верхняя граница (целевая функция)
            dual_bound = m.getDualbound()    # Нижняя граница

        if stage >= pyscipopt.scip.PY_SCIP_STAGE.PRESOLVING:
            nlpiters = m.getNLPIterations()    # Количество итераций нелинейного программирования

# Формируется и возвращается словарь, содержащий информацию о текущем состоянии оптимизационной модели
        return {
            "primal_bound": primal_bound,    # Верхняя граница (целевая функция)
            "dual_bound": dual_bound,        # Нижняя граница
            "nlpiters": nlpiters,            # Количество итераций нелинейного программирования
            "nnodes": nnodes,                # Количество узлов в дереве поиска
            "solvingtime": solvingtime,      # Время, затраченное на решение задачи
            "status": status,                # Статус текущей оптимизационной модели
        }

"""
Наследует от класса ecole.dynamics.PrimalSearchDynamics
Этот класс представляет собой реализацию динамики для поиска примитивных (тривиальных) решений в контексте SCIP
"""
class RootPrimalSearchDynamics(ecole.dynamics.PrimalSearchDynamics):
    def __init__(self, time_limit, n_trials=-1):    # time_limit - ограничение времени на оптимизацию,
                                    # n_trials - количество проб при каждом узле (-1 вероятно неограниченное количество проб)
        super().__init__(
            trials_per_node=n_trials,    #  количество проб при каждом узле
            depth_freq=1,    # частота выполнения проб на каждой глубине дерева поиска
            depth_start=0,     # начальная глубина, на которой применяются пробы
            depth_stop=0    # конечная глубина, на которой применяются пробы
        )  # только к корневому узлу дерева поиска
        self.time_limit = time_limit

    def reset_dynamics(self, model):    # Метод для сброса динамики. Принимает объект модели
        pyscipopt_model = model.as_pyscipopt()    # Преобразует модель model в формат, с которым можно работать в библиотеке pyscipopt

        # disable SCIP heuristics   Отключает эвристики SCIP
        pyscipopt_model.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)

        # disable restarts    Устанавливает параметры модели для отключения перезапусков оптимизации
        model.set_params(
            {"estimation/restarts/restartpolicy": "n",}
        )

        # process the root node
        done, action_set = super().reset_dynamics(model)    # вызов метода reset_dynamics родительского класса 
                                            # о множестве допустимых действий для данного состояния оптимизационной задачи

        # set time limit after reset
        reset_time = pyscipopt_model.getSolvingTime()    # затраченное на решение модели после сброса
        pyscipopt_model.setParam("limits/time", self.time_limit + reset_time)   # yстанавливает ограничение времени для модели, добавляя к текущему времени сброса значение 

        return done, action_set

"""
наследуется от класса ecole.dynamics.BranchingDynamics
Основана на обратном вызове ветвления SCIP с максимальным приоритетом и без ограничения по глубине. 
Динамика передает управление пользователю каждый раз, когда вызывается обратный вызов ветвления.
Пользователь получает в качестве набора действий список кандидатов для ветвления 
и ожидается выбор одного из них в качестве действия.

"""
class BranchingDynamics(ecole.dynamics.BranchingDynamics):
    def __init__(self, time_limit):
        super().__init__(pseudo_candidates=True) # Вызывает конструктор родительского.
                            # pseudo_candidates Определяет, содержит ли набор действий псевдо-кандидатов для ветвления (SCIPgetPseudoBranchCands)
                            # или кандидатов для ветвления по ЛП (SCIPgetPseudoBranchCands).
        self.time_limit = time_limit # Инициализирует атрибут time_limit значением, переданным в аргументе.

    def reset_dynamics(self, model):  # Метод reset_dynamics для сброса динамики
        pyscipopt_model = model.as_pyscipopt()    # Преобразует модель model в формат, с которым можно работать в библиотеке pyscipopt

        # disable SCIP heuristics
        pyscipopt_model.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF) # Отключает эвристики SCIP

        # disable restarts
        model.set_params(    #  Устанавливает параметры модели для отключения перезапусков
            {"estimation/restarts/restartpolicy": "n",}
        )

        # process the root node Для корневого узла
        done, action_set = super().reset_dynamics(model)    """Вызывает метод reset_dynamics базового класса (ecole.dynamics.BranchingDynamics) и возвращает результат.
                            В ecole: 
                                Начинает решение до первого узла ветвления.
                            Начинает решение с использованием стандартных настроек SCIP (SCIPsolve) и
                            передает управление пользователю на первом этапе ветвления.
                            Пользователи могут наследоваться от этой динамики, чтобы изменить стандартные настройки, 
                            такие как предварительное решение и плоскости разрезов.
                                        """ 
                                                        
        # set time limit after reset
        reset_time = pyscipopt_model.getSolvingTime() # Получает время, затраченное на решение модели после сброса
        pyscipopt_model.setParam("limits/time", self.time_limit + reset_time) # Устанавливает ограничение времени для модели, добавляя к текущему времени сброса значение time_limit

        return done, action_set    # done: Решена ли задача.
                            # action_set: Список индексов переменных-кандидатов для ветвления.
                                    # Доступные кандидаты зависят от параметров в init().
                                    # Индексы переменных (значения в action_set) - это их позиция в исходной задаче (SCIPvarGetProbindex).
                                    # Порядок переменных в action_set произвольный.

"""
Наследуется от класса ecole.dynamics.ConfiguringDynamics
Динамика настройки параметров решения.
Эти динамики предназначены для использования в качестве (контекстуального) бандита
для поиска хороших параметров для SCIP.
"""
class ConfiguringDynamics(ecole.dynamics.ConfiguringDynamics):
    def __init__(self, time_limit):
        super().__init__()    # Вызывает конструктор родительского класса, чтобы создать новую динамику
        self.time_limit = time_limit

    def reset_dynamics(self, model):    # Метод для сброса динамики. Принимает объект модели
        pyscipopt_model = model.as_pyscipopt() # Преобразует модель в формат, с которым можно работать в библиотеке pyscipopt

        # process the root node
        done, action_set = super().reset_dynamics(model)    """ Вызывает метод reset_dynamics базового класса
                            В ecole: 
                            Ничего не делает.
                            Пользователи могут наследоваться от этой динамики,
                            чтобы изменить момент в процессе решения, когда будут установлены параметры 
                            (например, после предварительного решения).
                            Параметры: model: Состояние процесса принятия решений Маркова. Передается средой.
                            Возвращает: done: Решена ли задача. Всегда ложь.
                                        action_set: Не используется.
                            """

        # set time limit after reset
        reset_time = pyscipopt_model.getSolvingTime() # Получает время, затраченное на решение модели после сброса.
        pyscipopt_model.setParam("limits/time", self.time_limit + reset_time) # Устанавливает ограничение времени для модели, добавляя к текущему времени сброса значение time_limit

        return done, action_set    # done: Решена ли задача. Всегда ложь.
                                    # action_set: Не используется.

    def step_dynamics(self, model, action): # Метод для выполнения шага динамики. Принимает объект модели и действие
        forbidden_params = [    # Список параметров SCIP, изменение которых запрещено
            "limits/time",
            "timing/clocktype",
            "timing/enabled",
            "timing/reading",
            "timing/rareclockcheck",
            "timing/statistictiming",
        ]

        for param in forbidden_params:    # Проверяет, содержатся ли запрещенные параметры в действии
            if param in action:    # Если да, выбрасывается исключение ValueError
                raise ValueError(f"Setting the SCIP parameter '{param}' is forbidden.")

        done, action_set = super().step_dynamics(model, action) """ Вызывает метод step_dynamics базового класса
                            В ecole: 
                            Устанавливает параметры и решает задачу.
                            Параметры: model: Состояние процесса принятия решений Маркова. Передается средой.
                                       action: Отображение имен параметров и их значений.
                            Возвращает: done: Решена ли задача. Всегда истина.
                                        action_set: Не используется.
                            """

        return done, action_set    # done: Решена ли задача. Всегда истина.
                                   # action_set: Не используется.

"""
Подкласс ecole.environment.Environment
Экосистемный частично наблюдаемый процесс принятия решений Маркова (POMDP) Ecole.
Подобно OpenAI Gym, среды представляют задачу, которую агент должен решить. 
Для максимальной настраиваемости различные компоненты объединяются в этом классе.
"""
class ObjectiveLimitEnvironment(ecole.environment.Environment):
    def reset(self, instance, objective_limit=None, *dynamics_args, **dynamics_kwargs):
        """We add one optional parameter not supported by Ecole yet: the instance's objective limit.
            Мы добавляем один дополнительный параметр, который пока не поддерживается Ecole: ограничение целевой функции для экземпляра
            instance: Задача комбинаторной оптимизации, которую необходимо решить во время нового эпизода. Это может быть либо путь к файлу с задачей, который может быть прочитан SCIP, либо модель, данные по определению задачи которой будут скопированы.
            dynamics_args: Дополнительные аргументы передаются как есть в основную динамику.
            dynamics_kwargs: Дополнительные аргументы передаются как есть в основную динамику.
        Этот метод переводит среду в новое начальное состояние, то есть начинает новый эпизод. Метод можно вызывать в любой момент времени"""
        self.can_transition = True # Устанавливается флаг can_transition в True, указывающий, что среда может выполнять переходы
        try:
            if isinstance(instance, ecole.core.scip.Model): #если входной аргумент instance является экземпляром ecole.core.scip.Model
                self.model = instance.copy_orig() # Создается копия модели
            else:
                self.model = ecole.core.scip.Model.from_file(instance) #иначе загружается модель из файла
            self.model.set_params(self.scip_params) # Устанавливаются параметры модели.

            # >>> changes specific to this environment Изменения, специфичные для этой среды
            if objective_limit is not None: # Если предоставлен ограничитель целевой функции
                self.model.as_pyscipopt().setObjlimit(objective_limit) # устанавливается соответствующий параметр модели
            # <<<

            self.dynamics.set_dynamics_random_state(self.model, self.random_engine) # Устанавливается случайное состояние для динамики среды

            # Reset data extraction functions Сбрасываются функции извлечения данных
            self.reward_function.before_reset(self.model)
            self.observation_function.before_reset(self.model)
            self.information_function.before_reset(self.model)

            # Place the environment in its initial state Среда помещается в начальное состояние путем вызова метода reset_dynamics у динамики
            done, action_set = self.dynamics.reset_dynamics(
                self.model, *dynamics_args, **dynamics_kwargs
            )
            self.can_transition = not done # Обновляется флаг can_transition на основе статуса done

            # Extract additional data to be returned by reset Извлекаются дополнительные данные для возврата из метода
            reward_offset = self.reward_function.extract(self.model, done) # вызывается метод extract объекта reward_function для извлечения 
                                                                        # значения награды из текущего состояния модели. Значение done указывает,
                                                                        # завершено ли текущее состояние (True, если среда в терминальном состоянии)
            if not done:     # Если не завершено текущее состояние
                observation = self.observation_function.extract(self.model, done) # Вызывается метод extract объекта observation_function 
                                                        # для извлечения наблюдения из текущего состояния модели
            else:     # Если среда в терминальном состоянии
                observation = None
            information = self.information_function.extract(self.model, done) # Вызывается метод для извлечения дополнительной информации из текущего состояния модели

            return observation, action_set, reward_offset, done, information 
                        """ observation: Наблюдение, извлеченное из начального состояния. 
                                         Обычно используется для выполнения следующего действия.
                            action_set: Опциональное подмножество, которое определяет, 
                                        какие действия принимаются в следующем переходе.
                                        Для некоторых сред, множество действий может изменяться при каждом переходе.
                            reward_offset: Смещение на общее накопленное вознаграждение, также известное как начальное вознаграждение. 
                                           Это вознаграждение не влияет на обучение (поскольку действий еще не было предпринято), 
                                           но может быть использовано для целей оценки. Например, в общем накопленном вознаграждении 
                                           эпизода можно учесть вычисления, произошедшие во время reset() (например, время вычислений, количество итераций ЛП в предварительном решении и так далее).
                            done: Булев флаг, указывающий, является ли текущее состояние терминальным.
                                  Если этот флаг истинен, то текущий эпизод завершен, и метод step() больше вызвать нельзя.
                            info: Коллекция средоспецифичной информации о переходе. 
                                  Это необходимо для проблемы управления, но полезно для получения представления о среде.
                        """
        except Exception as e:
            self.can_transition = False # Обработчик исключений устанавливает can_transition в False
            raise e            # и повторно вызывает исключение в случае возникновения ошибки

# Этот класс представляет собой окружение для корневого прямого поиска
class RootPrimalSearch(ObjectiveLimitEnvironment):
    __Dynamics__ = RootPrimalSearchDynamics
    __DefaultInformationFunction__ = DefaultInformationFunction

# Этот класс представляет собой окружение для процесса ветвления
class Branching(ObjectiveLimitEnvironment):
    __Dynamics__ = BranchingDynamics
    __DefaultInformationFunction__ = DefaultInformationFunction

# Этот класс представляет собой окружение для конфигурации
class Configuring(ObjectiveLimitEnvironment):
    __Dynamics__ = ConfiguringDynamics
    __DefaultInformationFunction__ = DefaultInformationFunction
