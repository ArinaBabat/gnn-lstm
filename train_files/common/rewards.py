import ecole
import pyscipopt    # интерфейс для SCIP


class IntegralParameters:
    def __init__(self, offset=None, initial_primal_bound=None, initial_dual_bound=None):
        # Установка атрибутов объекта
        self._offset = offset  # Смещение
        self._initial_primal_bound = initial_primal_bound  # Начальная примитивная оценка
        self._initial_dual_bound = initial_dual_bound  # Начальная двойственная оценка

    # Метод для извлечения значений параметров
    def fetch_values(self, model):
        # Проверка и установка смещения
        self.offset = self._offset() if callable(self._offset) else self._offset

        # Проверка и установка начальной примитивной оценки
        self.initial_primal_bound = (
            self._initial_primal_bound()
            if callable(self._initial_primal_bound)
            else self._initial_primal_bound
        )

        # Проверка и установка начальной двойственной оценки
        self.initial_dual_bound = (
            self._initial_dual_bound()
            if callable(self._initial_dual_bound)
            else self._initial_dual_bound
        )

        # Получение значений по умолчанию, если они не предоставлены
        if self.offset is None:
            self.offset = 0.0  # Значение смещения по умолчанию

        if self.initial_primal_bound is None:
            self.initial_primal_bound = model.as_pyscipopt().getObjlimit()  # Получение ограничения для начальной примитивной оценки

        if self.initial_dual_bound is None:
            m = model.as_pyscipopt()
            # Установка начальной двойственной оценки в бесконечность или минус бесконечность в зависимости от направления оптимизации
            self.initial_dual_bound = (
                -m.infinity() if m.getObjectiveSense() == "minimize" else m.infinity()
            )

"""
Определение класса TimeLimitPrimalIntegral, который является подклассом ecole.reward.PrimalIntegral
В ecole: Разность примитивного интеграла.
    Награда определяется как примитивный интеграл с момента предыдущего состояния,
    где интеграл вычисляется относительно времени решения. Время решения зависит от
    операционной системы: оно включает время, затраченное в методе reset(),
    и время, затраченное на ожидание от агента.

"""
class TimeLimitPrimalIntegral(ecole.reward.PrimalIntegral):
    # Конструктор класса
    def __init__(self):
        # Инициализация атрибута parameters объектом класса IntegralParameters
        self.parameters = IntegralParameters()
        
        # Вызов конструктора суперкласса ecole.reward.PrimalIntegral
        super().__init__(    # Создает функцию награды PrimalIntegral.
            wall=True,  # Использовать стеночные временные лимиты (используется время от стены (wall time))
            """
            Функция, которая принимает модель ecole и возвращает кортеж начального примитивного
                ограничения и смещения для вычисления примитивного ограничения относительно него.
                Значения должны быть упорядочены как (смещение, начальное примитивное ограничение).
                Если не предоставлено, функция по умолчанию возвращает (0, -1e20) для задачи
                на максимумизацию и (0, 1e20) в противном случае.
            """
            bound_function=lambda model: (
                self.parameters.offset,
                self.parameters.initial_primal_bound,
            ),  # Функция для получения ограничений для интеграла
        )

    # Метод для установки параметров
    def set_parameters(
        self, objective_offset=None, initial_primal_bound=None, initial_dual_bound=None
    ):
        # Создание нового объекта IntegralParameters с переданными значениями
        self.parameters = IntegralParameters(
            offset=objective_offset,
            initial_primal_bound=initial_primal_bound,
            initial_dual_bound=initial_dual_bound,
        )

    # Метод, вызываемый перед сбросом состояния среды
    def before_reset(self, model):
        # Извлечение значений параметров из модели и установка их в объект parameters
        self.parameters.fetch_values(model)
        
        # Вызов соответствующего метода суперкласса
        super().before_reset(model)

    # Метод для извлечения значения интеграла
    def extract(self, model, done):
        # Извлечение значения интеграла с использованием метода суперкласса
        reward = super().extract(model, done)

        # Коррекция значения интеграла, если временной лимит не достигнут
        if done:
            m = model.as_pyscipopt()
            
            # Вычисление времени, оставшегося до истечения временного лимита
            time_left = max(m.getParam("limits/time") - m.getSolvingTime(), 0)
            
            # Получение текущего примитивного ограничения
            if m.getStage() < pyscipopt.scip.PY_SCIP_STAGE.TRANSFORMED:
                primal_bound = m.getObjlimit()
            else:
                primal_bound = m.getPrimalbound()

            # Получение значений из параметров
            offset = self.parameters.offset
            initial_primal_bound = self.parameters.initial_primal_bound

            # Учет направления целевой функции (максимизация или минимизация)
            if m.getObjectiveSense() == "minimize":
                reward += (min(primal_bound, initial_primal_bound) - offset) * time_left
            else:
                reward += (
                    -(max(primal_bound, initial_primal_bound) - offset) * time_left
                )

        return reward


class TimeLimitDualIntegral(ecole.reward.DualIntegral):
    def __init__(self):
        self.parameters = IntegralParameters()
        super().__init__(
            wall=True,
            bound_function=lambda model: (
                self.parameters.offset,
                self.parameters.initial_dual_bound,
            ),
        )

    def set_parameters(
        self, objective_offset=None, initial_primal_bound=None, initial_dual_bound=None
    ):
        self.parameters = IntegralParameters(
            offset=objective_offset,
            initial_primal_bound=initial_primal_bound,
            initial_dual_bound=initial_dual_bound,
        )

    def before_reset(self, model):
        self.parameters.fetch_values(model)
        super().before_reset(model)

    def extract(self, model, done):
        reward = super().extract(model, done)

        # adjust the final reward if the time limit has not been reached
        if done:
            m = model.as_pyscipopt()
            # keep integrating over the time left
            time_left = max(m.getParam("limits/time") - m.getSolvingTime(), 0)
            if m.getStage() < pyscipopt.scip.PY_SCIP_STAGE.TRANSFORMED:
                dual_bound = (
                    -m.infinity()
                    if m.getObjectiveSense() == "minimize"
                    else m.infinity()
                )
            else:
                dual_bound = m.getDualbound()

            offset = self.parameters.offset
            initial_dual_bound = self.parameters.initial_dual_bound

            # account for the model's objective direction (maximization vs minimization)
            if m.getObjectiveSense() == "minimize":
                reward += -(max(dual_bound, initial_dual_bound) - offset) * time_left
            else:
                reward += (min(dual_bound, initial_dual_bound) - offset) * time_left

        return reward


class TimeLimitPrimalDualIntegral(ecole.reward.PrimalDualIntegral):
    def __init__(self):
        self.parameters = IntegralParameters()
        super().__init__(
            wall=True,
            bound_function=lambda model: (
                self.parameters.initial_primal_bound,
                self.parameters.initial_dual_bound,
            ),
        )

    def set_parameters(
        self, objective_offset=None, initial_primal_bound=None, initial_dual_bound=None
    ):
        self.parameters = IntegralParameters(
            offset=objective_offset,
            initial_primal_bound=initial_primal_bound,
            initial_dual_bound=initial_dual_bound,
        )

    def before_reset(self, model):
        self.parameters.fetch_values(model)
        super().before_reset(model)

    def extract(self, model, done):
        reward = super().extract(model, done)

        # adjust the final reward if the time limit has not been reached
        if done:
            m = model.as_pyscipopt()
            # keep integrating over the time left
            time_left = max(m.getParam("limits/time") - m.getSolvingTime(), 0)
            if m.getStage() < pyscipopt.scip.PY_SCIP_STAGE.TRANSFORMED:
                primal_bound = m.getObjlimit()
                dual_bound = (
                    -m.infinity()
                    if m.getObjectiveSense() == "minimize"
                    else m.infinity()
                )
            else:
                primal_bound = m.getPrimalbound()
                dual_bound = m.getDualbound()

            initial_primal_bound = self.parameters.initial_primal_bound
            initial_dual_bound = self.parameters.initial_dual_bound

            # account for the model's objective direction (maximization vs minimization)
            if m.getObjectiveSense() == "minimize":
                reward += (
                    min(primal_bound, initial_primal_bound)
                    - max(dual_bound, initial_dual_bound)
                ) * time_left
            else:
                reward += (
                    -(
                        max(primal_bound, initial_primal_bound)
                        - min(dual_bound, initial_dual_bound)
                    )
                    * time_left
                )

        return reward
