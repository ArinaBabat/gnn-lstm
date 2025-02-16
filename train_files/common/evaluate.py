import argparse    # Библиотека для обработки аргументов командной строки
import csv
import json
import pathlib     # Модуль для манипуляции путями файловой системы
import time

import ecole as ec
import numpy as np


class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert)
    or pseudocost scores (weak expert for exploration) when called at every node.
    Этот пользовательский класс функции наблюдения будет случайным образом возвращать либо оценки стронг-бранчинга (дорогостоящего эксперта) 
    либо оценки псевдокост (слабого эксперта для исследования), когда его вызывают на каждом узле
    """

    def __init__(self, expert_probability):     # Инициализация экземпляра класса . Принимает параметр expert_probability, вероятность выбора стронг бренчинга.
        self.expert_probability = expert_probability
        # Создание экземпляров классов Pseudocosts и StrongBranchingScores из библиотеки Ecole
        self.pseudocosts_function = ec.observation.Pseudocosts()"""функция наблюдения за псевдокостами на узлах метода ветвей
                                    и границ. Псевдокост - это дешевое приближение к оценкам стронг-бранчинга. 
                                    Измеряет качество ветвления для каждой переменной. 
                                    Всегда ветвится на переменную с наивысшим псевдокостом. """
        self.strong_branching_function = ec.observation.StrongBranchingScores() """Функция наблюдения за оценками стронг-бранчинга
                                            на узле метода ветвей и границ. Эта функция наблюдения получает оценки для всех переменных
                                            кандидатов для LP или псевдокандидатов на узле метода ветвей и границ. 
                                            Оценка стронг-бранчинга измеряет качество каждой переменной для ветвления (чем выше, тем лучше).
                                            Эта функция наблюдения извлекает массив, содержащий оценки стронг-бранчинга для каждой
                                            переменной в задаче. Переменные упорядочены в соответствии с их позицией в исходной задаче
                                            (SCIPvarGetProbindex), следовательно, к ним можно обращаться по индексам в action_set среды ветвления.
                                            Для переменных, для которых оценка стронг-бранчинга не применима, заполняются значениями NaN."""

    def before_reset(self, model):
        """
        This function will be called at initialization of the environment (before dynamics are reset).
        Эта функция будет вызвана при инициализации среды (до сброса динамики)
        """
        # Вызов методов before_reset для обоих типов наблюдений
        self.pseudocosts_function.before_reset(model) # метод before_reset не делает ничего
                                    # вызывается при инициализации функции наблюдения перед сбросом динамики.
        self.strong_branching_function.before_reset(model) # так же

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        Следует ли возвращать оценки стронг-бранчинга или оценки псевдокост на данном узле?
        """
        probabilities = [1 - self.expert_probability, self.expert_probability] # Вероятности выбора дорогостоящего и дешевого экспертов
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities)) # Выбор эксперта в соответствии с вероятностями
        if expert_chosen: # Возврат оценок выбранного эксперта
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() # Создание объекта парсера аргументов командной строки с помощью библиотеки argparse
                        # предоставляет удобный интерфейс для анализа и обработки аргументов командной строки, переданных при запуске программы
   # Добавление аргументов командной строки. Каждый аргумент определяет, как скрипт должен вести себя в зависимости от переданных параметров
    parser.add_argument("--start", type=int, default=0, help="start")
    parser.add_argument("--end", type=int, default=-1, help="end")
    parser.add_argument("--exp_name", type=str, default="", help="end")
    parser.add_argument(
        "task", help="Task to evaluate.", choices=["primal", "dual", "config"],
    )
    parser.add_argument(
        "problem",
        help="Problem benchmark to process.",
        choices=["item_placement", "load_balancing", "anonymous"],
    )
    parser.add_argument(
        "-t",
        "--timelimit",
        help="Episode time limit (in seconds).",
        default=argparse.SUPPRESS,
        type=float,
    )
    parser.add_argument(
        "-d", "--debug", help="Print debug traces.", action="store_true",
    )
    parser.add_argument(
        "-f",
        "--folder",
        help="Instance folder to evaluate.",
        default="valid",
        type=str,
        choices=("valid", "test", "train"),
    )
    args = parser.parse_args() # Парсинг переданных аргументов командной строки.

    print(f"Evaluating the {args.task} task agent.")

    # collect the instance files
    """
    Определение путей к файлам с инстансами задачи и файлу результатов в зависимости от переданных аргументов. 
    Проверяется параметр args.problem, и на основе его значения выбирается соответствующий путь.
    """
    if args.problem == "item_placement":
        instances_path = pathlib.Path(
            f"../../instances/1_item_placement/{args.folder}/"
        )
        instances_path = pathlib.Path(
            f"/data/ml4co-competition/instances/1_item_placement/{args.folder}/"
        )
        results_file = pathlib.Path(
            f"results/{args.task}/1_item_placement{args.exp_name}{args.start}_{args.end}.csv"
        )
    elif args.problem == "load_balancing":
        instances_path = pathlib.Path(
            f"../../instances/2_load_balancing/{args.folder}/"
        )
        instances_path = pathlib.Path(
            f"/data/ml4co-competition/instances/2_load_balancing/{args.folder}/"
        )
        results_file = pathlib.Path(
            f"results/{args.task}/2_load_balancing{args.exp_name}{args.start}_{args.end}.csv"
        )
    elif args.problem == "anonymous":
        instances_path = pathlib.Path(f"../../instances/3_anonymous/{args.folder}/")
        instances_path = pathlib.Path(
            f"/data/ml4co-competition/instances/3_anonymous/{args.folder}/"
        )
        results_file = pathlib.Path(
            f"results/{args.task}/3_anonymous{args.exp_name}{args.start}_{args.end}.csv"
        )

    print(f"Processing instances from {instances_path.resolve()}")
    instance_files = list(instances_path.glob("*.mps.gz"))[args.start : args.end] # Получение списка файлов с инстансами в указанном диапазоне

    print(f"Saving results to {results_file.resolve()}")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_fieldnames = [    # Определение полей, которые будут записаны в файл результатов.
        "instance",
        "seed",
        "initial_primal_bound",
        "initial_dual_bound",
        "objective_offset",
        "cumulated_reward",
    ]
    with open(results_file, mode="w") as csv_file:    # Открывает CSV-файл results_file в режиме записи
        writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames) #оздает объект csv.DictWriter для записи результатов в файл. Заголовки CSV определены в results_fieldnames
        writer.writeheader()    #Запись в файл

    import sys

    sys.path.insert(1, str(pathlib.Path.cwd())) # Вставляет текущий путь в системный путь, чтобы импортировать модули из текущего рабочего каталога.

    # set up the proper agent, environment and goal for the task Настроить соответствующего агента, среду и цель для задачи
    if args.task == "primal":
        from agents.primal import Policy, ObservationFunction
        from environments import RootPrimalSearch as Environment
        from rewards import TimeLimitPrimalIntegral as BoundIntegral

        time_limit = 5 * 60

    elif args.task == "dual":
        from agents.dual import (
            Policy,
            ObservationFunction,
        )  # agents.dual submissions.random.
        from environments import Branching as Environment  # environments
        from rewards import TimeLimitDualIntegral as BoundIntegral  # rewards

        time_limit = 15 * 60

    elif args.task == "config":
        from agents.config import Policy, ObservationFunction
        from environments import Configuring as Environment
        from rewards import TimeLimitPrimalDualIntegral as BoundIntegral

        time_limit = 15 * 60

    # override from command-line argument if provided Переопределить, если предоставлено через аргумент командной строки
    time_limit = getattr(args, "timelimit", time_limit)

    # Создает объекты агента, функции наблюдения, и целевой функции
    policy = Policy(problem=args.problem)
    observation_function = ObservationFunction(problem=args.problem)
    integral_function = BoundIntegral()

    # Создает объект среды env с установленными параметрами времени и функциями наблюдения и целевой функцией
    env = Environment(  # Environment
        time_limit=time_limit,
        observation_function=observation_function,  # (ExploreThenStrongBranch(expert_probability=1.0),observation_function),              #observation_function, ec.observation.NodeBipartite()
        reward_function=-integral_function,  # negated integral (minimization) Интеграл с отрицательным знаком (минимизация)
    )

    # evaluation loop Цикл оценки
    instance_count = 0
    """Проходится по списку экземпляров задач, представленных файлами, и в каждой итерации устанавливает
    зерно (seed) для агента, среды и функции наблюдения для обеспечения детерминированного поведения"""
    for seed, instance in enumerate(instance_files): 
        instance_count += 1
        # seed both the agent and the environment (deterministic behavior) Установите зерно для агента и среды (детерминированное поведение)
        # Это делается для обеспечения детерминированного поведения
        observation_function.seed(seed) # Устанавливает зерно для генератора случайных чисел, используемого в функции наблюдения
        policy.seed(seed) # Устанавливает зерно для генератора случайных чисел, используемого в стратегии агента
        env.seed(seed) # Устанавливает зерно для генератора случайных чисел в среде

        # read the instance's initial primal and dual bounds from JSON file Считать начальные примитивные и двойственные оценки для экземпляра из файла JSON
        with open(instance.with_name(instance.stem).with_suffix(".json")) as f:    
            instance_info = json.load(f)    # Читает информацию о текущем экземпляре задачи из JSON файла. 
                                        # Эта информация включает начальные прямые и двойственные оценки

        # set up the reward function parameters for that instance Настроить параметры функции вознаграждения для этого экземпляра
        initial_primal_bound = instance_info["primal_bound"] # Извлекает начальную оценку решения задачи оптимизации
        initial_dual_bound = instance_info["dual_bound"] # Извлекает начальную оценку для двойственного решения
        objective_offset = 0     # Устанавливает смещение для целевой функции

        # Устанавливает параметры функции вознаграждения. Используется для интегрирования значений в процессе обучения
        integral_function.set_parameters(
            initial_primal_bound=initial_primal_bound,
            initial_dual_bound=initial_dual_bound,
            objective_offset=objective_offset,
        )

        print()
        print(f"Instance {instance.name}")
        print(f"  seed: {seed}")
        print(f"  initial primal bound: {initial_primal_bound}")
        print(f"  initial dual bound: {initial_dual_bound}")
        print(f"  objective offset: {objective_offset}")

        # reset the environment сброс среды в начальное состояние для текущего экземпляра задачи
        observation, action_set, reward, done, info = env.reset(
            str(instance), objective_limit=initial_primal_bound
        )

        if args.debug:    # проверяет, включен ли режим отладки
            print(f"  info: {info}")
            print(f"  reward: {reward}")
            print(f"  action_set: {action_set}")

        cumulated_reward = 0  # discard initial reward используется для отслеживания накопленного вознаграждения в процессе взаимодействия с средой

        cumulated_rewards = [] # будет использоваться для хранения значений накопленного вознаграждения на каждом временном шаге
        # loop over the environment
        while not done:     # Взаимодействует с средой, выполняя действия согласно стратегии агента. 
                            # Повторяется, пока среда не завершит эпизод.
            action = policy(action_set, observation) # Агент выбирает действие, используя свою стратегию (policy). Принимает текущий 
                                # action_set (набор доступных действий) и observation (наблюдение) для принятия решения о следующем действии.
            # (scores, scores_are_expert), node_observation = observation
            # action = action_set[scores[action_set].argmax()]
            observation, action_set, reward, done, info = env.step(action) # Агент выполняет выбранное действие в среде, 
                            # возвращает новое наблюдение, обновленный action_set, полученное вознаграждение, 
                            # флаг завершения эпизода, и дополнительную информацию о шаге
            if args.debug:
                print(f"  action: {action}")
                print(f"  info: {info}")
                print(f"  reward: {reward}")
                print(f"  action_set: {action_set}")

            cumulated_reward += reward # Обновление накопленного вознаграждения путем добавления текущего вознаграждения

            cumulated_rewards.append(cumulated_reward) # Значение накопленного вознаграждения добавляется в список

        print(f"  cumulated reward (to be maximized): {cumulated_reward}")
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

        np.save(f"/data/load_random_{instance_count}.npy", cumulated_rewards) # Сохраняет массив cumulated_rewards в файл с использованием номера экземпляра задачи в имени файла.
        # print(step_count)
        # save instance results
        with open(results_file, mode="a") as csv_file: #Добавляет результаты текущего экземпляра в CSV-файл с результатами.
            writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
            writer.writerow(
                {
                    "instance": str(instance),    # Имя текущего экземпляра
                    "seed": seed,    # Зерно, используемое для инициализации агента и среды, чтобы обеспечить детерминированное поведение
                    "initial_primal_bound": initial_primal_bound, # Начальная примитивная оценка (bound) для текущего экземпляра
                    "initial_dual_bound": initial_dual_bound, # Начальная двойственная оценка (bound) для текущего экземпляра
                    "objective_offset": objective_offset, # Смещение целевой функции
                    "cumulated_reward": cumulated_reward, # Накопленное вознаграждение для текущего экземпляра
                }
            )
