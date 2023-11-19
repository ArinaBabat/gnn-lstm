import argparse    # для обработки аргументов командной строки
import csv    # для работы с CSV-файлами
import json
import pathlib    # для работы с путями к файлам и директориям
import time    
import datetime    
import ecole    
import numpy as np    
import sys    
import pandas as pd
import gzip    # для работы с файлами в формате Gzip (сжатие данных)
import pickle    # для сериализации и десериализации объектов Python

sys.path.insert(1, '/'.join(str(pathlib.Path.cwd()).split('/')[0:-1])) # добавляет второй элемент в sys.path, 
                                        # который является текущей рабочей директорией за исключением последнего элемента
                                        # чтобы иметь доступ к модулям из предыдущего каталога
parser = argparse.ArgumentParser() # создает парсер аргументов командной строки
parser.add_argument(    # Программе разрешено принимать аргумент -problem, 
    "-problem",        # который выбирает задачу для обработки из предопределенного набора
    help="Problem benchmark to process.",
    choices=["policy2_64_0", "policy2_64_1", "policy2_64_2","policy2_64_3", \
    "policy2_128_0", "policy2_128_1", "policy2_128_2","policy2_128_3", \
    "policy2_256_0", "policy2_256_1", "policy2_256_2","policy2_256_3", \
    "policy3_64_0", "policy3_64_1", "policy3_64_2","policy3_64_3", \
    "policy3_128_0", "policy3_128_1", "policy3_128_2","policy3_128_3", \
    "policy3_256_0", "policy3_256_1", "policy3_256_2","policy3_256_3", \
    ],
)
args = parser.parse_args() # парсит аргументы командной строки и сохраняет их в объект args
                # Теперь можно использовать args.problem для доступа к значению аргумента -problem

from dual import (
    Policy,
    ObservationFunction,
)  # agents.dual submissions.random.

from environments import Branching as Environment  # environments
from rewards import TimeLimitDualIntegral as BoundIntegral  # rewards

inst = "/content/explore_nuri/Nuri/instances/train_test/mas76.mps.gz" # устанавливает переменную inst для указания пути к файлу с задачей для обработки
out_dir = "/content/gdrive/MyDrive/evaluate_data"    # переменная для указания директории, в которой будут сохранены результаты

print(f"instance:{inst}")    # информация о текущем экземпляре задачи

time_limit = 15 * 60

strbr = ecole.observation.StrongBranchingScores()    # Создает объект для измерения стронг бранчинга. Описание из ecole в evaluate.py
      
policy = Policy(problem=args.problem) # Создает объект Policy для обработки задачи в зависимости от выбранного аргумента

env = ecole.environment.Branching(observation_function=ObservationFunction(problem=args.problem)) #Создает среду Branching с функцией наблюдения, соответствующей выбранной задаче

observation, action_set, reward, done, info = env.reset(inst) # Инициализирует среду, сбрасывает ее в начальное состояние, возвращает начальные значения
correct_predictions = 0
total_predictions = 0
rand_accuracy = 0

sum_metric1 = 0 # policy_num / strbr_num
sum_metric2 = 0 # policy_score / strbr_score
sum_rand_metric1 = 0
sum_rand_metric2 = 0

sum_err = 0
sum_rand_err = 0
total_action_set = 0
sum_acc = 0

acc_list = {            # словарь, в котором будут храниться массивы для каждой метрики
  'metric1':np.array([]),
  'exp_metric1':np.array([]),
  'metric2':np.array([]),
  'exp_metric2':np.array([]),
  'err':np.array([]),
  'exp_err':np.array([]),
  'rand_metric1':np.array([]),
  'rand_metric2':np.array([]),
  'rand_err':np.array([]),

}

while not done:
    if total_predictions == 1000:
      break
    policy_action = policy(action_set, observation)
    strbr_scores = strbr.extract(env.model, done)
    strbr_action = action_set[strbr_scores[action_set].argmax()]
    rand_actions_id = np.random.randint(0, len(action_set), 10)
    rand_actions = [action_set[rand_action_id] for rand_action_id in rand_actions_id] 

    if policy_action.item() == strbr_action:
        correct_predictions += 1
    total_predictions += 1
    total_action_set += len(action_set)
    exp_rough_accuracy =  total_predictions / total_action_set
    
    policy_action_id = np.where(action_set == policy_action.item())[0][0]
    policy_score = strbr_scores[action_set][policy_action_id]
    rand_scores = [strbr_scores[action_set][rand_action_id] for rand_action_id in rand_actions_id] 
    sorted_strbr_scores = sorted(strbr_scores[action_set])
    policy_score_top = np.where(sorted_strbr_scores == policy_score)[0][0]
    rand_scores_top = [np.where(sorted_strbr_scores == rand_score)[0][0] for rand_score in rand_scores]
   
    
    acc_list['metric1'] = np.append(acc_list['metric1'], policy_score_top / len(action_set))
    acc_list['exp_metric1'] = np.append(acc_list['exp_metric1'], 0.5 * (len(action_set) + 1) / len(action_set))
    acc_list['metric2'] = np.append(acc_list['metric2'], policy_score / sorted_strbr_scores[-1])
    acc_list['exp_metric2'] = np.append(acc_list['exp_metric2'], sum(sorted_strbr_scores) / (len(sorted_strbr_scores) * sorted_strbr_scores[-1]))
    acc_list['err'] = np.append(acc_list['err'], sorted_strbr_scores[-1] - policy_score)
    acc_list['exp_err'] = np.append(acc_list['exp_err'], sorted_strbr_scores[-1] - sum(sorted_strbr_scores) / len(sorted_strbr_scores))

    acc_list['rand_metric1'] =np.append(acc_list['rand_metric1'], rand_scores_top[0] / len(action_set))
    acc_list['rand_metric2'] =np.append(acc_list['rand_metric2'], rand_scores[0] / sorted_strbr_scores[-1])
    acc_list['rand_err'] = np.append(acc_list['rand_err'], sorted_strbr_scores[-1] - rand_scores[0])
    '''
    if total_predictions <=1:
        acc_list['rand_metric1'] =np.array([rand_score_top / len(action_set) for rand_score_top in rand_scores_top])
        acc_list['rand_metric2'] =np.array([rand_score / sorted_strbr_scores[-1] for rand_score in rand_scores])
        acc_list['rand_err'] = np.array([sorted_strbr_scores[-1] - rand_score for rand_score in rand_scores])
    else:
        acc_list['rand_metric1'] = np.array([*acc_list['rand_metric1'], np.array([rand_score_top / len(action_set) for rand_score_top in rand_scores_top])])
        acc_list['rand_metric2'] = np.array([*acc_list['rand_metric2'], np.array([rand_score / sorted_strbr_scores[-1] for rand_score in rand_scores])])
        acc_list['rand_err'] = np.array([*acc_list['rand_err'], np.array([sorted_strbr_scores[-1] - rand_score for rand_score in rand_scores])])
    '''
    sum_metric1 += policy_score_top / len(action_set)
    sum_metric2 += policy_score / sorted_strbr_scores[-1]

    sum_rand_metric1 += 0.5 * (len(action_set) + 1) / len(action_set)
    sum_rand_metric2 += sum(sorted_strbr_scores) / (len(sorted_strbr_scores) * sorted_strbr_scores[-1])

    sum_err +=(sorted_strbr_scores[-1] - policy_score)
    sum_rand_err += (sorted_strbr_scores[-1] - sum(sorted_strbr_scores) / len(sorted_strbr_scores))
    print("======================================")
    print(f"iteration: {total_predictions}")
    print(f"len = {len(action_set)}, action_set: {action_set}")
    print(f"rand_actions: {rand_actions}")
    print(f"rand_actions_id: {rand_actions_id}")
    print(f"rand_scores: {rand_scores}")
    print(f"rand_scores_top: {rand_scores_top}")

    print(f"unsorted sb scores: {strbr_scores[action_set]}")
    print(f"sorted sb scores: {sorted_strbr_scores}")
    print(f"policy_score: {policy_score}")
    print(f"strbr_score: {sorted_strbr_scores[-1]}")
    print(f"policy_score_top {policy_score_top}")

    print(f"policy_action: {policy_action}")
    print(f"strbr_action: {strbr_action}")
    print(f"action_set: {action_set}")

    print(f"current rough accuracy: {correct_predictions/total_predictions}")
    print(f"expected rough accuracy:  {exp_rough_accuracy}")

    print(f"metric1: {sum_metric1 / total_predictions}")
    print(f"rand_metric1: {sum_rand_metric1 / total_predictions}")

    print(f"metric2: {sum_metric2 / total_predictions}")
    print(f"rand_metric2: {sum_rand_metric2 / total_predictions}")

    print(f"sum_err: {sum_err/total_predictions}")
    print(f"rand_sum_err: {sum_rand_err / total_predictions}")
    observation, action_set, reward, done, info = env.step(strbr_action)

print(f"total rough accuracy {correct_predictions/total_predictions}")
print(f"total random rought accuracy:  {exp_rough_accuracy}")

print(f"total metric1: {sum_metric1 / total_predictions}")
print(f"total expected metric1: {sum_rand_metric1 / total_predictions}")

print(f"total metric2: {sum_metric2 / total_predictions}")
print(f"total expected metric2: {sum_rand_metric2 / total_predictions}")

print(f"total error: {sum_err/total_predictions}")
print(f"total expected error: {sum_rand_err / total_predictions}")

date_name = '_'.join(str(datetime.datetime.now()).split())
inst_name = inst.split('/')[-1].split('.')[0]
fileout = f"{out_dir}/{inst_name}_{date_name}.pkl"
print(f"acc_list[rand_metric1]: {acc_list['rand_metric1']}")

acc_list['sum_metric1'] = (sum_metric1 / total_predictions, sum_rand_metric1 / total_predictions)
acc_list['sum_metric2'] = (sum_metric2 / total_predictions, sum_rand_metric2 / total_predictions)
acc_list['sum_err'] = (sum_err / total_predictions, sum_rand_err / total_predictions)


with gzip.open(fileout, 'wb') as f:
    pickle.dump(acc_list, f)
print(f"saved in {fileout}!")
#with open(f"{out_dir}/{inst_name}_{date_name}.json", "w") as file:
#    json.dump(acc_list, file)
