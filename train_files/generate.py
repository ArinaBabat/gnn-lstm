import os    # взаимодействие с ос
import sys    # функции и переменные, взаимодействующие с интерпретатором Python
import glob    # для поиска файлов с использованием шаблонов (при удалении файлов с расширением .pkl)
import gzip    # для работы с gzip-сжатыми файлами
import json
import ecole
import queue    # очередь FIFO
import pickle    # для сериализации и десериализации объектов Python
import shutil    # утилиты для работы с файлами и директориями
import argparse    # 
import threading    # предоставляет многозадачность
import numpy as np
from pathlib import Path    # 
from datetime import datetime
import time
import collections    # специализированные контейнеры данных
import multiprocessing    # для параллельных вычислений
import random
import zipfile    # позволяет создавать, читать и обновлять архивы ZIP

# import environment
# добавление пути к модулю Branching в пакете common.environments
sys.path.append('../..')
from common.environments import Branching as Environment


def myprint(*args, **kwargs):
    print(f'[{datetime.now()}]', end=' ')
    print(*args, **kwargs)


def archive(instance, out_dir):
    inst_name = str(instance).split('/')[-1].split('.')[0]
    zip_dir = f"{out_dir}/{inst_name}"
    zipfile_name = f"{zip_dir}/{inst_name}.zip"
    out_zip = zipfile.ZipFile(zipfile_name, 'a')

    for folder, subfolders, files in os.walk(zip_dir):
        for file in files:
            if file.endswith('.pkl'):
                out_zip.write(os.path.join(folder, file), os.path.relpath(os.path.join(folder, file), zip_dir),
                              compress_type=zipfile.ZIP_DEFLATED)
    out_zip.close()
    sample_to_del = glob.glob(f"{zip_dir}/*.pkl")
    for f in sample_to_del:
        os.remove(f)
    myprint(f"archived in {zipfile_name}")


class ExploreThenStrongBranch:
    """
    Custom observation function.
    Queries the expert with a given probability. Returns variable scores given
    by the expert (if queried), or pseudocost scores otherwise.

    Parameters
    ----------
    expert_probability : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    """

    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        """
        Reset internal data at the start of episodes.
        Called before environment dynamics are reset.

        Parameters
        ----------
        model : ecole.scip.Model
            Model defining the current state of the solver.
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        """
        Extract the observation on the given state.

        Parameters
        ----------
        model : ecole.scip.Model
            Model defining the current state of the solver.
        done : bool
            Flag indicating if the state is terminal.

        Returns
        -------
        scores : np.ndarray
            Variable scores.
        scores_are_expert : bool
            Flag indicating whether scores are given by the expert.
        """
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            myprint("strong branchin chosen")
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)


def collect_samples(instance, out_dir, query_expert_prob, time_limit, rng):
    myprint(f"instances = {instance}")
    inst_name = str(instance).split('/')[-1].split('.')[0]
    os.makedirs(f"{out_dir}/{inst_name}")

    seed = rng.randint(2 ** 32)
    episode = 0
    observation_function = {'scores': ExploreThenStrongBranch(expert_probability=query_expert_prob),
                            'node_observation': ecole.observation.NodeBipartite()}
    scip_parameters = {
        "limits/memory": 2048,
        "limits/time": time_limit,
    }
    env = Environment(
        time_limit=time_limit,
        observation_function=observation_function,
        scip_params=scip_parameters,
    )
    try:
        env.seed(seed)
        observation, action_set, _, done, _ = env.reset(str(instance), objective_limit=None)
    except Exception as e:
        myprint("error")
        done = True
        with open("error_log.txt", "a") as f:
            f.write(f"Error occurred solving {instance} with seed {seed}\n")
            f.write(f"{e}\n")
    while not done:
        scores, scores_are_expert = observation["scores"]
        node_observation = observation["node_observation"]
        node_observation = (node_observation.row_features,
                            (node_observation.edge_features.indices,
                             node_observation.edge_features.values),
                            node_observation.column_features)

        action = action_set[scores[action_set].argmax()]

        if scores_are_expert:
            data = [node_observation, action, action_set, scores]
            filename = f'{out_dir}/{inst_name}/sample_{episode}.pkl'
            with gzip.open(filename, 'wb') as f:
                pickle.dump({
                    'episode': episode,
                    'instance': instance,
                    'seed': seed,
                    'data': data,
                }, f)
            myprint(f"[{inst_name}] written sample {episode}")
            episode += 1
            # myprint(dict_sample_count)
            # myprint(f"instance == {str(instance)}")
            # myprint(f"dict[{str(instance)}]:{dict_sample_count[str(instance)]}")
            # myprint(f"[instance: {str(instance)}]  samples : {dict_sample_count[str(instance)]}")

        #time_start_step = time.time()
        try:
            observation, action_set, _, done, _ = env.step(action)
            # myprint(f"[{inst_name}] done: {done}")
        except Exception as e:
            with open("error_log.txt", "a") as f:
                f.write(f"Error occurred solving {instance} with seed {seed}\n")
                f.write(f"{e}\n")
        isSolved = env.model.is_solved
        # myprint(f"[{inst_name}] solved: {isSolved}")

        if isSolved:
            with open(f"{out_dir}/{inst_name}/{inst_name}_status.txt", "w") as file:
                file.write(f"solved: {isSolved} samples: {episode}")
            myprint(f"[{inst_name}] solved: {isSolved}")
            break
        #time_end_step = time.time()
        #step_time = time_end_step - time_start_step
        if done:
            myprint(f"[{inst_name}] done")


if __name__ == '__main__':

    # set dir
    instances_train = glob.glob(f'/content/explore_nuri/Nuri/instances/train0/instances/3_anonymous/valid/*.mps.gz')
    rand = random.random()
    myprint(f"instances_train {instances_train}")
    myprint(rand)
    out_dir = f'/content/gdrive/MyDrive/samples_ml4co/experiment/num_{rand}' 
    os.makedirs(out_dir)

    # parameters
    node_record_prob = 1  # probability of running the expert strategy and collecting samples.
    time_limit = 60*60*2  # time limit for solving each instance
    train_size = len(instances_train)

    #dict_instance_info = {}
    #for instance_curr in instances_train:
    #    dict_instance_info.update({instance_curr: [False, 0]})
    #myprint(dict_instance_info)

    myprint(f"{len(instances_train)} train instances for {train_size} samples")
    myprint(f"time limit: {time_limit / 3600} hours")
    rng = np.random.RandomState(100)

    for instance in instances_train:
        p = multiprocessing.Process(target=collect_samples,
                                    args=(instance, out_dir, node_record_prob, time_limit, rng))
        p.start()
        myprint(f"[{instance}] after start")
        p.join(time_limit)
        myprint(f"[{instance}] after join")
        # If procces alive, kill him
        if p.is_alive():
            myprint(f"kill {instance}")
            # Terminate
            p.terminate()
        archive(instance, out_dir)
