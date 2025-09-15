from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from bert_score import BERTScorer
from swift.plugin.orm import ORM, orms
import os
import re
from typing import TYPE_CHECKING, Dict, List, Union
import copy
import json
import time
import numpy as np


if TYPE_CHECKING:
    from swift.llm import InferRequest



class CXRTrekStage8BERTScoreReward(ORM):

    def compute_bert_metrics(self, labels, input_generations):
        generations = copy.deepcopy(input_generations)

        for i in range(len(generations)):
            if len(generations[i]) == 0:
                generations[i] = "<none>"
            if len(generations[i]) == 1 and generations[i][0] in [':.,']:
                generations[i] = "<none>"

        scorer = BERTScorer(
            model_type="distilroberta-base",
            batch_size=256,
            lang="en",
            rescale_with_baseline=True,
            idf=False,
            idf_sents=None  # 仅当 use_idf=True 时需要提供
        )
        # cands = generations, refs = ground_truth
        P, R, F1 = scorer.score(generations, labels)

        bertscore_f1_list = F1.tolist()
        bertscore_f1_avg = np.mean(bertscore_f1_list)

        return np.round(bertscore_f1_avg, 4), bertscore_f1_list

    def __call__(self, completions, **kwargs) -> List[float]:

        # time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        # import random
        # random_idx = random.randint(0, 10000)
        # with open(f"./debug_bertscore-new-{time_stamp}-{random_idx}.json", "w") as f:
        #     kwargs_dict = dict(kwargs)
        #     json.dump({
        #         "completions": completions,
        #         # "data_dict": kwargs['data_dict'],
        #         'kwargs': {k: str(v) if not isinstance(v, dict) else v for k, v in kwargs_dict.items() if k not in ['trainer_state']},
        #     }, f, indent=4)

        # exit()

        rewards = []
        chat_ids = []
        chat_references = []
        chat_completions = []
        for chat_id, chat_dict in kwargs['trajectory_inputs'].items():
            assert len(chat_dict) == 1, f"chat_dict = {chat_dict}"
            for turn_id, qa in enumerate(chat_dict[0]['data_dict']['cxrtrek_data']['content']):
                # [0: system, 1: user, 2: assistant, 3: user, 4: assistant, ...]
                assert chat_dict[0]['messages'][2 + 2 * turn_id]['role'] == 'assistant', f"chat_dict = {chat_dict}, turn_id = {turn_id}"
                if qa['stage'] != 8:
                    continue
                completion = chat_dict[0]['messages'][2 + 2 * turn_id]['content']
                answer = qa['answer']
                chat_ids.append(chat_id)
                chat_references.append(answer)
                chat_completions.append(completion)
        
        with torch.no_grad():
            bertscore_f1_avg, bertscore_f1_list = self.compute_bert_metrics(
                labels=chat_references,
                input_generations=chat_completions,
            )
            rewards = bertscore_f1_list

        chat_rewards = []
        for chat_id in set(chat_ids):
            mean_reward = [rewards[i] for i in range(len(rewards)) if chat_ids[i] == chat_id]
            assert len(mean_reward) >= 1, f"chat_id {chat_id} has no reward, chat_ids = {chat_ids}"
            chat_rewards.append(float(np.mean(mean_reward)))
        
        return chat_rewards


    # OLD Version Schedulerr 
    # def __call__(self, completions, **kwargs) -> List[float]:

    #     rewards = []

    #     chat_ids = []
    #     is_stage8 = []
    #     references = []

    #     for chat_id, chat_list in kwargs['trajectory_inputs'].items():
    #         for turn in chat_list:
    #             num_turns = turn['rollout_infos']['num_turns']
    #             qas = turn['data_dict']['cxrtrek_data']['content'][num_turns - 1]
    #             references.append(qas['answer'])
    #             chat_ids.append(chat_id)
    #             is_stage8.append(qas['stage'] == 8)

    #     # only compute reward for stage 8
    #     temp_chat_ids = [chat_ids[i] for i in range(len(chat_ids)) if is_stage8[i]]
    #     temp_references = [references[i] for i in range(len(references)) if is_stage8[i]]
    #     temp_completions = [completions[i] for i in range(len(completions)) if is_stage8[i]]

    #     with torch.no_grad():
    #         bertscore_f1_avg, bertscore_f1_list = self.compute_bert_metrics(
    #             labels=temp_references,
    #             input_generations=temp_completions,
    #         )
    #         rewards = bertscore_f1_list


    #     ### try-1: len of rewards = len of completions
    #     completions_rewards = []
    #     for chat_id in chat_ids:
    #         mean_reward = [rewards[i] for i in range(len(rewards)) if temp_chat_ids[i] == chat_id]
    #         assert len(mean_reward) >= 1, f"chat_id {chat_id} has no reward, temp_chat_ids = {temp_chat_ids}"
    #         completions_rewards.append(float(np.mean(mean_reward)))
        
    #     try:
    #         assert len(completions_rewards) == len(completions), f"len(completions_rewards)={len(completions_rewards)} != len(completions)={len(completions)}"
    #     except Exception as e:
    #         time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    #         import random
    #         random_idx = random.randint(0, 10000)
    #         with open(f"./debug_bertscore-{time_stamp}-{random_idx}.json", "w") as f:
    #             kwargs_dict = dict(kwargs)
    #             json.dump({
    #                 "completions": completions,
    #                 # "data_dict": kwargs['data_dict'],
    #                 'kwargs': {k: str(v) if not isinstance(v, dict) else v for k, v in kwargs_dict.items() if k not in ['trainer_state']},
    #             }, f, indent=4)

    #         print(kwargs['data_dict'][0].keys(), len(kwargs['data_dict'][0]))
    #         print("Len of completions:", len(completions))
    #         # print(kwargs['data_dict']['cxrtrek_data'])
    #         print("Completions sample:", completions[:2])
    #         print("Calculating BERTScore rewards...")
    #         print(f"Number of completions: {len(completions)}")
    #         print(f"Number of references: {len(references)}")
    #         print("References sample:", references[:2])
    #         raise e

    #     ### try-2: len of rewards = len of chat [wrong]
    #     # completions_rewards = []
    #     # for chat_id in set(chat_ids):
    #     #     mean_reward = [rewards[i] for i in range(len(rewards)) if temp_chat_ids[i] == chat_id]
    #     #     assert len(mean_reward) >= 1, f"chat_id {chat_id} has no reward, temp_chat_ids = {temp_chat_ids}"
    #     #     completions_rewards.append(float(np.mean(mean_reward)))

    #     return completions_rewards



orms.update({
    'cxrtrek_stage8_bertscore': CXRTrekStage8BERTScoreReward,
})
