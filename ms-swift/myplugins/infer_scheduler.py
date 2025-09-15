import asyncio
from abc import ABC
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from swift.plugin import ContextManager, Env, context_managers, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.utils import remove_response
import time

if TYPE_CHECKING:
    from swift.llm.infer.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, RequestConfig,
                                          RolloutOutput)
    from swift.llm.template import RolloutInferRequest
    from swift.llm.infer.infer_engine import GRPOVllmEngine
    from swift.llm.utils import Messages


class CXRTrekScheduler_NoThink_OnlyRewardStage8(MultiTurnScheduler):
    """
    Scheduler for multi-turn reasoning with Thinking class models.

    Key Features:
    1. No thinking parsing, only the final answer is needed.
    2. Each round's conversation history is processed independently.
    3. Returns a list of RolloutOutput objects, one for each round.
    4. Please set `--loss_scale all` for training last round response.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        log_file = "./cxrtrek_scheduler_nothink_onlyrewardstage8-" + time_stamp + ".log"
        self.log_fh = open(log_file, "a")
        print(f"Logging to {log_file}")

    async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
                  **kwargs) -> List['RolloutOutput']:
        """
        Execute multi-turn inference for Thinking models.

        Args:
            infer_request (RolloutInferRequest): The initial inference request containing the conversation history.
            request_config (RequestConfig): Configuration for the inference request.
            **kwargs: Additional arguments for the inference engine.

        Returns:
            List[RolloutOutput]: A list of RolloutOutput objects, one for each reasoning round.
        """

        # rollout loop
        from swift.llm.infer.protocol import RolloutOutput
        current_request = infer_request
        current_turn = 1
        rollout_infos = {}
        total_response_ids = []
        total_response_loss_mask = []

        # prepare messages for rollout
        current_request.messages = current_request.messages[:2]
        assert current_request.messages[0]['role'] == 'system' and current_request.messages[1]['role'] == 'user', \
            f"Initial messages should contain system and user messages only, got: {current_request.messages}"

        while True:
            messages = current_request.messages
            if current_turn == 1 or not messages[-1]['content']:
                # If it's the first turn or the last message content is empty(dummy), remove the response
                remove_response(messages)

            # Get model response
            response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(
                current_request, request_config, **kwargs)
            response_choice: 'ChatCompletionResponseChoice' = response.choices[0]

            # Update conversation history
            completion = response_choice.message.content
            is_continuation = False
            if messages[-1]['role'] == 'assistant':
                messages[-1]['content'] += completion
                is_continuation = True
            else:
                messages.append({'role': 'assistant', 'content': completion})

            # Check stopping conditions
            should_stop = self.check_finished(current_request, response_choice, current_turn)

            # double-check if user forget to judge the max_turns
            if self.max_turns:
                should_stop = should_stop or (current_turn >= self.max_turns)

            # Prepare next turn
            ret = self.step(current_request, response_choice, current_turn)
            current_request: 'RolloutInferRequest' = ret['infer_request']

            # Track response tokens and masks
            return_token_id = False
            if 'response_token_ids' in ret:
                if is_continuation and total_response_ids:
                    total_response_ids[-1].extend(ret['response_token_ids'])
                else:
                    total_response_ids.append(ret['response_token_ids'])
                return_token_id = True

            if 'response_loss_mask' in ret:
                assert return_token_id, 'You must return response_token_ids if you want to return response_loss_mask'
                assert len(ret['response_loss_mask']) == len(ret['response_token_ids']), \
                    'response_loss_mask must have the same length as response_token_ids'
                if is_continuation and total_response_loss_mask:
                    total_response_loss_mask[-1].extend(ret['response_loss_mask'])
                else:
                    total_response_loss_mask.append(ret['response_loss_mask'])

            if 'rollout_infos' in ret:
                # Always overwrite the rollout info for this step.
                # If you need to keep all step-wise details, switch to append or merge instead.
                rollout_infos.update(ret['rollout_infos'])

            if should_stop:
                if is_continuation and total_response_ids:
                    # for continuation and total_response_ids is not empty
                    # we need to extend the last turn's response_token_ids and response_loss_mask
                    total_response_ids[-1].extend(response_choice.token_ids)
                    if total_response_loss_mask:
                        total_response_loss_mask[-1].extend([1] * len(response_choice.token_ids))

                return RolloutOutput(
                    response=response,
                    messages=messages,
                    response_token_ids=total_response_ids,
                    response_loss_mask=total_response_loss_mask,
                    rollout_infos={
                        **rollout_infos, 'num_turns': current_turn
                    },
                )

            if current_request.messages[-1]['role'] == 'assistant':
                # Add a dummy response to allow engine to continue generating
                current_request.messages.append({'role': 'assistant', 'content': None})

            current_turn += 1

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:
        data_dict = infer_request.data_dict.get("data_dict", {}).get("cxrtrek_data", {})
        assert data_dict, f"data_dict is empty, please check your dataset, infer_request = {infer_request}"

        if current_turn >= len(data_dict['content']):
            return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        data_dict = infer_request.data_dict.get("data_dict", {}).get("cxrtrek_data", {})
        assert data_dict, f"data_dict is empty, please check your dataset, infer_request = {infer_request}"

        if current_turn < len(data_dict['content']):
            question = data_dict['content'][current_turn]['question']  # add new question
            infer_request.messages.append({'role': 'user', 'content': question})

        stage_of_turn = data_dict['content'][current_turn - 1]['stage']
        token_ids = response_choice.token_ids
        if stage_of_turn == 8:
            loss_mask = [1] * len(token_ids)
        else:
            loss_mask = [0] * len(token_ids)

        return {
            'infer_request': infer_request,
            'response_token_ids': token_ids,
            'response_loss_mask': loss_mask,
        }



    # async def run(self, infer_request: 'RolloutInferRequest', request_config: 'RequestConfig',
    #               **kwargs) -> List['RolloutOutput']:
    #     """
    #     Execute multi-turn inference for Thinking models.

    #     Args:
    #         infer_request (RolloutInferRequest): The initial inference request containing the conversation history.
    #         request_config (RequestConfig): Configuration for the inference request.
    #         **kwargs: Additional arguments for the inference engine.

    #     Returns:
    #         List[RolloutOutput]: A list of RolloutOutput objects, one for each reasoning round.
    #     """
    #     from swift.llm.infer.protocol import RolloutOutput

    #     current_request = infer_request
    #     current_turn = 1
    #     rollout_outputs = []

    #     # prepare messages for rollout
    #     print("current_request.messages = ", current_request.messages)
    #     current_request.messages = current_request.messages[:2]
    #     assert current_request.messages[0]['role'] == 'system' and current_request.messages[1]['role'] == 'user', \
    #         f"Initial messages should contain system and user messages only, got: {current_request.messages}"

    #     while True:
    #         messages = current_request.messages
    #         # Obtain model response for the current turn
    #         response: 'ChatCompletionResponse' = await self.infer_engine.infer_async(
    #             current_request, request_config, **kwargs)
    #         response_choice: 'ChatCompletionResponseChoice' = response.choices[0]
    #         completion = response_choice.message.content

    #         self.log_fh.write(f"[Turn {current_turn} question]: {messages[-1]['content']}\n")
    #         self.log_fh.write(f"[Turn {current_turn} Response]: {completion}\n")
    #         # flush
    #         self.log_fh.flush()

    #         # Append the assistant's response to the message history
    #         messages.append({'role': 'assistant', 'content': completion})

    #         # Create a RolloutOutput for the current round
    #         round_output = RolloutOutput(
    #             response=response,
    #             messages=messages,
    #             response_token_ids=response_choice.token_ids,
    #             rollout_infos={'num_turns': current_turn})
    #         # Store the output for this round
    #         rollout_outputs.append(round_output)

    #         # Determine whether to stop the multi-turn reasoning
    #         should_stop = self.check_finished(current_request, response_choice, current_turn)

    #         if should_stop:
    #             break

    #         # Prepare for the next turn by updating the inference request
    #         ret = self.step(current_request, response_choice, current_turn)
    #         current_request: 'RolloutInferRequest' = ret['infer_request']
    #         current_turn += 1

    #     return rollout_outputs

    

multi_turns.update({
    'cxrtrek_scheduler_nothink_onlyrewardstage8': CXRTrekScheduler_NoThink_OnlyRewardStage8
})
