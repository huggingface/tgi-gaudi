# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Mllama model."""

from typing import Optional, Tuple, List, Union

import torch
import torch.utils.checkpoint
from torch import nn
from loguru import logger

from transformers.activations import ACT2FN
from optimum.habana.transformers.models import GaudiMllamaForConditionalGeneration 
from optimum.habana.transformers.models.mllama.modeling_mllama import _prepare_cross_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast

class MllamaForConditionalGeneration(GaudiMllamaForConditionalGeneration):

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        token_idx: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Copied from MllamaForConditionalGeneration::forward: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L2077
        The only differences are:
            - add token_idx input
            - add use_flash_attention and flash_attention_recompute
        """
        full_text_row_masked_out_mask = kwargs.get("full_text_row_masked_out_mask", None)
        logger.info(f"forward++++++++++++++++++++++++")
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            token_idx=token_idx,
            use_flash_attention=use_flash_attention,
            flash_attention_recompute=flash_attention_recompute,
        )
        
        logits = outputs[0]
        logger.info(f"retrun_dict: {return_dict}")
        if not return_dict:
            logger.info(f"type(outputs)={type(outputs)}")
            logger.info(f"outputs.logits.shape={logits.shape}")
            output = (logits,) + outputs[1:]
            return output

        logger.info(f"type(outputs)={type(outputs)}")
        logger.info(f"outputs.logits.shape={outputs.logits.shape}")
        logger.info(f"outputs.past_key_values[0][0].shape={outputs.past_key_values[0][0].shape}")
        logger.info(f"outputs.past_key_values[0][1].shape={outputs.past_key_values[0][1].shape}")
        logger.info(f"outputs.past_key_values[1][0].shape={outputs.past_key_values[1][0].shape}")
        logger.info(f"outputs.past_key_values[1][1].shape={outputs.past_key_values[1][1].shape}")
        logger.info(f"outputs.past_key_values[3][0].shape={outputs.past_key_values[3][0].shape}")
        logger.info(f"outputs.past_key_values[3][1].shape={outputs.past_key_values[3][1].shape}")
        logger.info(f"len(outputs.past_key_values)={len(outputs.past_key_values)}")
        logger.info(f"len(outputs.past_key_values)={len(outputs.past_key_values[0])}")
        logger.info(f"forward-----------------")
        return outputs
    
    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        """
        Copied from MllamaForConditionalGeneration::prepare_inputs_for_generation: https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/mllama/modeling_mllama.py#L2208
        The only differences are:
            - add token_idx handling
            - add bucket_internal handling
            - add use_flash_attention and flash_attention_recompute
        """
        logger.info(f"prepare_inputs_for_generation")
        logger.info(f"input_ids: {input_ids.shape if input_ids is not None else None}")
        logger.info(f"inputs_embeds: {inputs_embeds.shape if inputs_embeds is not None else None}")
        logger.info(f"attention_mask: {attention_mask.shape if attention_mask is not None else None}")
        logger.info(f"position_ids: {position_ids.shape if position_ids is not None else None}")
        logger.info(f"pixel_values: {pixel_values.shape if pixel_values is not None else None}")
        logger.info(f"aspect_ratio_ids: {aspect_ratio_ids.shape if aspect_ratio_ids is not None else None}")
        logger.info(f"aspect_ratio_mask: {aspect_ratio_mask.shape if aspect_ratio_mask is not None else None}")
        logger.info(f"cross_attention_mask: {cross_attention_mask.shape if cross_attention_mask is not None else None}")
        #logger.info(f"past_key_values: {past_key_values}")
        #import pdb; pdb.set_trace()
        #breakpoint()
        
        
        
        
        token_idx = kwargs.get("token_idx", None)
        if token_idx is None:
            return super().prepare_inputs_for_generation(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                cross_attention_mask=cross_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                
                **kwargs,
            )
        else:
            use_flash_attention = kwargs.get("use_flash_attention", False)
            flash_attention_recompute = kwargs.get("flash_attention_recompute", False)
            position_ids = kwargs.get("position_ids", None)
            output_attentions = kwargs.get("output_attentions", None)
            output_hidden_states = kwargs.get("output_hidden_states", None)
            return_dict = kwargs.get("return_dict", None)
            labels = kwargs.get("labels", None)
            cross_attention_states = kwargs.get("cross_attention_states", None)
            
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            bucket_internal = kwargs.get("bucket_internal", None)
            
            if past_key_values is not None:
                if token_idx is not None:
                    input_ids = torch.index_select(input_ids, 1, token_idx - 1)
                elif inputs_embeds is not None:  # Exception 1
                    input_ids = input_ids[:, -cache_position.shape[0] :]
                elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                    input_ids = input_ids[:, cache_position]
            elif bucket_internal and token_idx is not None:
                # for the 1st token we can slice the inputs till token idx for the fwd pass.
                print(f"1111111111111111")
                input_ids = input_ids[:, :token_idx]
                attention_mask = attention_mask[:, :token_idx]
                if cross_attention_mask is not None:
                    cross_attention_mask = cross_attention_mask[:, :token_idx, ...]

            # TODO: we have no attention_mask so this won't work, check if we really won't need attention mask and find another way
            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values:
                    if token_idx is not None:
                        position_ids = torch.index_select(position_ids, 1, token_idx - 1)
                    else:
                        position_ids = position_ids[:, -input_ids.shape[1] :]

                    # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                    position_ids = position_ids.clone(memory_format=torch.contiguous_format)

    
            
            
            if pixel_values is not None and inputs_embeds is not None:
                raise ValueError(
                    "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
                )

            if pixel_values is not None and cross_attention_states is not None:
                raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

            if pixel_values is not None:
                if aspect_ratio_ids is None:
                    raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
                # get vision tokens from vision model
                vision_outputs = self.vision_model(
                    pixel_values=pixel_values,
                    aspect_ratio_ids=aspect_ratio_ids,
                    aspect_ratio_mask=aspect_ratio_mask,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions,
                    return_dict=return_dict,
                    use_flash_attention=use_flash_attention,
                )
                cross_attention_states = vision_outputs[0]
                cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                    -1, cross_attention_states.shape[-2], self.hidden_size
                )

            if cross_attention_mask is not None:
                cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                    cross_attention_mask,
                    num_vision_tokens=self.vision_model.num_patches,
                    dtype=self.dtype,
                    token_idx=token_idx,
                )
            else:
                full_text_row_masked_out_mask = None

            if cross_attention_mask is not None:
                if cache_position is not None:
                    cross_attention_mask = cross_attention_mask[:, :, cache_position]
                    full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]
                elif past_key_values is not None:
                    if token_idx is not None:
                        cross_attention_mask = torch.index_select(cross_attention_mask, -2, token_idx - 1)
                        full_text_row_masked_out_mask = torch.index_select(
                            full_text_row_masked_out_mask, -2, token_idx - 1
                        )
                    else:
                        cross_attention_mask = cross_attention_mask[:, :, -1:]
                        full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, -1:]

            # if past_key_values is not None:
            #     logger.info(f"past_key_values is not None")
            #     seq_len = input_ids.shape[1]
            #     pad_len = seq_len - token_idx.item()
            #     input_ids = torch.index_select(input_ids, 1, token_idx - 1)
            #     # Retrieve the first layer to inspect the logits and mask out the hidden states
            #     # that are set to 0
            #     first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

            #     # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
            #     batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

            #     # Get the target length
            #     past_length = first_layer_past_key_value.shape[-1]
            #     extended_attention_mask = torch.ones(
            #         (attention_mask.shape[0], past_length),
            #         dtype=attention_mask.dtype,
            #         device=attention_mask.device,
            #     )
            #     # Filter out only the tokens that can be un-attended, this can happen
            #     # if one uses Llava + Fused modules where the cache on the
            #     # first iteration is already big enough, or if one passes custom cache
            #     valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
            #     new_batch_index = batch_index[valid_indices]
            #     new_non_attended_tokens = non_attended_tokens[valid_indices]

            #     # Zero-out the places where we don't need to attend
            #     extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

            #     attention_mask = extended_attention_mask
            #     attention_mask[:, -pad_len:] = 0

            # if attention_mask is not None and position_ids is None:
            #     # create position_ids on the fly for batch generation
            #     position_ids = attention_mask.long().cumsum(-1) - 1
            #     position_ids.masked_fill_(attention_mask == 0, 1)
            #     if past_key_values:
            #         if token_idx is not None:
            #             position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            #         else:
            #             position_ids = position_ids[:, -input_ids.shape[1] :]
                
            #     # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
            #     position_ids = position_ids.clone(memory_format=torch.contiguous_format)


            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and past_key_values is None:
                model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
            else:
                # The clone here is for the same reason as for `position_ids`.
                model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

            if num_logits_to_keep is not None:
                model_inputs["num_logits_to_keep"] = num_logits_to_keep


            # keep cache_position implementation as None for HPU
            cache_position = None

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                    "token_idx": token_idx,
                    "labels": labels,
                    "return_dict": kwargs.get("return_dict"),
                    "full_text_row_masked_out_mask": full_text_row_masked_out_mask,
                    "use_flash_attention": use_flash_attention,
                    "cross_attention_mask": cross_attention_mask,
                    "cross_attention_states": cross_attention_states,
                    "output_attentions": output_attentions,
                    "flash_attention_recompute": flash_attention_recompute,
                }
            )

            return model_inputs