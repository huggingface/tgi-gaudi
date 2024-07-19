import re
import torch
import math
import os
import time
from PIL import Image
from io import BytesIO
import base64

from opentelemetry import trace
from typing import Optional, Tuple, List, Type, Dict
from loguru import logger

import text_generation_server.habana_quantization_env as hq_env
from transformers import PreTrainedTokenizerBase, AutoProcessor, AutoTokenizer, AutoConfig
from transformers.image_processing_utils import select_best_resolution
from text_generation_server.pb import generate_pb2
from text_generation_server.models.cache_manager import (
    get_cache_manager,
)
from text_generation_server.models.types import Generation
from text_generation_server.utils import make_tokenizer_optional, pad_next_token_chooser_parameters, HeterogeneousNextTokenChooser
from text_generation_server.utils.debug import dbg_trace
from text_generation_server.utils.speculate import get_speculate
from text_generation_server.utils.tokens import batch_top_tokens

import text_generation_server.habana_quantization_env as hq_env
import habana_frameworks.torch as htorch
from optimum.habana.checkpoint_utils import get_repo_root
from optimum.habana.transformers.generation import MODELS_OPTIMIZED_WITH_STATIC_SHAPES
from optimum.habana.transformers.models import GaudiLlavaNextForConditionalGeneration
from .causal_lm import CausalLM, CausalLMBatch, CausalLMRequest, remove_kv_cache_from_output, round_up, PREFILL_BATCH_BUCKET_SIZE, PAD_SEQUENCE_TO_MULTIPLE_OF

tracer = trace.get_tracer(__name__)

IMAGES = re.compile(r"!\[[^\]]*\]\((.*?)\s*(\"(?:.*[^\"])\")?\s*\)")


def split(string) -> List[Dict[str, str]]:
    parts = []
    cursor = 0
    for pattern in IMAGES.finditer(string):
        start = pattern.start()
        if start != cursor:
            parts.append({"type": "text", "content": string[cursor:start]})

        parts.append({"type": "image", "content": pattern.group(1)})
        cursor = pattern.end()

    if cursor != len(string):
        parts.append({"type": "text", "content": string[cursor:]})

    return parts


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def image_text_replacement(image_input, config, image_id) -> str:
    if config.model_type == "idefics2":
        # TODO technically depends on image splitting which is not implemented.
        num_features = 320
        return (
            "<fake_token_around_image>"
            + "<image>" * num_features
            + "<fake_token_around_image>"
        )
    elif config.model_type == "llava_next":
        height, width = image_input["image_sizes"][image_id]
        num_features = get_number_of_features(height, width, config)
        from loguru import logger

        logger.info(f"Found {num_features} in image of resolution {height}x{width}")
        return "<image>" * num_features
    else:
        raise RuntimeError(f"Unknown config {config.model_type} for multimodal")


def get_unpadded_features(
    height: int, width: int, npatches: int, num_patch_height: int, num_patch_width: int
) -> Tuple[int, int]:
    current_height = npatches * num_patch_height
    current_width = npatches * num_patch_width

    aspect_ratio: float = width / height
    current_aspect_ratio: float = current_width / current_height
    if aspect_ratio > current_aspect_ratio:
        new_height = (height * current_width) // width
        current_height = new_height
    else:
        new_width = (width * current_height) // height
        current_width = new_width

    unpadded_features = current_height * current_width
    newline_features = current_height
    return (unpadded_features, newline_features)


def get_number_of_features(height: int, width: int, config) -> int:
    # From config
    # Hardcoded for CLIP for now
    # image_grid_pinpoints = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
    image_grid_pinpoints = config.image_grid_pinpoints
    image_size = config.vision_config.image_size
    patch_size = config.vision_config.patch_size

    assert image_size % patch_size == 0

    npatches = image_size // patch_size

    num_patch_height, num_patch_width = get_anyres_image_grid_shape(
        [height, width],
        image_grid_pinpoints,
        image_size,
    )
    unpadded_features, newline_features = get_unpadded_features(
        height, width, npatches, num_patch_height, num_patch_width
    )
    # The base patch covers the entire image
    base_features = npatches**2
    return unpadded_features + newline_features + base_features


def load_data_uri(image_uri: str) -> Image.Image:
    image_uri = image_uri.split(",")[-1]
    content = base64.b64decode(image_uri)
    image = Image.open(BytesIO(content))
    return image


class VlmCausalLMBatch(CausalLMBatch):
    pixel_values: Optional[List[torch.Tensor]]
    # pixel_attention_mask: Optional[List[torch.Tensor]]
    image_sizes: Optional[List[Tuple[int, int]]]

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches):
        batch = super(VlmCausalLMBatch, cls).concatenate(batches)
        batch.pixel_values = None
        # batch.pixel_attention_mask = None
        batch.image_sizes = None
        return batch

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]):
        batch = super().filter(request_ids)
        batch.pixel_values = None
        # batch.pixel_attention_mask = None
        batch.image_sizes = None
        return batch

    @classmethod
    def recombine(cls, batches: List["VlmCausalLMBatch"], pad_token_id: int) -> "VlmCausalLMBatch":
        batch = super().recombine(batches, pad_token_id)
        batch.pixel_values = None
        # batch.pixel_attention_mask = None
        batch.image_sizes = None
        return batch

    @classmethod
    def batch_tokenized_inputs(cls, requests, tokenizer, processor, config):
        batch_inputs = []
        image_inputs = []
        max_truncation = 0
        for r in requests:
            chunks = split(r.inputs)
            full_text = ""
            image_id = 0
            for chunk in chunks:
                if chunk["type"] == "text":
                    full_text += chunk["content"]
                elif chunk["type"] == "image":
                    image = chunk["content"]
                    # Should never receive URLs anymore, processing should be done
                    # On the rust layer.
                    # This avoid making n queries per TP
                    # if image.startswith("https://") or image.startswith("http://"):
                    #     image = processor.image_processor.fetch_images(image)
                    if image.startswith("data:"):
                        image = load_data_uri(image)
                    else:
                        raise RuntimeError(
                            "Cannot process input image not starting with data:"
                        )
                    image_input = processor.image_processor(image, return_tensors="pt")
                    full_text += image_text_replacement(image_input, config, image_id)
                    image_inputs.append(image_input)
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk['type']}")

            batch_inputs.append(full_text)
            max_truncation = max(max_truncation, r.truncate)

        batch_tokenized_inputs = tokenizer(
            batch_inputs, truncation=True, max_length=max_truncation
        )["input_ids"]
        if image_inputs:
            image_input = image_inputs[0]
            new_image_inputs = {
                "pixel_values": torch.cat(
                    [img["pixel_values"] for img in image_inputs], dim=0
                ),
            }
            # if "pixel_attention_mask" in image_input:
            #     new_image_inputs["pixel_attention_mask"] = torch.cat(
            #         [img["pixel_attention_mask"] for img in image_inputs], dim=0
            #     )
            if "image_sizes" in image_input:
                new_image_inputs["image_sizes"] = torch.cat(
                    [img["image_sizes"] for img in image_inputs], dim=0
                )
            image_inputs = new_image_inputs
        else:
            image_inputs = None
        return batch_tokenized_inputs, image_inputs

    @classmethod
    def from_pb_processor(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        processor,
        config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "VlmCausalLMBatch":
        # batch_tokenized_inputs, image_inputs = cls.batch_tokenized_inputs(
        #     pb.requests, tokenizer, processor, config
        # )
        # batch = cls.from_tokenized(pb, tokenizer, batch_tokenized_inputs, dtype, device)

        requests = [CausalLMRequest.from_pb(idx, req, tokenizer) for idx, req in enumerate(pb.requests)]

        batch_inputs = []
        image_inputs = []
        for r in requests:
            chunks = split(r.data.inputs)
            full_text = ""
            image_id = 0
            for chunk in chunks:
                if chunk["type"] == "text":
                    full_text += chunk["content"]
                elif chunk["type"] == "image":
                    image = chunk["content"]
                    # Should never receive URLs anymore, processing should be done
                    # On the rust layer.
                    # This avoid making n queries per TP
                    # if image.startswith("https://") or image.startswith("http://"):
                    #     image = processor.image_processor.fetch_images(image)
                    if image.startswith("data:"):
                        image = load_data_uri(image)
                    else:
                        raise RuntimeError(
                            "Cannot process input image not starting with data:"
                        )
                    image_input = processor.image_processor(image, return_tensors="pt")
                    full_text += image_text_replacement(image_input, config, image_id)
                    image_inputs.append(image_input)
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk['type']}")
            batch_inputs.append(full_text)

        max_input_length = max(r.data.truncate for r in requests)
        max_new_tokens = max(r.stopping_criteria.max_new_tokens for r in requests)

        # TODO: Add support for sparse batches
        top_n_tokens = [r.top_n_tokens for r in pb.requests]
        top_n_tokens_tensor = torch.tensor(top_n_tokens, device=device, dtype=torch.int64)

        # TODO: by tokenizing all inputs at once we loose information on actual input lengths
        # this means that we cannot shift inputs to the left after a long input sequence
        # was filtered out
        new_bs = round_up(len(requests), PREFILL_BATCH_BUCKET_SIZE)
        missing_inputs = new_bs - len(requests)
        dummy_inputs = ["?"] * missing_inputs
        parameters = [r.parameters for r in pb.requests]
        # append the dummy parameters for dummy request
        parameters = pad_next_token_chooser_parameters(parameters, new_bs)

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            pb=parameters,
            dtype=dtype,
            device=device,
            tokenizer=tokenizer,
            quantization_enabled=hq_env.is_quantization_enabled,
        )
        tokenized_inputs = tokenizer(
            batch_inputs + dummy_inputs,
            return_tensors="pt",
            padding="longest",
            return_token_type_ids=False,
            truncation=True,
            max_length=max_input_length,
        )

        if image_inputs:
            image_input = image_inputs[0]
            new_image_inputs = {
                "pixel_values": torch.cat(
                    [img["pixel_values"] for img in image_inputs], dim=0
                ),
            }
            # if "pixel_attention_mask" in image_input:
            #     new_image_inputs["pixel_attention_mask"] = torch.cat(
            #         [img["pixel_attention_mask"] for img in image_inputs], dim=0
            #     )
            if "image_sizes" in image_input:
                new_image_inputs["image_sizes"] = torch.cat(
                    [img["image_sizes"] for img in image_inputs], dim=0
                )
            image_inputs = new_image_inputs
        else:
            image_inputs = None

        input_len = tokenized_inputs["input_ids"].shape[1]

        bucket_size = max_input_length
        left_padding = max_input_length - input_len
        if input_len < max_input_length and PAD_SEQUENCE_TO_MULTIPLE_OF != 0:
            assert PAD_SEQUENCE_TO_MULTIPLE_OF <= max_input_length, "PAD_SEQUENCE_TO_MULTIPLE_OF cannot be higher than max_input_length"
            rounded_seq_len = round_up(input_len + 1, PAD_SEQUENCE_TO_MULTIPLE_OF)
            if rounded_seq_len <= max_input_length:
                bucket_size = rounded_seq_len - 1
            else:
                bucket_size = max_input_length - 1
            left_padding = bucket_size - input_len

        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]

        # Allocate space for first token
        input_ids = torch.nn.functional.pad(
            input_ids, (left_padding, 1), value=tokenizer.pad_token_id
        )
        attention_mask = torch.nn.functional.pad(
            attention_mask, (left_padding, 1), value=0
        )
        all_input_ids = torch.nn.functional.pad(
            input_ids, (0, max_new_tokens), value=tokenizer.pad_token_id
        ).T.split(1, dim=1)

        # New input length after left padding
        input_len = bucket_size
        for r in requests:
            r.input_length = input_len
            r.prefix_offset = input_len - 5
            r.read_offset = input_len
            r.all_input_ids = all_input_ids[r.idx]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        htorch.core.mark_step()

        batch = cls(
            batch_id=pb.id,
            requests=requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            merged_kv_cache=False,
            next_token_chooser=next_token_chooser,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            input_length=input_len,
        )

        if image_inputs is not None:
            batch.pixel_values = image_inputs["pixel_values"].to(device=device)
            # if "pixel_attention_mask" in image_inputs:
            #     batch.pixel_attention_mask = image_inputs["pixel_attention_mask"].to(
            #         device=device
            #     )
            # else:
            #     batch.pixel_attention_mask = None
            if "image_sizes" in image_inputs:
                batch.image_sizes = image_inputs["image_sizes"].to(device=device)
            else:
                batch.image_sizes = None
        else:
            batch.pixel_values = None
            # batch.pixel_attention_mask = None
            batch.image_sizes = None

        return batch


class VlmCausalLM(CausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.prev_bs = 0
        if use_medusa:
            raise RuntimeError("Medusa decoding is not enabled for AutoModel")

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        make_tokenizer_optional(tokenizer)

        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

        # Create model
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        dtype = torch.bfloat16 if dtype is None else dtype
        device = torch.device("hpu")

        if hq_env.is_quantization_enabled:
            htorch.core.hpu_set_env()

        if world_size > 1:
            model = self.get_deepspeed_model(
                model_id, dtype, revision
            )
            model = self.prepare_model_for_quantization(model)
        else:
            get_repo_root(model_id)

            # Check support for rope scaling
            model_kwargs = {}
            config = AutoConfig.from_pretrained(
                model_id
            )
            if hasattr(config, "rope_scaling"):
                model_kwargs["rope_scaling"] = self.get_rope_scaling()

            model = GaudiLlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                **model_kwargs
            )
            model.image_offset = 2927
            model.text_tokens_pos = torch.tensor(
                [[   0,    1,    2,    3,    4, 2932, 2933, 2934, 2935, 2936, 2937, 2938,
                2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950,
                2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962,
                2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974,
                2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986,
                2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998,
                2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010,
                3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022,
                3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034,
                3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045]],
                device='hpu:0')
            model = self.prepare_model_for_quantization(model)
            model = model.eval().to(device)

        self.enable_hpu_graph = os.getenv("ENABLE_HPU_GRAPH", "true").lower() == "true"
        self.limit_hpu_graph = os.getenv("LIMIT_HPU_GRAPH", "false").lower() == "true"
        model = remove_kv_cache_from_output(model)
        if self.enable_hpu_graph:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph
            model = wrap_in_hpu_graph(model, disable_tensor_cache=True)
        else:
            if LAZY_MODE == 0:
                # It is said that "keep_input_mutations" is safe for inference to be done
                dbg_trace(
                    "TORCH COMPILE", f'Torch compiling of model')
                model.model = torch.compile(model.model, backend="hpu_backend", options={"keep_input_mutations": True})

        model = self.setup_quantization(model)

        if model.config.model_type not in MODELS_OPTIMIZED_WITH_STATIC_SHAPES:
            raise ValueError(f"Model type {model.config.model_type} is not supported!")

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None:
                tokenizer.pad_token_id = model.config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        kwargs = {
            "use_cache": True,
            "return_dict": True,
        }

        if model.config.model_type in ["llama", "mistral"]:
            kwargs["attn_softmax_bf16"] = True
            kwargs["trim_logits"] = True

            if os.getenv("USE_FLASH_ATTENTION", "false").lower() == "true":
                kwargs["use_flash_attention"] = True
            if os.getenv("FLASH_ATTENTION_RECOMPUTE", "false").lower() == "true":
                kwargs["flash_attention_recompute"] = True

        self.speculate = get_speculate()

        super(CausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
            kwargs=kwargs,
        )

        # Create profiler
        ranks_to_profile = [int(val) for val in os.getenv("PROF_RANKS", "0").split(',')]
        record_shapes = os.getenv("PROF_RECORD_SHAPES", "false").lower() == "true"
        output_dir = os.getenv("PROF_PATH", "/tmp/hpu_profile")
        self.profiling_warmup_steps = int(os.getenv("PROF_WARMUPSTEP", "0")) if rank in ranks_to_profile else 0
        self.profiling_steps = int(os.getenv("PROF_STEP", "0")) if rank in ranks_to_profile else 0
        self.profiling_wait_steps = int(os.getenv("PROF_WAITSTEP", "0"))
        if self.profiling_steps > 0:
            self.hb_profiler = HabanaProfile(
                wait=self.profiling_wait_steps,
                warmup=self.profiling_warmup_steps,
                active=self.profiling_steps,
                output_dir=output_dir,
                record_shapes=record_shapes
            )
            self.hb_profiler.start()
        else:
            self.hb_profiler = None
        self.step = 0

    @property
    def batch_type(self) -> Type[VlmCausalLMBatch]:
        return VlmCausalLMBatch

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        pixel_values,
        # pixel_attention_mask,
        image_sizes,
        token_idx,
        past_key_values: Optional[List[Tuple]] = None,
        bypass_hpu_graph: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Model Forward

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "token_idx": token_idx,
        }

        if self.has_position_ids:
            kwargs["position_ids"] = position_ids

        if bypass_hpu_graph != None:
            kwargs["bypass_hpu_graphs"] = bypass_hpu_graph

        kwargs.update(self.kwargs)

        outputs = self.model.forward(
            pixel_values=pixel_values,
            # pixel_attention_mask=pixel_attention_mask,
            image_sizes=image_sizes,
            **kwargs,
        )

        # if batch.pixel_values is not None:
        #     batch.pixel_values = None
        # if batch.pixel_attention_mask is not None:
        #     batch.pixel_attention_mask = None
        # if batch.image_sizes is not None:
        #     batch.image_sizes = None

        if past_key_values is not None:
            return outputs
        else:
            return outputs.logits, outputs.past_key_values

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batches: List[VlmCausalLMBatch]
    ) -> Tuple[List[Generation], Optional[VlmCausalLMBatch], Tuple[int, int]]:
        start = time.time_ns()
        # Results
        generations: List[Generation] = []
        prev_batches = []
        requests_to_generate = []
        # In order to pipeline any actions on CPU we perform the operation in 3 main stages:
        # Stage 1. Collect next token ids of any previously started generations
        for batch_id, batch in enumerate(batches):
            if batch.logits is not None:
                logits = batch.logits
                past = batch.past
                prefill = batch.past_key_values is None
                if prefill:
                    # no right padding for prefill
                    token_idx_scalar = batch.attention_mask.shape[-1] - 1
                    token_idx = torch.tensor(token_idx_scalar).to(self.device)
                else:
                    token_idx_scalar = batch.attention_mask.shape[-1] - batch.right_padding
                    token_idx = torch.tensor(token_idx_scalar).to(self.device)

                # Select next token
                input_length = batch.input_length
                if logits.shape[-2] > 1:
                    next_token_ids, next_token_logprobs, logprobs, _, _ = batch.next_token_chooser(
                        batch.input_ids, logits[:, input_length - 1: input_length, :].squeeze(-2), self.speculate
                    )
                else:
                    next_token_ids, next_token_logprobs, logprobs, _, _ = batch.next_token_chooser(
                        batch.input_ids, logits.squeeze(-2), self.speculate
                    )
                # Speculation is not active for causal
                accepted_ids = torch.ones_like(batch.input_ids)[:, 0]
                batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
                    batch.top_n_tokens,
                    batch.top_n_tokens_tensor,
                    logprobs,
                    accepted_ids,
                )

                prev_batches.append({
                    'next_token_ids': next_token_ids,
                    'next_token_logprobs': next_token_logprobs,
                })

                for req_idx, req in enumerate(batch.requests):
                    requests_to_generate.append({
                        'req': req,
                        'prev_req_idx': req.idx,
                        'batch_id': batch_id,
                        'seed': batch.next_token_chooser.seeds[req_idx],
                        'do_sample': batch.next_token_chooser.do_sample[req_idx],
                        'top_n_tokens': batch.top_n_tokens[req_idx],
                        'top_token_ids': batch_top_token_ids[req_idx],
                        'top_token_logprobs': batch_top_token_logprobs[req_idx],
                        'grammar_state': batch.next_token_chooser.fsm_grammar_states[req.idx],
                    })

                htorch.core.mark_step()

                # Add new token into input_ids
                batch.input_ids.index_copy_(1, token_idx, next_token_ids.unsqueeze(1))

                # Update attention_mask as we added a new token to input_ids
                batch.attention_mask.index_fill_(1, token_idx, 1)

                # Adjust lengths
                batch.input_length += 1

                # Update position_ids
                if prefill:
                    batch.position_ids = torch.index_select(batch.position_ids, 1, token_idx - 1) + 1
                else:
                    batch.position_ids += 1
                # Update past key values
                if prefill:
                    batch.past_key_values = past

        htorch.core.mark_step()

        # Stage 2. Prepare new batch for speculative scheduling
        if len(batches) > 1:
            batch = self.batch_type.concatenate(batches, self.tokenizer.pad_token_id)
        else:
            batch = batches[0]

        prefill = batch.past_key_values is None

        # Check if we need to do any bookkeeping first
        if not prefill:
            batch = batch.__class__.recombine([batch], self.tokenizer.pad_token_id)

        scenario = 'PREFILL' if prefill else 'GENERATE'
        if self.enable_hpu_graph and batch.batch_size != self.prev_bs:
            self.model.clear_cache()
            self.prev_bs = batch.batch_size
        dbg_trace(
            scenario, f'bs:{batch.batch_size} num_reqs:{len(batch.requests)} seq_len:{batch.seq_length} padding:{batch.right_padding}')
        assert batch.right_padding > 0, 'No more room for next token!'

        # Execute batch
        if prefill:
            # no right padding for prefill
            token_idx = torch.tensor(batch.attention_mask.shape[-1] - 1).to(self.device)
            batch.logits, batch.past = self.forward(
                batch.input_ids,
                batch.attention_mask,
                batch.position_ids,
                batch.pixel_values,
                # batch.pixel_attention_mask,
                batch.image_sizes,
                token_idx,
                batch.past_key_values,
                bypass_hpu_graph=prefill and self.limit_hpu_graph if self.enable_hpu_graph else None,
            )
        else:
            token_idx = torch.tensor(batch.attention_mask.shape[-1] - batch.right_padding).to(self.device)
            input_ids = torch.index_select(batch.input_ids, 1, token_idx - 1)
            batch.logits = self.forward(
                input_ids,
                batch.attention_mask,
                batch.position_ids,
                batch.pixel_values,
                # batch.pixel_attention_mask,
                batch.image_sizes,
                token_idx,
                batch.past_key_values,
                bypass_hpu_graph=prefill and self.limit_hpu_graph if self.enable_hpu_graph else None,
            )

        htorch.core.mark_step()

        if batch.pixel_values is not None:
            batch.pixel_values = None
        # if batch.pixel_attention_mask is not None:
        #     batch.pixel_attention_mask = None
        if batch.image_sizes is not None:
            batch.image_sizes = None

        start_decode = time.time_ns()

        # Stage 3. Finish and return previous generations
        stopped = len(requests_to_generate) > 0
        for prev_batch in prev_batches:
            prev_batch['next_token_logprobs'] = prev_batch['next_token_logprobs'].tolist()
            prev_batch['next_token_ids_cpu'] = prev_batch['next_token_ids'].cpu()
        htorch.core.mark_step()

        for req_data in requests_to_generate:
            req = req_data['req']
            i = req_data['prev_req_idx']
            prev_batch_id = req_data['batch_id']
            assert len(prev_batches) > prev_batch_id
            next_token_ids_cpu = prev_batches[prev_batch_id]['next_token_ids_cpu']
            next_token_logprobs = prev_batches[prev_batch_id]['next_token_logprobs']

            request = req.data
            input_length = req.input_length
            prefix_offset = req.prefix_offset
            read_offset = req.read_offset
            do_sample = req_data['do_sample']
            seed = req_data['seed']
            stopping_criteria = req.stopping_criteria
            all_input_ids = req.all_input_ids
            next_token_id = next_token_ids_cpu[i]
            next_token_logprob = next_token_logprobs[i]
            top_n_tokens = req_data['top_n_tokens']
            top_token_ids = req_data['top_token_ids']
            top_token_logprobs = req_data['top_token_logprobs']
            grammar_state = req_data['grammar_state']

            # Append next token to all tokens
            all_input_ids[input_length] = next_token_id
            new_input_length = input_length + 1

            # Generated token
            if is_tokenizer_transparent(self.tokenizer) and len(stopping_criteria.stop_sequence_criterias) == 0:
                next_token_text = ''
            else:
                next_token_text, prefix_offset, read_offset = self.decode_token(
                    all_input_ids[0:new_input_length, 0], prefix_offset, read_offset
                )

            # Evaluate stopping criteria
            stop, reason = stopping_criteria(
                next_token_id,
                next_token_text,
            )

            if not stop:
                stopped = False

            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    if is_tokenizer_transparent(self.tokenizer):
                        output_text = None
                    else:
                        output_text = self.decode(
                            all_input_ids[new_input_length - stopping_criteria.current_tokens: new_input_length, 0]
                        )
                    generated_text = GeneratedText(
                        output_text,
                        stopping_criteria.current_tokens,
                        reason,
                        seed if do_sample else None,
                    )
                else:
                    generated_text = None

                # Prefill
                if stopping_criteria.current_tokens == 1 and request.prefill_logprobs:
                    # Remove generated token to only have prefill and add nan for first prompt token
                    prefill_logprobs = [float("nan")] + next_token_logprobs
                    prefill_token_ids = all_input_ids[0: new_input_length - 1]
                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    prefill_tokens = Tokens(
                        prefill_token_ids,
                        prefill_logprobs,
                        prefill_texts,
                        is_special=[],
                    )
                else:
                    prefill_tokens = None

                if top_n_tokens > 0:
                    all_top_tokens = []
                    for top_token_ids, top_token_logprobs in zip(
                        top_token_ids, top_token_logprobs
                    ):
                        toptoken_texts = self.tokenizer.batch_decode(
                            top_token_ids,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=False,
                        )
                        special_toptokens = [
                            token_id in self.all_special_ids
                            for token_id in top_token_ids
                        ]
                        top_tokens = Tokens(
                            top_token_ids,
                            top_token_logprobs,
                            toptoken_texts,
                            special_toptokens,
                        )
                        all_top_tokens.append(top_tokens)
                    top_tokens = all_top_tokens
                else:
                    top_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    Tokens(
                        [next_token_id],
                        [next_token_logprob],
                        [next_token_text],
                        [next_token_id in self.all_special_ids],
                    ),
                    generated_text,
                    top_tokens,
                )

                generations.append(generation)

            batch.next_token_chooser = (
                batch.next_token_chooser.advance_grammar_single_with_past_state(
                    req.idx, next_token_id, grammar_state
                )
            )

            req.all_input_ids = all_input_ids
            req.input_length = new_input_length
            req.prefix_offset = prefix_offset
            req.read_offset = read_offset

        htorch.core.mark_step()
        self.step = self.step + 1
        if self.hb_profiler is not None:
            if self.step > self.profiling_wait_steps + self.profiling_warmup_steps + self.profiling_steps:
                self.hb_profiler.stop()
            else:
                self.hb_profiler.step()

        forward_ns = start_decode - start
        decode_ns = time.time_ns() - start_decode
        return generations, batch if not stopped else None, (forward_ns, decode_ns)
