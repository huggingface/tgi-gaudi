<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Text Generation Inference on Habana Gaudi

## Table of contents

- [Text Generation Inference on Habana Gaudi](#text-generation-inference-on-habana-gaudi)
  - [Table of contents](#table-of-contents)
  - [Running TGI on Gaudi](#running-tgi-on-gaudi)
  - [Running TGI with FP8 Precision](#running-tgi-with-fp8-precision)
  - [Adjusting TGI Parameters](#adjusting-tgi-parameters)
  - [Environment variables](#environment-variables)
  - [Profiler](#profiler)
  - [Supported Configurations](#supported-configurations)

## Running TGI on Gaudi

To use [🤗 text-generation-inference](https://github.com/huggingface/text-generation-inference) on Habana Gaudi/Gaudi2/Gaudi3, follow these steps:

1. Pull the official Docker image with:
   ```bash
   docker pull ghcr.io/huggingface/tgi-gaudi:2.0.4
   ```
> [!NOTE]
> Alternatively, you can build the Docker image using the `Dockerfile` located in this folder with:
> ```bash
> docker build -t tgi_gaudi .
> ```
2. Launch a local server instance:

    i. On 1 Gaudi card
   ```bash
   model=meta-llama/Llama-2-7b-hf
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e ENABLE_HPU_GRAPH=true -e LIMIT_HPU_GRAPH=true -e USE_FLASH_ATTENTION=true -e FLASH_ATTENTION_RECOMPUTE=true --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tgi-gaudi:2.0.4 --model-id $model --max-input-tokens 1024 --max-total-tokens 2048
   ```
   > For gated models such as [StarCoder](https://huggingface.co/bigcode/starcoder), you will have to pass `-e HUGGING_FACE_HUB_TOKEN=<token>` to the `docker run` command above with a valid Hugging Face Hub read token.

    ii. On 1 Gaudi card using PyTorch eager mode with torch compile:
   ```bash
   model=meta-llama/Llama-2-7b-hf
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e PT_HPU_LAZY_MODE=0 -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tgi-gaudi:2.0.4 --model-id $model --max-input-tokens 1024 --max-total-tokens 2048
   ```

    iii. On 8 Gaudi cards:
   ```bash
   model=meta-llama/Llama-2-70b-hf
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

   docker run -p 8080:80 -v $volume:/data --runtime=habana -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none  -e ENABLE_HPU_GRAPH=true -e LIMIT_HPU_GRAPH=true -e USE_FLASH_ATTENTION=true -e FLASH_ATTENTION_RECOMPUTE=true --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tgi-gaudi:2.0.4 --model-id $model --sharded true --num-shard 8 --max-input-tokens 1024 --max-total-tokens 2048
   ```
3. You can then send a simple request:
   ```bash
   curl 127.0.0.1:8080/generate \
     -X POST \
     -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":32}}' \
     -H 'Content-Type: application/json'
   ```
4. To run static batching benchmark, please refer to [TGI's benchmark tool](https://github.com/huggingface/text-generation-inference/tree/main/benchmark).

   To run it on the same machine, you can do the following:
   * `docker exec -it <docker name> bash` , pick the docker started from step 2 using docker ps
   * `text-generation-benchmark -t <model-id>` , pass the model-id from docker run command
   * after the completion of tests, hit ctrl+c to see the performance data summary.

5. To run continuous batching benchmark, please refer to [README in examples folder](https://github.com/huggingface/tgi-gaudi/blob/habana-main/examples/README.md).


## Running TGI with FP8 Precision

TGI-Gaudi supports FP8 precision inference with INC (Intel Neural Compressor) and HQT (Habana Quantization Toolkit). FP8 inference can be run by setting QUANT_CONFIG environment variable in the docker command. From TGI-GAUDI 2.0.4 release, INC is used by default for quantization. HQT will be removed in future releases. To use HQT, disable INC by setting `-e USE_INC=0` in docker command.

To run FP8 Inference:

1. Measure statistics by using [Optimum Habana measurement script](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation#running-with-fp8:~:text=use_deepspeed%20%2D%2Dworld_size%208-,run_lm_eval.py,-%5C%0A%2Do%20acc_70b_bs1_measure.txt)
2. Run requested model in TGI with proper QUANT_CONFIG setting - e.g. `-e QUANT_CONFIG=./quantization_config/maxabs_quant.json`.

> [!NOTE]
> Only models pointed in [Supported Configurations](#supported-configurations) are guaranteed to work with FP8

Additional hints to quantize model for TGI when using `run_lm_eval.py`:
* use `--limit_hpu_graphs` flag to save memory
* try to model your use case situation by adjusting `--batch_size` , `--max_new_tokens 512` and `--max_input_tokens 512`; in case of memory issues, lower those values
* use dataset/tasks suitable for your use case (see `--help` for defining tasks/datasets)

## Adjusting TGI parameters

Maximum sequence length is controlled by two arguments:
- `--max-input-tokens` is the maximum possible input prompt length. Default value is `4095`.
- `--max-total-tokens` is the maximum possible total length of the sequence (input and output). Default value is `4096`.

Maximum batch size is controlled by two arguments:
- For prefill operation, please set `--max-batch-prefill-tokens` as `bs * max-input-tokens`, where `bs` is your expected maximum prefill batch size.
- For decode operation, please set `--max-batch-total-tokens` as `bs * max-total-tokens`, where `bs` is your expected maximum decode batch size.
- Please note that batch size will be always padded to the nearest multiplication of `BATCH_BUCKET_SIZE` and `PREFILL_BATCH_BUCKET_SIZE`.

To ensure greatest performance results, at the beginning of each server run, warmup is performed. It's designed to cover major recompilations while using HPU Graphs. It creates queries with all possible input shapes, based on provided parameters (described in this section) and runs basic TGI operations on them (prefill, decode, concatenate).

Except those already mentioned, there are other parameters that need to be properly adjusted to improve performance or memory usage:

- `PAD_SEQUENCE_TO_MULTIPLE_OF` determines sizes of input length buckets. Since warmup creates several graphs for each bucket, it's important to adjust that value proportionally to input sequence length. Otherwise, some out of memory issues can be observed.
- `ENABLE_HPU_GRAPH` enables HPU graphs usage, which is crucial for performance results. Recommended value to keep is `true` .

For more information and documentation about Text Generation Inference, checkout [the README](https://github.com/huggingface/text-generation-inference#text-generation-inference) of the original repo.


## Environment variables

<div align="left">

| Name                        | Value(s)   | Default          | Description                                                                                                                      | Usage                        |
| --------------------------- | :--------- | :--------------- | :------------------------------------------------------------------------------------------------------------------------------- | :--------------------------- |
| ENABLE_HPU_GRAPH            | True/False | True             | Enable hpu graph or not                                                                                                          | add -e in docker run command |
| LIMIT_HPU_GRAPH             | True/False | False            | Skip HPU graph usage for prefill to save memory, set to `True` for large sequence/decoding lengths(e.g. 300/212)                 | add -e in docker run command |
| BATCH_BUCKET_SIZE           | integer    | 8                | Batch size for decode operation will be rounded to the nearest multiple of this number. This limits the number of cached graphs  | add -e in docker run command |
| PREFILL_BATCH_BUCKET_SIZE   | integer    | 4                | Batch size for prefill operation will be rounded to the nearest multiple of this number. This limits the number of cached graphs | add -e in docker run command |
| PAD_SEQUENCE_TO_MULTIPLE_OF | integer    | 128              | For prefill operation, sequences will be padded to a multiple of provided value.                                                 | add -e in docker run command |
| SKIP_TOKENIZER_IN_TGI       | True/False | False            | Skip tokenizer for input/output processing                                                                                       | add -e in docker run command |
| WARMUP_ENABLED              | True/False | True             | Enable warmup during server initialization to recompile all graphs. This can increase TGI setup time.                            | add -e in docker run command |
| QUEUE_THRESHOLD_MS          | integer    | 120              | Controls the threshold beyond which the request are considered overdue and handled with priority. Shorter requests are prioritized otherwise.                            | add -e in docker run command |
| USE_FLASH_ATTENTION         | True/False | False            | Whether to enable Habana Flash Attention, provided that the model supports it. Currently only llama and mistral supports this feature. Please refer to https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_PyTorch_Models.html?highlight=fusedsdpa#using-fused-scaled-dot-product-attention-fusedsdpa |
| FLASH_ATTENTION_RECOMPUTE   | True/False | False            | Whether to enable Habana Flash Attention in recompute mode on first token generation. |

</div>

## Profiler

To collect performance profiling, please set below environment variables:

<div align="left">

| Name               | Value(s)   | Default          | Description                                              | Usage                        |
| ------------------ | :--------- | :--------------- | :------------------------------------------------------- | :--------------------------- |
| PROF_WAITSTEP      | integer    | 0                | Control profile wait steps                               | add -e in docker run command |
| PROF_WARMUPSTEP    | integer    | 0                | Control profile warmup steps                             | add -e in docker run command |
| PROF_STEP          | integer    | 0                | Enable/disable profile, control profile active steps     | add -e in docker run command |
| PROF_PATH          | string     | /tmp/hpu_profile | Define profile folder                                    | add -e in docker run command |
| PROF_RANKS         | string     | 0                | Comma-separated list of ranks to profile                 | add -e in docker run command |
| PROF_RECORD_SHAPES | True/False | False            | Control record_shapes option in the profiler             | add -e in docker run command |
</div>


> The license to use TGI on Habana Gaudi is the one of TGI: https://github.com/huggingface/text-generation-inference/blob/main/LICENSE
>
> Please reach out to api-enterprise@huggingface.co if you have any question.

## Supported Configurations

Not all features of TGI are currently supported as this is still a work in progress.
Currently supported and validated configurations (other configurations are not guaranteed to work or ensure reasonable performance):

| Model                                          | BF16 | FP8 | Single Card | Multi-Cards |
|------------------------------------------------|------|-----|-------------|-------------|
| [Llama2-7B](#llama2-7b)                        | ✔    | ✔   | ✔           |             |
| [Llama2-70B](#llama2-70b)                      | ✔    | ✔   |             | ✔           |
| [Llama3-8B](#llama3-8b)                        | ✔    | ✔   | ✔           |             |
| [Llama3-70B](#llama3-70b)                      | ✔    | ✔   |             | ✔           |
| [Llama3.1-8B](#llama31-8b)                     | ✔    | ✔   | ✔           |             |
| [Llama3.1-70B](#llama31-70b)                   | ✔    | ✔   | ✔           | ✔           |
| CodeLlama-13B                                  | ✔    | ✔   | ✔           |             |
| Mixtral-8x7B                                   | ✔    | ✔   | ✔           | ✔           |
| Mistral-7B                                     | ✔    | ✔   | ✔           | ✔           |
| [Llava-v1.6-Mistral-7B](#llava-v16-mistral-7b) | ✔    | ✔   | ✔           |             |


### Llama2-7B

### BF16

```bash
model=meta-llama/Llama-2-7b-chat-hf
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e LIMIT_HPU_GRAPH=true \
   -e MAX_TOTAL_TOKENS=2048 \
   -e PREFILL_BATCH_BUCKET_SIZE=2 \
   -e BATCH_BUCKET_SIZE=32 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=256 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --model-id $model \
   --max-input-length 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 2048 --max-batch-total-tokens 65536 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 64
```

#### FP8

```bash
model=meta-llama/Llama-2-7b-chat-hf
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -v $PWD/quantization_config:/usr/src/quantization_config \
   -v $PWD/hqt_output:/usr/src/hqt_output \
   -e QUANT_CONFIG=./quantization_config/maxabs_quant.json \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e LIMIT_HPU_GRAPH=true \
   -e MAX_TOTAL_TOKENS=2048 \
   -e PREFILL_BATCH_BUCKET_SIZE=2 \
   -e BATCH_BUCKET_SIZE=32 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=256 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --model-id $model \
   --max-input-length 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 2048 --max-batch-total-tokens 65536 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 64
```

### Llama2-70B

#### BF16

```bash
model=meta-llama/Llama-2-70b-chat-hf
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e LIMIT_HPU_GRAPH=true \
   -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
   -e MAX_TOTAL_TOKENS=2048 \
   -e BATCH_BUCKET_SIZE=256 \
   -e PREFILL_BATCH_BUCKET_SIZE=4 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=64 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --sharded true --num-shard 8 \
   --max-input-length 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 4096 --max-batch-total-tokens 524288 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 512
```

#### FP8

```bash
model=meta-llama/Llama-2-70b-chat-hf
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -v $PWD/quantization_config:/usr/src/quantization_config \
   -v $PWD/hqt_output:/usr/src/hqt_output \
   -e QUANT_CONFIG=./quantization_config/maxabs_quant.json \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e LIMIT_HPU_GRAPH=true \
   -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
   -e MAX_TOTAL_TOKENS=2048 \
   -e BATCH_BUCKET_SIZE=256 \
   -e PREFILL_BATCH_BUCKET_SIZE=4 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=64 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --sharded true --num-shard 8 \
   --max-input-length 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 4096 --max-batch-total-tokens 524288 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 512
```

### Llama3-8B
### BF16

```bash
model=meta-llama/Meta-Llama-3-8B-Instruct
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e LIMIT_HPU_GRAPH=true \
   -e MAX_TOTAL_TOKENS=2048 \
   -e PREFILL_BATCH_BUCKET_SIZE=2 \
   -e BATCH_BUCKET_SIZE=32 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=256 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --model-id $model \
   --max-input-length 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 2048 --max-batch-total-tokens 65536 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 64
```

#### FP8

```bash
model=meta-llama/Meta-Llama-3-8B-Instruct
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -v $PWD/quantization_config:/usr/src/quantization_config \
   -v $PWD/hqt_output:/usr/src/hqt_output \
   -e QUANT_CONFIG=./quantization_config/maxabs_quant.json \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e LIMIT_HPU_GRAPH=true \
   -e MAX_TOTAL_TOKENS=2048 \
   -e PREFILL_BATCH_BUCKET_SIZE=2 \
   -e BATCH_BUCKET_SIZE=32 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=256 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --model-id $model \
   --max-input-length 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 2048 --max-batch-total-tokens 65536 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 64
```

### Llama3-70B

#### BF16

```bash
model=meta-llama/Meta-Llama-3-70B-Instruct
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e LIMIT_HPU_GRAPH=true \
   -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
   -e MAX_TOTAL_TOKENS=2048 \
   -e BATCH_BUCKET_SIZE=256 \
   -e PREFILL_BATCH_BUCKET_SIZE=4 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=64 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --sharded true --num-shard 8 \
   --max-input-length 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 4096 --max-batch-total-tokens 524288 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 512
```

#### FP8

```bash
model=meta-llama/Meta-Llama-3-70B-Instruct
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -v $PWD/quantization_config:/usr/src/quantization_config \
   -v $PWD/hqt_output:/usr/src/hqt_output \
   -e QUANT_CONFIG=./quantization_config/maxabs_quant.json \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e LIMIT_HPU_GRAPH=true \
   -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
   -e MAX_TOTAL_TOKENS=2048 \
   -e BATCH_BUCKET_SIZE=256 \
   -e PREFILL_BATCH_BUCKET_SIZE=4 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=64 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --sharded true --num-shard 8 \
   --max-input-length 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 4096 --max-batch-total-tokens 524288 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 512
```

### Llama3.1-8B
### BF16

```bash
model=meta-llama/Meta-Llama-3.1-8B-Instruct
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e LIMIT_HPU_GRAPH=true \
   -e MAX_TOTAL_TOKENS=2048 \
   -e PREFILL_BATCH_BUCKET_SIZE=2 \
   -e BATCH_BUCKET_SIZE=32 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=256 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --model-id $model \
   --max-input-length 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 2048 --max-batch-total-tokens 65536 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 64
```

#### FP8

```bash
model=meta-llama/Meta-Llama-3.1-8B-Instruct
hf_token=YOUR_ACCESS_TOKEN
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -v $PWD/quantization_config:/usr/src/quantization_config \
   -v $PWD/hqt_output:/usr/src/hqt_output \
   -e QUANT_CONFIG=./quantization_config/maxabs_quant.json \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e LIMIT_HPU_GRAPH=true \
   -e MAX_TOTAL_TOKENS=2048 \
   -e PREFILL_BATCH_BUCKET_SIZE=2 \
   -e BATCH_BUCKET_SIZE=32 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=256 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --model-id $model \
   --max-input-length 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 2048 --max-batch-total-tokens 65536 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 64
```

### Llama3.1-70B

#### BF16

```bash
model=meta-llama/Meta-Llama-3.1-70B-Instruct
hf_token=YOUR_ACCESS_TOKEN   # Llama2 is a gated model and requires a special access token
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e LIMIT_HPU_GRAPH=true \
   -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
   -e MAX_TOTAL_TOKENS=2048 \
   -e BATCH_BUCKET_SIZE=256 \
   -e PREFILL_BATCH_BUCKET_SIZE=4 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=64 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --sharded true --num-shard 8 \
   --max-input-length 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 4096 --max-batch-total-tokens 524288 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 512
```

#### FP8

```bash
model=meta-llama/Meta-Llama-3.1-70B-Instruct
hf_token=YOUR_ACCESS_TOKEN   # Llama2 is a gated model and requires a special access token
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -v $PWD/quantization_config:/usr/src/quantization_config \
   -v $PWD/hqt_output:/usr/src/hqt_output \
   -e QUANT_CONFIG=./quantization_config/maxabs_quant.json \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e TEXT_GENERATION_SERVER_IGNORE_EOS_TOKEN=true \
   -e LIMIT_HPU_GRAPH=true \
   -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
   -e MAX_TOTAL_TOKENS=2048 \
   -e BATCH_BUCKET_SIZE=256 \
   -e PREFILL_BATCH_BUCKET_SIZE=4 \
   -e PAD_SEQUENCE_TO_MULTIPLE_OF=64 \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --sharded true --num-shard 8 \
   --max-input-length 1024 --max-total-tokens 2048 \
   --max-batch-prefill-tokens 4096 --max-batch-total-tokens 524288 \
   --max-waiting-tokens 7 --waiting-served-ratio 1.2 --max-concurrent-requests 512
```

### Llava-v1.6-Mistral-7B

An image usually accounts for 2000 input tokens. For example, an image of size 512x512 is represented by 2800 tokens. Thus, `max-input-tokens` must be larger than the number of tokens associated to the image. Otherwise the image may be truncated. We set `BASE_IMAGE_TOKENS=2048` as the default image token number. This is the minimum value of `max-input-tokens`. You can override the environment variable `BASE_IMAGE_TOKENS` to change this value. The warmup will generate graphs with input length from `BASE_IMAGE_TOKENS` to `max-input-tokens`. For LLava-next 7B, the value of `max-batch-prefill-tokens` is 16384, which is calcualted as follows: `prefill_batch_size` = `max-batch-prefill-tokens` / `max-input-tokens`.

Multi-card Llava-next inference is currently not supported.

#### BF16

```bash
model=llava-hf/llava-v1.6-mistral-7b-hf
hf_token=YOUR_ACCESS_TOKEN   # HF access token
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e HF_HUB_ENABLE_HF_TRANSFER=1 \
   -e HUGGING_FACE_HUB_TOKEN=$hf_token \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
    -e PREFILL_BATCH_BUCKET_SIZE=1 \
    -e BATCH_BUCKET_SIZE=1 \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --model-id $model \
   --max-input-tokens 4096 --max-batch-prefill-tokens 16384 \
   --max-total-tokens 8192 --max-batch-total-tokens 32768
```

Send the simple request.
```bash
curl -N 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png)What is this a picture of?\n\n","parameters":{"max_new_tokens":16, "seed": 42}}' \
    -H 'Content-Type: application/json'
```


#### FP8

```bash
model=llava-hf/llava-v1.6-mistral-7b-hf
hf_token=YOUR_ACCESS_TOKEN   # HF access token
volume=$PWD/data   # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 \
   --runtime=habana \
   -v $volume:/data \
   -v $PWD/quantization_config:/usr/src/quantization_config \
   -v $PWD/hqt_output:/usr/src/hqt_output \
   -e QUANT_CONFIG=./quantization_config/maxabs_quant.json \
   -e HABANA_VISIBLE_DEVICES=all \
   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
   -e HF_HUB_ENABLE_HF_TRANSFER=1 \
   -e HUGGING_FACE_HUB_TOKEN=$hf_token \
   -e ENABLE_HPU_GRAPH=true \
   -e LIMIT_HPU_GRAPH=true \
   -e USE_FLASH_ATTENTION=true \
   -e FLASH_ATTENTION_RECOMPUTE=true \
    -e PREFILL_BATCH_BUCKET_SIZE=1 \
    -e BATCH_BUCKET_SIZE=1 \
   --cap-add=sys_nice \
   --ipc=host \
   ghcr.io/huggingface/tgi-gaudi:2.0.4 \
   --model-id $model \
   --max-input-tokens 4096 --max-batch-prefill-tokens 16384 \
   --max-total-tokens 8192 --max-batch-total-tokens 32768
```

Please note that the model warmup can take several minutes, especially for FP8 configs. To minimize this time in consecutive runs, please refer to [Disk Caching Eviction Policy](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_PyTorch_Models.html#disk-caching-eviction-policy).

Other sequence lengths can be used with proportionally decreased/increased batch size (the higher sequence length, the lower batch size).
Support for other models from Optimum Habana will be added successively.