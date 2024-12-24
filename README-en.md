# DEEPSPEED-FINETUNING-TURIAL

[**🌐English**](./README-en.md) ∙ [**📖简体中文**](./README.md)

This project is a friendly, zero-foundation training template tutorial, which includes a large number of comments in the code to detail how to use DeepSpeed for fine-tuning, as well as how to use DeepSpeed for distributed training. It can quickly achieve fine-tuning for various tasks. The project is based on the text summarization dataset LCSTS: A Large-Scale Chinese Short Text Summarization Dataset, and fine-tunes a large model specifically for Chinese text summarization.
**The file directory is as follows:**

```shell
.
├── README.md
├── checkpoints
├── config
│   └── dsconfig.json
├── data
│   ├── test.jsonl
│   ├── test_public.jsonl
│   ├── train.jsonl
│   └── valid.jsonl
├── fig
│   └── estimate_memory.png
├── logs
├── README-en.md
├── shell
│   ├── run_multi_gpu_freeze.sh
│   ├── run_multi_gpu_full-tuning.sh
│   ├── run_multi_gpu_lora.sh
│   ├── run_single_gpu_freeze.sh
│   ├── run_single_gpu_full-tuning.sh
│   └── run_single_gpu_lora.sh
├── train.py
└── util
    ├── dataprocess.py
    ├── __init__.py
    ├── __pycache__
    │   ├── dataprocess.cpython-311.pyc
    │   ├── __init__.cpython-311.pyc
    │   └── tools.cpython-311.pyc
    └── tools.py
```

# Training Process Step Template

## Estimated Required Memory

Use the function `estimate_zero3_model_states_mem_needs_all_live` , pass in the model, as well as the number of GPUs and nodes, to return a memory size that includes memory sizes of different combinations, assisting us in assessing the required memory and VRAM.
```python
from transformers import AutoModel,AutoTokenizer; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=6, num_nodes=1)
```

![estimate_memory](/fig/estimate_memory.png)

## Dataset process
This step involves defining a dataset class CustomDataset that inherits from torch.utils.data.Dataset, overriding the `__getitem__` and `__len__` methods to return a batch of data. For this, you can refer to the official PyTorch documentation or to an example I've provided.

## Define the model
The model can be defined by PyTorch or by the transformers library. The model we want to fine-tune here is the ChatGLM model. If you wish to train other models such as GPT, you can modify the `--model_name_or_path` parameter. The model used in this project is ChatGLM, hence the path is `--model_name_or_path THUDM/chatglm-6b`. If you are defining your own model, you can change `model=...` in the source code to the model defined by PyTorch code.

## Training model script `train.py`.

### Argument Parsing

define`parse_args`function：

```python
def parse_args():
    parser = argparse.ArgumentParser()
    # Here you can first use `parser.add_argument` to add some additional parameters, such as learning rate, epochs, batch size, etc.

	...

	# Here you can import a config file into a variable using the json module and pass that variable to the `config` parameter of `deepspeed.initialize`; alternatively, you can pass the `deepspeed_config.json` through the shell command line with `deepspeed --deepspeed --deepspeed_config deepspeed_config.json`, in which case you do not need to pass the variable to the `config` parameter of `deepspeed.initialize`.

    parser.add_argument("--ds_config", type=str, default="deepspeed_config.json", help="deepspeed_config路径")
    
    # Define some variables that DeepSpeed will automatically modify, such as `local_rank` and `global_rank`. `local_rank` is the GPU index within a single node, while `global_rank` is the global GPU index across multiple nodes.
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank")
    
    # Add DeepSpeed arguments.
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()
```

### deepspeed.init_distributed

It needs to be used before initializing `deepspeed.initialize`, and its function is similar to `torch.distributed.init_process_group(...)`.

In fact, the main code of DeepSpeed is somewhat similar to `torch.distributed`.

```python
args = parse_args()

if args.local_rank == -1:
    device = torch.device("cuda")
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed()
args.global_rank = torch.distributed.get_rank()
```

### deepspeed.initialize

`deepspeed.initialize` usage method:

- Return parameters include: [engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler]

- General input parameters:

  - model: Required, the model to be passed in

  - optimizer: Optional: defined by torch or config.json

  - model_parameters: Optional: the parameters of the model

  - training_data: Optional, dataset type

  - lr_scheduler: Optional: defined by torch or config.json

  - config: Optional: content of the config file. If you don't want to pass the config parameter, you can use the method in the shell command line with `deepspeed --deepspeed --deepspeed_config deepspeed_config.json`, which eliminates the need to pass the config parameter to deepspeed.initialize. Here, the parameter config is passed because it allows for more flexible changes to parameters, such as gradient_accumulation_steps, etc.

- Example usage: `model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, args=args, config=ds_config)`

### for loop

```python
model_engine.train()
for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()
```

Here, the scheduler and optimizer will be automatically called. If `gradient_accumulation_steps` is specified, an update will only occur after every `gradient_accumulation_steps`. The input and output parameters of the `model_engine` are exactly the same as the original model. After obtaining the loss, you only need to call `model_engine.backward(loss)` and `model_engine.step()`, without the need to call `model.zero_grad()` and `optimizer.step()` and other operations.

## Define deepspeed_config.json

Refer to this document for the meaning of each parameter:https://www.deepspeed.ai/docs/config-json/#batch-size-related-parameters. You can define it according to your needs.

Template for ZeRO3 distributed training config JSON file:

```json
{
  // Here, train_micro_batch_size_per_gpu and gradient_accumulation_steps are defined, and train_batch_size is ignored because train_batch_size = train_micro_batch_size_per_gpu * number of GPUs * gradient_accumulation_steps
  "gradient_accumulation_steps": 32,
  "train_micro_batch_size_per_gpu": 4,
  // How many steps to print information once, default is 10
  "steps_per_print": 1,
  // ZeRO optimization, stage 3 is chosen here, offloading to CPU memory
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu"
    },
    "offload_optimizer": {
      "device": "cpu"
    },
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "bf16": {
    "enabled": false
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 100
  },
  // Clip gradients greater than 1
  "gradient_clipping": 1.0,
  // Below, the optimizer and learning rate scheduler can be defined in PyTorch and passed to deepspeed.initialize
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      ...
    }
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      ...
    }
  },
  // This parameter allows you to use a custom torch.optim, but if you want to use the official one, just remove the following line
  "zero_force_ds_cpu_optimizer": false
}
```

Here, in order to achieve the desired learning rate curve, custom optimizer and scheduler are used, but the official recommendation is to use the optimizer and scheduler defined by them.

```
[rank0]: deepspeed.runtime.zero.utils.ZeRORuntimeException: You are using ZeRO-Offload with a client provided optimizer (<class 'torch.optim.adamw.AdamW'>) which in most cases will yield poor performance. Please either use deepspeed.ops.adam.DeepSpeedCPUAdam or set an optimizer in your ds-config (https://www.deepspeed.ai/docs/config-json/#optimizer-parameters). If you really want to use a custom optimizer w. ZeRO-Offload and understand the performance impacts you can also set <"zero_force_ds_cpu_optimizer": false> in your configuration file.
```

## Run shell command.

Here, it is divided into single GPU and multi-GPU training. The specific meanings of the parameters can be found in the `parse_args` function of `train.py`, which includes detailed comments.
### Single GPU training.

The official recommendation is to use the `--include localhost:...` parameter to specify GPUs, rather than through `CUDA_VISIBLE_DEVICES=...`. Additionally, you can specify the port for running with `--master_port=<port>`. If you don't add it, the system will assign a default port. Regarding `--offload_device nvme`, it's a choice between NVMe or CPU for offloading. If the CPU memory is insufficient, choose NVMe.

#### LoRA fine tuning

run `sh ./shell/run_single_gpu_lora.sh`

```shell
#!/bin/bash

deepspeed --include localhost:0 --master_port 12345 train.py \
          --model_name_or_path THUDM/chatglm3-6b \
          --src_len 512 \
          --tgt_len 128 \
          --data_path ./data/test.jsonl \
          --train_micro_batch_size_per_gpu 1\
          --gradient_accumulation_steps 16 \
          --max_lr 1e-4 \
          --initial_lr 1e-6 \
          --min_lr 1e-8 \
          --weight_decay 0.01 \
          --adam_beta1 0.9 \
          --adam_beta2 0.999\
          --epochs 1 \
          --output_dir ./checkpoints/ \
          --finetune_method lora \
          --ds_config_path ./config/dsconfig.json \
          --lora_alpha 32 \
          --lora_dropout 0.05 \
          --lora_target_modules query_key_value \
          --lora_r 8 \
          --offload_device cpu \
          --nvme_path ./mnt/nvme
```
#### Freeze Partial Parameters for Fine-tuning

run `sh ./shell/run_single_gpu_freeze.sh`

```shell
#!/bin/bash

deepspeed --include localhost:0 --master_port 12345 train.py \
          --model_name_or_path THUDM/chatglm3-6b \
          --src_len 512 \
          --tgt_len 128 \
          --data_path ./data/test.jsonl \
          --train_micro_batch_size_per_gpu 1\
          --gradient_accumulation_steps 16 \
          --max_lr 1e-4 \
          --initial_lr 1e-6 \
          --min_lr 1e-8 \
          --weight_decay 0.01 \
          --adam_beta1 0.9 \
          --adam_beta2 0.999\
          --epochs 1 \
          --output_dir ./checkpoints \
          --finetune_method freeze \
          --freeze_modules dense_h_to_4h \
          --ds_config_path ./config/dsconfig.json \
          --offload_device cpu \
          --nvme_path ./mnt/nvme
```

#### Full fine tuning

run `sh ./shell/run_single_gpu_full-tuning.sh`

```shell
#!/bin/bash

deepspeed --include localhost:0 --master_port 12345 train.py \
          --model_name_or_path THUDM/chatglm3-6b \
          --src_len 512 \
          --tgt_len 128 \
          --data_path ./data/test.jsonl \
          --train_micro_batch_size_per_gpu 1\
          --gradient_accumulation_steps 16 \
          --max_lr 1e-4 \
          --initial_lr 1e-6 \
          --min_lr 1e-8 \
          --weight_decay 0.01 \
          --adam_beta1 0.9 \
          --adam_beta2 0.999\
          --epochs 1 \
          --output_dir ./checkpoints \
          --finetune_method full-tuning \
          --ds_config_path ./config/dsconfig.json \
          --offload_device cpu \
          --nvme_path ./mnt/nvme
```

### Multi-GPU training

#### LoRA Fine-tuning

run `sh ./shell/run_multi_gpu_lora.sh`

```shell
#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6 --master_port 12345 train.py \
          --model_name_or_path THUDM/chatglm3-6b \
          --src_len 512 \
          --tgt_len 128 \
          --data_path ./data/test.jsonl \
          --train_micro_batch_size_per_gpu 1\
          --gradient_accumulation_steps 16 \
          --max_lr 1e-4 \
          --initial_lr 1e-6 \
          --min_lr 1e-8 \
          --weight_decay 0.01 \
          --adam_beta1 0.9 \
          --adam_beta2 0.999\
          --epochs 1 \
          --output_dir ./checkpoints/ \
          --finetune_method lora \
          --ds_config_path ./config/dsconfig.json \
          --lora_alpha 32 \
          --lora_dropout 0.05 \
          --lora_target_modules query_key_value \
          --lora_r 8 \
          --offload_device cpu \
          --nvme_path ./mnt/nvme
```

#### Freeze Partial Parameters for Fine-tuning

run `sh ./shell/run_multi_gpu_freeze.sh`

```shell
#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6 --master_port 12345 train.py \
          --model_name_or_path THUDM/chatglm3-6b \
          --src_len 512 \
          --tgt_len 128 \
          --data_path ./data/test.jsonl \
          --train_micro_batch_size_per_gpu 1\
          --gradient_accumulation_steps 16 \
          --max_lr 1e-4 \
          --initial_lr 1e-6 \
          --min_lr 1e-8 \
          --weight_decay 0.01 \
          --adam_beta1 0.9 \
          --adam_beta2 0.999\
          --epochs 1 \
          --output_dir ./checkpoints \
          --finetune_method freeze \
          --freeze_modules dense_h_to_4h \
          --ds_config_path ./config/dsconfig.json \
          --offload_device cpu \
          --nvme_path ./mnt/nvme
```

#### Full fine tuning

run `sh ./shell/run_multi_gpu_full-tuning.sh`

```shell
#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6 --master_port 12345 train.py \
          --model_name_or_path THUDM/chatglm3-6b \
          --src_len 512 \
          --tgt_len 128 \
          --data_path ./data/test.jsonl \
          --train_micro_batch_size_per_gpu 1\
          --gradient_accumulation_steps 16 \
          --max_lr 1e-4 \
          --initial_lr 1e-6 \
          --min_lr 1e-8 \
          --weight_decay 0.01 \
          --adam_beta1 0.9 \
          --adam_beta2 0.999\
          --epochs 1 \
          --output_dir ./checkpoints \
          --finetune_method full-tuning \
          --ds_config_path ./config/dsconfig.json \
          --offload_device cpu \
          --nvme_path ./mnt/nvme
```

# Common Issues You May Encounter

1. RuntimeError: Unable to JIT load the async_io op due to it not being compatible due to hardware/software issue. None

When encountering the `RuntimeError: Unable to JIT load the async_io op due to it not being compatible due to hardware/software issue. None` error, it means that DeepSpeed's `async_io` operation cannot be loaded via Just-In-Time (JIT) compilation on the current system due to hardware or software compatibility issues. This problem is related to the compatibility of the `libaio` library. To resolve this issue, you need to ensure that the `libaio-dev` library is installed on your system, as it is required by the `async_io` operation. On Ubuntu systems, you can install `libaio-dev` using the following command:

```bash
sudo apt install libaio-dev
```

2. AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'

Set the environment variable before the shell command: `export DS_SKIP_CUDA_CHECK=1`. The solution comes from a blogger.[deepspeed使用zero3 + offload报错:AttributeError: ‘DeepSpeedCPUAdam‘ object has no attribute ‘ds_opt_adam_deepspeed zero3 offload-CSDN博客](https://blog.csdn.net/qq_44193969/article/details/137051032)

