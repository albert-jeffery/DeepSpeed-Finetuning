# DEEPSPEED-FINETUNING-TURIAL
[**🌐English**](./README-en.md) ∙ [**📖简体中文**](./README.md)

这个项目是一个友好的、零基础训练模板教程，代码中包含大量注释，详细介绍如何使用deepspeed进行finetuning，以及如何使用deepspeed进行分布式训练，介绍如何使用deepspeed进行模型训练。可以快速实现多种任务的微调和针对不同的模型，根据需求进行修改argparse参数或者源代码即可。该项目以文本总结数据集LCSTS: A Large-Scale Chinese Short Text Summarization Dataset为基础，微调一个针对中文文本总结的大模型。

**文件目录如下：**

```shell
.
├── 读我.md
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
├── readme.md
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

# 训练流程步骤模板

## 预估所需的内存

使用estimate_zero3_model_states_mem_needs_all_live函数，传入模型，以及gpu数量和节点数量，返回一个内存大小，这个内存大小包括不同组合的内存大小，辅助我们评估所需要的内存和显存。
```python
from transformers import AutoModel,AutoTokenizer; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=6, num_nodes=1)
```

![estimate_memory](/fig/estimate_memory.png)

## 处理数据集
这一步包括定义一个数据集类CustomDataset，继承torch.utils.data.Dataset，重写`__getitem__`和`__len__`方法，返回一个batch的数据，这里可以参考pytorch官方文档，或者参考我写的一个例子。

## 定义模型
模型可以是pytorch定义的，也可以是transformers定义的，这里想要微调的是chatglm模型，若是你想要训练其他模型如gpt，可以修改`--model_name_or_path`参数。本项目使用的模型是chatglm，因此路径是`--model_name_or_path THUDM/chatglm-6b`。如果你是自己定义的模型可以将源代码中的`model=...`修改为pytorch代码定义的模型即可。

## 训练模型代码train.py

### Argument Parsing

定义`parse_args`函数：

```python
def parse_args():
    parser = argparse.ArgumentParser()
    # 这里可以先用parser.add_argument添加一些其他参数，例如学习率，epoch，batch_size等参数
    ...
    
    # 这里可以通过json模块导入config文件给一个变量，并通过该变量传给deepspeed.initialize的config参数；也可以在shell命令行里面通过deepspeed --deepspeed --deepspeed_config deepspeed_config.json的方式，该方式就不需要通过该变量传给deepspeed.initialize的config参数
    parser.add_argument("--ds_config", type=str, default="deepspeed_config.json", help="deepspeed_config路径")
    
    # 定义一些deepspeed会自动改变的变量，local_rank，globa_lrank，local_rank为gpu编号（单个节点内的），globa_lrank为gpu全局编号（多个节点）
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank")
    
    # 加入deepspeed的arguments
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()
```

### deepspeed.init_distributed

需要在初始化deepspeed.initialize前使用，作用和torch.distributed.init_process_group(...)类似

其实deepspeed的main代码和torch.distributed有点像

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

### 初始化deepspeed.initialize

`deepspeed.initialize`使用方法：

- 返回参数有：[engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler]

- 一般输入参数:

  -  model: 必须，传入的model

  -  optimizer: 可选: torch定义或者config.json定义

  -  model_parameters: 可选：模型的parameters

  -  training_data: 可选，dataset类型

  -  lr_schduler: 可选: torch定义或者config.json定义

  -  config: 可选：config文件内容， 当你不想传入config参数，可以通过在shell命令行里面通过`deepspeed --deepspeed --deepspeed_config deepspeed_config.json`的方式，该方式就不需要通过该变量传给deepspeed.initialize的config参数，这里采用的是传入参数config，因为可以更加灵活的改变参数，比如gradient_accumulation_steps等

- 用法举例`model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, args=args, config=ds_config)`

### for循环训练

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

这里会自动调用scheduler和optimizer，如果指定gradient_accumulation_steps，会在gradient_accumulation_steps才进行一次update，这里model_engine的输入输出参数和原来的model是完全一样的。在获得loss后，你可以只是在调用model_engine.backward(loss)和model_engine.step()，无需调用model.zero_grad()和optimizer.step()等操作。

## 定义deepspeed的config的json文件

每个参数意义看这个文档https://www.deepspeed.ai/docs/config-json/#batch-size-related-parameters，可根据需求进行定义
ZeRO3分布式训练config的json文件定义模板：

```json
{
   //这里定义train_micro_batch_size_per_gpu和gradient_accumulation_steps然后忽略train_batch_size，因为train_batch_size = train_micro_batch_size_per_gpu * gpu数量 * gradient_accumulation_steps
  "gradient_accumulation_steps": 32,
  "train_micro_batch_size_per_gpu": 4,
    //多少个step打印一次信息，默认10
  "steps_per_print": 1,
    //ZeRO优化，这里选择的是阶段3，卸载到cpu内存上
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
    //梯度大于1进行裁剪
  "gradient_clipping": 1.0,
	//下面优化器和学习率优化器可以自己pytorch定义传入deepspeed.initialize
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
    //这个参数让你能够使用自己定义的torch.optim，但是如果你想用官方的你就把下面这句去掉即可
   "zero_force_ds_cpu_optimizer": false
}
```

这里是为了能够获得自己想要的学习率曲线，所以采用自定义的optimizer和scheduler，但是官方推荐用他们定义的optimizer和scheduler：

```
[rank0]: deepspeed.runtime.zero.utils.ZeRORuntimeException: You are using ZeRO-Offload with a client provided optimizer (<class 'torch.optim.adamw.AdamW'>) which in most cases will yield poor performance. Please either use deepspeed.ops.adam.DeepSpeedCPUAdam or set an optimizer in your ds-config (https://www.deepspeed.ai/docs/config-json/#optimizer-parameters). If you really want to use a custom optimizer w. ZeRO-Offload and understand the performance impacts you can also set <"zero_force_ds_cpu_optimizer": false> in your configuration file.
```

## 运行shell指令

这里分为单gpu和多gpu训练，具体参数意思可以看train.py的parse_args函数，里面都有详细注释。
### 单gpu训练

官方推荐使用--include localhost:...参数，去指定gpu，而不是通过CUDA_VISIBLE_DEVICES=...，而且可以通过`--master_port=<port>`指定运行的端口。不加的话，就是系统默认分配。关于`--offload_device nvme`是选nvme还是cpu，如果cpu内存不够就选nvme。

#### LoRA微调

运行`sh ./shell/run_single_gpu_lora.sh`

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
#### Freeze 冻结部分参数微调

运行`sh ./shell/run_single_gpu_freeze.sh`

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

#### 全量微调

运行`sh ./shell/run_single_gpu_full-tuning.sh`

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

### 多gpu训练

#### LoRA微调

运行`sh ./shell/run_multi_gpu_lora.sh`

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

#### Freeze 冻结部分参数微调

运行`sh ./shell/run_multi_gpu_freeze.sh`

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

#### 全量微调

运行`sh ./shell/run_multi_gpu_full-tuning.sh`

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

# 你可能遇见的问题

1. RuntimeError: Unable to JIT load the async_io op due to it not being compatible due to hardware/software issue. None

出现`RuntimeError: Unable to JIT load the async_io op due to it not being compatible due to hardware/software issue. None`错误时，意味着DeepSpeed的`async_io`操作无法在当前系统上即时编译（JIT）加载，因为存在硬件或软件兼容性问题。这个问题与`libaio`库的兼容性有关。要解决这个问题，你需要确保系统上安装了`libaio-dev`库。这个库是`async_io`操作所依赖的。在Ubuntu系统上，你可以通过以下命令安装`libaio-dev`：

```bash
sudo apt install libaio-dev
```

2. AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'

在shell命令前设置好环境变量export DS_SKIP_CUDA_CHECK=1。解决方法来自博主[deepspeed使用zero3 + offload报错:AttributeError: ‘DeepSpeedCPUAdam‘ object has no attribute ‘ds_opt_adam_deepspeed zero3 offload-CSDN博客](https://blog.csdn.net/qq_44193969/article/details/137051032)

