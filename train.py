"""
Copyright (c) 2024 albert-jeffery. All rights reserved.
Project: DeepSpeed-Finetuning
GitHub: https://github.com/albert-jeffery/DeepSpeed-Finetuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import inspect
import argparse
import deepspeed
import os
import numpy as np
import time
from util.tools import *
import json
from transformers import AutoTokenizer, AutoModel
from util.dataprocess import CustomDataset, CustomDataCollator
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    raise RuntimeError("tensorboard not installed, run pip install tensorboard. pip install tensorboard")

def parse_args():
    parser = argparse.ArgumentParser(description='DeepSpeed ZeRO')

    # which model you tend to finetuing
    parser.add_argument('--model_name_or_path', type=str, default='THUDM/chatglm3-6b', help='model name or path, you can also pass the path of model you want to finetune')
    parser.add_argument('--src_len', type=int, default=512, help='max source sentence length')
    parser.add_argument('--tgt_len', type=int, default=128, help='max target sentence length')

    # dataset params
    parser.add_argument('--data_path', type=str, default='./data/train_pubblic.json', help='Path to the training dataset.')

    # typical params
    parser.add_argument('--train_micro_batch_size_per_gpu', type=int, default=8, help='batch size per gpu')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='gradient accumulation steps')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='max learning rate')
    parser.add_argument('--initial_lr', type=float, default=1e-6, help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-8, help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='adam beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='adam beta2')
    parser.add_argument('--fused', action='store_true', help='whether to use fused optimizer, if you can load all prameters of the model on a single gpu.')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/',help='save dir')

    # log dir of tensorboard
    parser.add_argument('--log_dir', type=str, default='./logs', help='log dir')

    # which fintuning method to use
    parser.add_argument('--finetune_method', type=str, default='lora', help='finetune method, support parameters lora, freeze, full-tuning')

    # freeze modules
    parser.add_argument('--freeze_modules', type=str, default='dense_h_to_4h', help='the layer of model you wanna freeze')

    # LoRA params
    parser.add_argument('--lora_alpha', type=int, default=32, help='alpha for LoRA')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='dropout probability for LoRA')
    parser.add_argument('--lora_target_modules', type=str, default='query_key_value', help='target modules for LoRA, the name of layer in model you wanna use LoRA')
    parser.add_argument('--lora_r', type=int, default=8, help='r for LoRA')

    # deepspeed params
    parser.add_argument('--ds_config_path', type=str, default='./config/ds_config.json', help='path to deepspeed config file')
    parser.add_argument('--offload_device', type=str, default='cpu', help='offload device, cpu or nvme, which mean you want to offload the model to cpu memory or nvme ssd')
    parser.add_argument('--nvme_path', type=str, default='./mnt/nvme', help='path to nvme ssd')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument('--global_rank', type=int, default=-1, help='global rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # init deepspeed
    if args.local_rank == -1:
        r"""
        when you don't use deepspeed to train your own model with a single gpu, you need to modify the 
        train loop according to pytorch traditional grammer. when arg.local_rank == -1, mean you are using 
        pytorch to train your model with a single gpu.
        """
        device = torch.device("cuda")
    else:
        device = torch.device("cuda", args.local_rank)
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()
        torch.distributed.barrier()
    master_process = (args.local_rank == 0)

    if master_process:
        # determine is there a folder called logs
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        # of course, you can achieve log by setting the dsconfig.json file, without using the SummaryWriter
        tb_writer = SummaryWriter(args.log_dir)
        print("builded tensorboard writer")

    # get model, and you can add other finetuning methods here, eg: prefix tuning, P-tuning, etc.
    if args.finetune_method == "full-tuning":
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    elif args.finetune_method == "lora":
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        lora_target_modules = args.lora_target_modules.split(",")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            target_modules=lora_target_modules,
            inference_mode=False
            )
        model = get_peft_model(model, lora_config)
    elif args.finetune_method == "freeze":
        freeze_modules = args.freeze_modules.split(",")
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        for name, param in model.named_parameters():
            if any(item in name for item in freeze_modules): 
                param.requires_grad = False
    else:
        raise ValueError("Invalid finetune method")
    
    if args.fused:
        model = model.to(device)

    # print the number of the model parameters
    if master_process:
        print(model)
        print(f"Local rank: {args.local_rank} \n Total number of parameters: {sum(p.numel() for p in model.parameters())}")

    # init optimizer
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    # init optimizer with weight decay and no weight decay
    optim_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    r"""using fused adamw if fused is available, and note that the fused adamw is only used in 
    the situation when you can load all prameters of the model on a single gpu.     
    """
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device.type == 'cuda' and args.fused
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=args.initial_lr, 
        betas=(args.adam_beta1, args.adam_beta2),
        **extra_args
        )
    print(f"using fused AdamW: {use_fused}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    train_dataset = CustomDataset(args.data_path, tokenizer=tokenizer)
    # get the dataloader
    if args.local_rank == -1:
        data_collator = CustomDataCollator(tokenizer=tokenizer)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.train_micro_batch_size_per_gpu, 
            shuffle=True,
            collate_fn=data_collator,
            )
    else:
        train_sampler = DistributedSampler(train_dataset)
        data_collator = CustomDataCollator(tokenizer=tokenizer)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            sampler=train_sampler, 
            batch_size=args.train_micro_batch_size_per_gpu,
            collate_fn=data_collator,
        )

    r"""
    Define torch shceduler, which will automatically adjust LR.
    div_factor (float): Determines the initial learning rate via
        initial_lr = max_lr/div_factor
        Default: 25
    final_div_factor (float): Determines the minimum learning rate via
        min_lr = initial_lr/final_div_factor
        Default: 1e4
    So you just need to set max_lr, initial_lr, min_lr, and epochs.
    """
    max_lr = args.max_lr
    div_factor = int(max_lr / args.initial_lr)
    final_div_factor = int(args.initial_lr / args.min_lr)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=max_lr, 
        steps_per_epoch=len(train_loader), 
        epochs=args.epochs, 
        anneal_strategy="cos",
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        three_phase=True,
        pct_start=0.35
    )

    # load the deepspeed config json file
    ds_config = json.load(open(args.ds_config_path))
    ds_config['train_micro_batch_size_per_gpu'] = args.train_micro_batch_size_per_gpu
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    ds_config['zero_optimization']['offload_param']['device'] = args.offload_device
    ds_config['zero_optimization']['offload_optimizer']['device'] = args.offload_device
    if args.offload_device == "nvme":
        # here don't use os.path.mkdir, because it may cover the original directory.
        if not os.path.exists(args.nvme_path):
            raise ValueError(f"nvme path does not exist, please make directory {args.nvme_path} by yourself.")
        else:
            ds_config['zero_optimization']['offload_param']['nvme_path'] = args.nvme_path
            ds_config['zero_optimization']['offload_optimizer']['nvme_path'] = args.nvme_path

    if master_process:        
        print(f"DeepSpeed config: {ds_config}")

    # deepspeed initialize
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config,
        optimizer=optimizer,
        lr_scheduler=scheduler
        )
    
    # train
    lr_list = []
    loss_past = float('inf')
    loss_epoch = 0.0
    loss_now = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        # get the learning rate, and add it to the list
        lr_list.append(engine.lr_scheduler.get_last_lr()[0])
        engine.train()
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            r"""
            the parameters of engine you created by deepspeed.initialize is according to the forward function
            of model. model is the one was wrapped by deepspeed.initialize, which coule be a chatglm model, your
            custom model, or any other model. moreover, the output of engine is also the output of original model.
            for chatglm, the output is return CausalLMOutputWithPast(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            ), which inclue the loss, logits, past_key_values, and so on.
            """
            outputs = engine(input_ids=inputs, labels=labels)
            engine.backward(outputs.loss)
            engine.step()
            loss_epoch += outputs.loss.item()
            loss_now += outputs.loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                loss_now = loss_now / args.gradient_accumulation_steps
                if master_process:
                    t1 = time.time()
                    mins, secs = epoch_time(t0, t1)
                    print(f"[Epoch {epoch}][Step {step}][Time {mins}:{secs}] loss: {loss_now}") 
                loss_now = 0.0    
        loss_epoch = loss_epoch / len(train_loader)
        # save the best model
        if loss_epoch < loss_past:
            loss_past = loss_epoch
            r"""
            if you do not use .save_checkpoint(), you need to use the 
            engine._zero3_consolidated_16bit_state_dict() to consolidate 
            the state dict distributed across different processes. Futhermore,
            .save_checkpoint() can not be used in specific process. eg: global rank 0.
            More details can be found in the official document.
            """
            engine.save_checkpoint(f"{args.output_dir}{epoch}-{loss_epoch}")   
        # print the loss
        if master_process:
            t1 = time.time()
            mins, secs = epoch_time(t0, t1)
            tb_writer.add_scalar("loss", loss_epoch, epoch)
            print(f"epoch: {epoch}, loss: {loss_epoch}, min:{mins}, sec:{secs}")
            t0 = t1
        loss_epoch = 0.0

    # plot learning rate    
    try:
        plot_lr(lr_list, args.epochs)
    except:
        print("can't plot learning rate curve")

if __name__ == "__main__":
    main()