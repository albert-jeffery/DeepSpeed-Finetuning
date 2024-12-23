"""
Copyright (c) 2024 albert-jeffery. All rights reserved.
Project: DeepSpeed-Finetuning
GitHub: https://github.com/albert-jeffery/DeepSpeed-Finetuning
brief: the .py file is used to write some tools for the project.
"""

import matplotlib.pyplot as plt

def plot_lr(lr_list, epochs):
    plt.plot(list(range(1, epochs+1)), lr_list)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("learning rate's curve")
    plt.show()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs