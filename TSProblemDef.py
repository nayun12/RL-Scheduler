
import torch
import numpy as np


def get_random_problems(batch_size, problem_size, machine_size):
    problems = torch.rand(size=(batch_size, problem_size+1, machine_size+1))
    # problem_size + 1 : token
    # machine_size + 1 : due_date
    # problems[:,:,-1] = problems[:,:,-1] + 1
    # problems.shape: (batch, problem, 2)
    problems[:,0,:] = 0
    return problems