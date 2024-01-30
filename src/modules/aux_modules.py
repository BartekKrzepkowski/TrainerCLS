from collections import defaultdict
from copy import deepcopy

import torch
from torch.distributions import Categorical

from src.utils import prepare
from src.utils.utils_optim import get_every_but_forbidden_parameter_names, FORBIDDEN_LAYER_TYPES

from torch.func import functional_call, vmap, grad


class TraceFIM(torch.nn.Module):
    def __init__(self, x_held_out, model, num_classes):
        super().__init__()
        self.device = next(model.parameters()).device
        self.x_held_out = x_held_out
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.ft_criterion = vmap(self.grad_and_trace, in_dims=(None, None, 0), randomness="different")
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        print("penalized_parameter_names: ", self.penalized_parameter_names)
        self.labels = torch.arange(num_classes).to(self.device)
        self.logger = None
        
    def compute_loss(self, params, buffers, sample):
        batch = sample.unsqueeze(0)
        y_pred = functional_call(self.model, (params, buffers), (batch, ))
        # y_sampled = Categorical(logits=y_pred).sample()
        prob = torch.nn.functional.softmax(y_pred, dim=1)
        idx_sampled = prob.multinomial(1)
        y_sampled = self.labels[idx_sampled].long().squeeze(-1)
        loss = self.criterion(y_pred, y_sampled)
        return loss
    
    def grad_and_trace(self, params, buffers, sample):
        sample_traces = {}
        sample_grads = grad(self.compute_loss, has_aux=False)(params, buffers, sample)
        for param_name in sample_grads:
            gr = sample_grads[param_name]
            if gr is not None:
                trace_p = (torch.pow(gr, 2)).sum()
                sample_traces[param_name] = trace_p
        return sample_traces

    def forward(self, step):
        self.model.eval()
        params = {k: v.detach() for k, v in self.model.named_parameters() if k in self.penalized_parameter_names and v.requires_grad}
        buffers = {}
        ft_per_sample_grads = self.ft_criterion(params, buffers, self.x_held_out)
        ft_per_sample_grads = {k: v.detach().data for k, v in ft_per_sample_grads.items()}
        evaluators = defaultdict(float)
        overall_trace = 0.0
        for param_name in ft_per_sample_grads:
            trace_p = ft_per_sample_grads[param_name].mean()
            evaluators[f'trace_fim/{param_name}'] += trace_p.item()
            if param_name in self.penalized_parameter_names:
                overall_trace += trace_p.item()
         
        evaluators[f'trace_fim/overall_trace'] = overall_trace
        evaluators['steps/trace_fim'] = step
        self.model.train()
        self.logger.log_scalars(evaluators, step)
