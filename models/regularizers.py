import torch

def l2_perturbation(real, adv):
    return torch.sqrt(((real-adv)**2).sum(dim=1))

def l0_perturbation(real, adv):
    return (real != adv).sum(dim=1)

def energy_regularizer(real, adv, l2_weight=1.0, l0_weight=0.1):
    return l2_weight*l2_perturbation(real, adv).mean() + l0_weight*l0_perturbation(real, adv).float().mean()
