def fgsm_attack(x, y, model, epsilon, loss_fn):
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = loss_fn(output, y)
    loss.backward()
    return x_adv + epsilon*x_adv.grad.sign()
