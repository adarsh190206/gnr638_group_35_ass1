def update_param(data, grad, lr):
    if isinstance(data, list):
        return [
            update_param(d, g, lr)
            for d, g in zip(data, grad)
        ]
    else:
        return data - lr * grad


def zero_grad_recursive(grad):
    if isinstance(grad, list):
        return [zero_grad_recursive(g) for g in grad]
    else:
        return 0.0


class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.requires_grad:
                p.data = update_param(p.data, p.grad, self.lr)

    def zero_grad(self):
        for p in self.parameters:
            if p.requires_grad:
                p.grad = zero_grad_recursive(p.grad)
