import math
from framework.tensor import Tensor


class CrossEntropyLoss:
    def __call__(self, logits, targets):
        # logits: (B, num_classes)
        # targets: list of size B

        B = len(logits.data)
        num_classes = len(logits.data[0])

        loss_value = 0.0
        probs = []

        for b in range(B):

            # Numerically stable softmax
            max_logit = max(logits.data[b])
            exp_vals = [math.exp(v - max_logit) for v in logits.data[b]]
            sum_exp = sum(exp_vals)

            softmax = [v / sum_exp for v in exp_vals]
            probs.append(softmax)

            loss_value -= math.log(softmax[targets[b]] + 1e-12)

        loss_value /= B

        out = Tensor(loss_value, requires_grad=True)
        out._prev = {logits}

        def _backward():
            grad_logits = []

            for b in range(B):
                grad_row = []
                for c in range(num_classes):
                    grad = probs[b][c]
                    if c == targets[b]:
                        grad -= 1.0
                    grad_row.append(grad / B)
                grad_logits.append(grad_row)

            logits.grad = grad_logits

        out._backward = _backward
        return out
