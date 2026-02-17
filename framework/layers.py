from framework.tensor import Tensor
import random
import cpp_conv


def random_matrix(rows, cols):
    return [[random.uniform(-0.1, 0.1) for _ in range(cols)] for _ in range(rows)]


class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Tensor(
            random_matrix(in_features, out_features),
            requires_grad=True
        )

        self.bias = Tensor(
            [0.0 for _ in range(out_features)],
            requires_grad=True
        )

    def __call__(self, x):
        # x: (B, in_features)

        B = len(x.data)

        output = []

        for b in range(B):
            row = []
            for j in range(self.out_features):
                s = 0.0
                for i in range(self.in_features):
                    s += x.data[b][i] * self.weight.data[i][j]
                s += self.bias.data[j]
                row.append(s)
            output.append(row)

        out = Tensor(output, requires_grad=True)
        out._prev = {x, self.weight, self.bias}

        def _backward():
            # grad_output: (B, out_features)

            # grad wrt x
            if x.requires_grad:
                grad_x = []
                for b in range(B):
                    row = []
                    for i in range(self.in_features):
                        s = 0.0
                        for j in range(self.out_features):
                            s += out.grad[b][j] * self.weight.data[i][j]
                        row.append(s)
                    grad_x.append(row)
                x.grad = grad_x

            # grad wrt weight
            if self.weight.requires_grad:
                grad_w = []
                for i in range(self.in_features):
                    row = []
                    for j in range(self.out_features):
                        s = 0.0
                        for b in range(B):
                            s += x.data[b][i] * out.grad[b][j]
                        row.append(s)
                    grad_w.append(row)
                self.weight.grad = grad_w

            # grad wrt bias
            if self.bias.requires_grad:
                grad_b = []
                for j in range(self.out_features):
                    s = 0.0
                    for b in range(B):
                        s += out.grad[b][j]
                    grad_b.append(s)
                self.bias.grad = grad_b

        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight, self.bias]


def random_kernel(k):
    return [[random.uniform(-0.1, 0.1) for _ in range(k)] for _ in range(k)]


def random_kernel_4d(out_c, in_c, k):
    return [
        [
            [[random.uniform(-0.1, 0.1) for _ in range(k)] for _ in range(k)]
            for _ in range(in_c)
        ]
        for _ in range(out_c)
    ]


def random_bias(out_c):
    return [0.0 for _ in range(out_c)]


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.kernel = Tensor(
            random_kernel_4d(out_channels, in_channels, kernel_size),
            requires_grad=True
        )

        self.bias = Tensor(
            random_bias(out_channels),
            requires_grad=True
        )

    def __call__(self, x):
        # x: (B, C_in, H, W)

        B = len(x.data)
        C_in = self.in_channels
        C_out = self.out_channels
        K = self.kernel_size

        H = len(x.data[0][0])
        W = len(x.data[0][0][0])

        out_h = H - K + 1
        out_w = W - K + 1

        output = cpp_conv.conv_forward_batch(
            x.data,
            self.kernel.data,
            self.bias.data
        )

        requires_grad = x.requires_grad or self.kernel.requires_grad
        out = Tensor(output, requires_grad=requires_grad)

        out._prev = {x, self.kernel, self.bias}

        def _backward():

            grad_input, grad_kernel, grad_bias = cpp_conv.conv_backward_batch(
                x.data,
                self.kernel.data,
                out.grad
            )

            # ----- ACCUMULATE GRADIENTS PROPERLY -----

            # Input grad
            if x.requires_grad:
                if x.grad is None:
                    x.grad = grad_input
                else:
                    for b in range(B):
                        for c in range(C_in):
                            for i in range(H):
                                for j in range(W):
                                    x.grad[b][c][i][j] += grad_input[b][c][i][j]

            # Kernel grad
            if self.kernel.requires_grad:
                if self.kernel.grad is None:
                    self.kernel.grad = grad_kernel
                else:
                    for oc in range(C_out):
                        for ic in range(C_in):
                            for ki in range(K):
                                for kj in range(K):
                                    self.kernel.grad[oc][ic][ki][kj] += \
                                        grad_kernel[oc][ic][ki][kj]

            # Bias grad
            if self.bias.requires_grad:
                if self.bias.grad is None:
                    self.bias.grad = grad_bias
                else:
                    for oc in range(C_out):
                        self.bias.grad[oc] += grad_bias[oc]

        out._backward = _backward
        return out

    def parameters(self):
        return [self.kernel, self.bias]

class MaxPool2D:
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size

    def __call__(self, x):
        # x: (B, C, H, W)

        B = len(x.data)
        C = len(x.data[0])
        H = len(x.data[0][0])
        W = len(x.data[0][0][0])
        K = self.kernel_size

        out_h = H // K
        out_w = W // K

        output = []
        self.max_indices = []

        for b in range(B):
            batch_out = []
            batch_indices = []

            for c in range(C):
                channel_out = []
                channel_indices = []

                for i in range(out_h):
                    row = []
                    row_indices = []

                    for j in range(out_w):
                        max_val = -float('inf')
                        max_pos = (0, 0)

                        for ki in range(K):
                            for kj in range(K):
                                val = x.data[b][c][i*K+ki][j*K+kj]
                                if val > max_val:
                                    max_val = val
                                    max_pos = (i*K+ki, j*K+kj)

                        row.append(max_val)
                        row_indices.append(max_pos)

                    channel_out.append(row)
                    channel_indices.append(row_indices)

                batch_out.append(channel_out)
                batch_indices.append(channel_indices)

            output.append(batch_out)
            self.max_indices.append(batch_indices)

        requires_grad = x.requires_grad
        out = Tensor(output, requires_grad=requires_grad)
        out._prev = {x} if x.requires_grad else set()

        def _backward():
            if x.requires_grad:
                for b in range(B):
                    for c in range(C):
                        for i in range(out_h):
                            for j in range(out_w):
                                max_i, max_j = self.max_indices[b][c][i][j]
                                x.grad[b][c][max_i][max_j] += out.grad[b][c][i][j]

        out._backward = _backward
        return out


class Flatten:
    def __call__(self, x):
        # x: (B, C, H, W)

        B = len(x.data)
        C = len(x.data[0])
        H = len(x.data[0][0])
        W = len(x.data[0][0][0])

        output = []

        for b in range(B):
            flat = []
            for c in range(C):
                for i in range(H):
                    for j in range(W):
                        flat.append(x.data[b][c][i][j])
            output.append(flat)

        requires_grad = x.requires_grad
        out = Tensor(output, requires_grad=requires_grad)
        out._prev = {x} if x.requires_grad else set()

        def _backward():
            if x.requires_grad:
                for b in range(B):
                    idx = 0
                    for c in range(C):
                        for i in range(H):
                            for j in range(W):
                                x.grad[b][c][i][j] += out.grad[b][idx]
                                idx += 1

        out._backward = _backward
        return out


    def parameters(self):
        return []
