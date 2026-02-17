# framework/tensor.py

def zeros_like(data):
    if isinstance(data, list):
        return [zeros_like(x) for x in data]
    else:
        return 0.0


def ones_like(data):
    if isinstance(data, list):
        return [ones_like(x) for x in data]
    else:
        return 1.0


def add_lists(a, b):
    if isinstance(a, list):
        return [add_lists(x, y) for x, y in zip(a, b)]
    else:
        return a + b


def mul_lists(a, b):
    if isinstance(a, list):
        return [mul_lists(x, y) for x, y in zip(a, b)]
    else:
        return a * b


def scalar_mul(scalar, data):
    if isinstance(data, list):
        return [scalar_mul(scalar, x) for x in data]
    else:
        return scalar * data


def relu_list(data):
    if isinstance(data, list):
        return [relu_list(x) for x in data]
    else:
        return max(0.0, data)


def relu_grad(data):
    if isinstance(data, list):
        return [relu_grad(x) for x in data]
    else:
        return 1.0 if data > 0 else 0.0


def matmul(a, b):
    # a: (m x n)
    # b: (n x p)
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            s = 0.0
            for k in range(len(b)):
                s += a[i][k] * b[k][j]
            row.append(s)
        result.append(row)
    return result


def transpose(m):
    return list(map(list, zip(*m)))


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.grad = zeros_like(data)
        self.requires_grad = requires_grad

        self._backward = lambda: None
        self._prev = set()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    # -----------------
    # Addition
    # -----------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(add_lists(self.data, other.data), requires_grad=True)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad = add_lists(self.grad, out.grad)
            if other.requires_grad:
                other.grad = add_lists(other.grad, out.grad)

        out._backward = _backward
        return out

    # -----------------
    # Elementwise Multiplication
    # -----------------
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(mul_lists(self.data, other.data), requires_grad=True)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad = add_lists(
                    self.grad,
                    mul_lists(other.data, out.grad)
                )
            if other.requires_grad:
                other.grad = add_lists(
                    other.grad,
                    mul_lists(self.data, out.grad)
                )

        out._backward = _backward
        return out

    # -----------------
    # Matrix Multiplication
    # -----------------
    def matmul(self, other):
        out = Tensor(matmul(self.data, other.data), requires_grad=True)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad = add_lists(
                    self.grad,
                    matmul(out.grad, transpose(other.data))
                )
            if other.requires_grad:
                other.grad = add_lists(
                    other.grad,
                    matmul(transpose(self.data), out.grad)
                )

        out._backward = _backward
        return out

    # -----------------
    # ReLU
    # -----------------
    def relu(self):
        out = Tensor(relu_list(self.data), requires_grad=True)
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                self.grad = add_lists(
                    self.grad,
                    mul_lists(relu_grad(self.data), out.grad)
                )

        out._backward = _backward
        return out

    # -----------------
    # Backward Engine
    # -----------------
    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = ones_like(self.data)

        for node in reversed(topo):
            node._backward()
