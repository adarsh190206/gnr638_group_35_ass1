from framework import backend


class SimpleCNN:

    def __init__(self, num_classes, in_channels=1):


        # =====================================
        # Conv Layer 1
        # Input: (B, 1, 28, 28)
        # Output: (B, 8, 26, 26)
        # =====================================

        self.conv1_w = backend.Tensor([8, 1, 3, 3])
        self.conv1_b = backend.Tensor([8])

        # After maxpool → (B, 8, 13, 13)

        # =====================================
        # Conv Layer 2
        # Input: (B, 8, 13, 13)
        # Output: (B, 16, 11, 11)
        # =====================================

        self.conv2_w = backend.Tensor([16, 8, 3, 3])
        self.conv2_b = backend.Tensor([16])

        # After maxpool → (B, 16, 5, 5)

        # =====================================
        # Fully Connected Layers
        # =====================================

        # 16 × 5 × 5 = 400
        self.fc1_w = backend.Tensor([400, 128])
        self.fc1_b = backend.Tensor([128])

        self.fc2_w = backend.Tensor([128, num_classes])
        self.fc2_b = backend.Tensor([num_classes])

        # =====================================
        # Initialize weights
        # =====================================

        for p in self.parameters():
            backend.random_init(p)


    # =====================================
    # Parameter list
    # =====================================

    def parameters(self):
        return [
            self.conv1_w, self.conv1_b,
            self.conv2_w, self.conv2_b,
            self.fc1_w, self.fc1_b,
            self.fc2_w, self.fc2_b
        ]

    # =====================================
    # Zero gradients
    # =====================================

    def zero_grad(self):
        for p in self.parameters():
            backend.zero_grad(p)

    # =====================================
    # Forward pass
    # =====================================

    def forward(self, x):

        self.x = x

        # ---- Conv 1 ----
        self.c1 = backend.conv_forward(
            self.x, self.conv1_w, self.conv1_b
        )
        self.r1 = backend.relu_forward(self.c1)
        self.p1 = backend.maxpool_forward(self.r1)

        # ---- Conv 2 ----
        self.c2 = backend.conv_forward(
            self.p1, self.conv2_w, self.conv2_b
        )
        self.r2 = backend.relu_forward(self.c2)
        self.p2 = backend.maxpool_forward(self.r2)

        # ---- Flatten ----
        self.flat = backend.flatten_forward(self.p2)

        # ---- FC1 ----
        self.fc1 = backend.linear_forward(
            self.flat, self.fc1_w, self.fc1_b
        )
        self.r3 = backend.relu_forward(self.fc1)

        # ---- FC2 ----
        self.logits = backend.linear_forward(
            self.r3, self.fc2_w, self.fc2_b
        )

        return self.logits

    # =====================================
    # Backward pass
    # =====================================

    def backward(self, targets):

        self.zero_grad()
        self.x.grad = [0.0] * len(self.x.grad)
        self.c1.grad = [0.0] * len(self.c1.grad)
        self.r1.grad = [0.0] * len(self.r1.grad)
        self.p1.grad = [0.0] * len(self.p1.grad)
        self.c2.grad = [0.0] * len(self.c2.grad)
        self.r2.grad = [0.0] * len(self.r2.grad)
        self.p2.grad = [0.0] * len(self.p2.grad)
        self.flat.grad = [0.0] * len(self.flat.grad)
        self.fc1.grad = [0.0] * len(self.fc1.grad)
        self.r3.grad = [0.0] * len(self.r3.grad)
        self.logits.grad = [0.0] * len(self.logits.grad)

        loss = backend.cross_entropy(self.logits, targets)

        # ---- FC2 ----
        backend.linear_backward(
            self.r3, self.fc2_w, self.fc2_b, self.logits
        )

        # ---- ReLU 3 ----
        backend.relu_backward(self.fc1, self.r3)

        # ---- FC1 ----
        backend.linear_backward(
            self.flat, self.fc1_w, self.fc1_b, self.fc1
        )

        # ---- Flatten ----
        backend.flatten_backward(self.p2, self.flat)

        # ---- Pool 2 ----
        backend.maxpool_backward(self.r2, self.p2)

        # ---- ReLU 2 ----
        backend.relu_backward(self.c2, self.r2)

        # ---- Conv 2 ----
        backend.conv_backward(
            self.p1, self.conv2_w, self.conv2_b, self.c2
        )

        # ---- Pool 1 ----
        backend.maxpool_backward(self.r1, self.p1)

        # ---- ReLU 1 ----
        backend.relu_backward(self.c1, self.r1)

        # ---- Conv 1 ----
        backend.conv_backward(
            self.x, self.conv1_w, self.conv1_b, self.c1
        )

        return loss

    # =====================================
    # SGD Step
    # =====================================

    def step(self, lr):

        for p in self.parameters():
            backend.sgd_update(p, lr)
