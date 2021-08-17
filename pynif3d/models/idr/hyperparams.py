class SoftplusParams:
    __slots__ = ["beta"]

    def __init__(self, beta=100):
        self.beta = beta


class GeometricInitParams:
    __slots__ = ["bias"]

    def __init__(self, bias=0.6):
        self.bias = bias


class IDRHyperParams:
    def __init__(self, softplus=None, geometric_init=None):
        self.softplus = softplus
        if softplus is None:
            self.softplus = SoftplusParams()

        self.geometric_init = geometric_init
        if geometric_init is None:
            self.geometric_init = GeometricInitParams()
