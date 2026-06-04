

class Metric:
    def __init__(self, name: str = None, params: dict = None):
        self.name = name
        self.params = params if params is not None else {}

    def __str__(self):
        return f"{self.name}"
    
    def calculate(self):
        raise NotImplementedError("Subclasses must implement the calculate method.")


class FullReferenceMetric(Metric):
    def __init__(self, name: str = None, params: dict = None):
        super().__init__(name=name, params=params)

    def calculate(self, X, Y):
        raise NotImplementedError("Subclasses must implement the calculate method.")