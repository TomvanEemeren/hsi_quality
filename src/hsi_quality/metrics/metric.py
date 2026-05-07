class Metric:
    def __init__(self, name: str = None):
        self.name = name

    def __str__(self):
        return f"{self.name}"
    
    def calculate(self):
        raise NotImplementedError("Subclasses must implement the calculate method.")