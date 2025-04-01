class BaseLLM:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
    
    def build_model(self, **kwargs):
        raise NotImplementedError

    def generate(self, **kwargs):
        raise NotImplementedError