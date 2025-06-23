class Agent:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, 'tools'):
            self.tools = []

    def run(self, *args, **kwargs):
        raise NotImplementedError('Method run must be implemented by subclasses')
