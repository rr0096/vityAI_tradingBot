class BaseTool:
    name: str = ''
    description: str = ''

    def __init__(self, *args, **kwargs):
        pass

    def _run(self, *args, **kwargs):
        raise NotImplementedError('Tool logic not implemented')

    def run(self, *args, **kwargs):
        return self._run(*args, **kwargs)
