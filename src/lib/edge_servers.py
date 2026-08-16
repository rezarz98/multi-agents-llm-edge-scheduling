from .gpu import GPU


class EdgeServer:
    def __init__(self, id, gpu_speeds):
        self.id = id
        self.gpus = self.create_gpus(gpu_speeds)

    def create_gpus(self, gpu_speeds):
        return [
            GPU(id=i, speed=speed, bs_id=self.id, current_time=0)
            for i, speed in enumerate(gpu_speeds, start=1)
        ]

    def __str__(self):
        gpu_info = "\n".join(str(gpu) for gpu in self.gpus)
        return f"EdgeServer-{self.id}\nGPUs:\n{gpu_info}"
