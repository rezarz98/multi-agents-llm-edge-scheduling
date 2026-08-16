class GPU:
    def __init__(self, id, speed, bs_id, current_time=0):
        self.id = id
        self.speed = speed
        self.bs_id = bs_id
        self.current_time = current_time
        self.completed_tasks = []

    def __str__(self):
        return f"GPU-{self.id} (Speed: {self.speed}, Current Time: {self.current_time})"
