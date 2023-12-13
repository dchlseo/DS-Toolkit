class MovingAverageFilter:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.classes = []

    def update(self, new_class):
        self.classes.append(new_class)
        if len(self.classes) >  self.window_size:
            self.classes.pop(0)

    def get_average(self):
        if not self.classes:
            return None
        return round(sum(self.classes) / len(self.classes))
    