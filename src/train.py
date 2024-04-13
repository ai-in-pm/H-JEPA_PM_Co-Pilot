from .data_utils import generate_self_supervised_data
from .model import model

def train_model(training_data, task_specific_data):
    self_supervised_data = generate_self_supervised_data(training_data)
    model.fit(self_supervised_data, epochs=10, batch_size=32)
    model.fit(task_specific_data, epochs=5, batch_size=32)
