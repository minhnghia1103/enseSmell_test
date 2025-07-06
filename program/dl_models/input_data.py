class Input_data:
    def __init__(self, train_data, train_labels, eval_data, eval_labels, max_input_length):
        self.train_data = train_data
        self.train_labels = train_labels
        self.eval_data = eval_data
        self.eval_labels = eval_labels
        self.max_input_length = max_input_length

class Input_CombineData:
    def __init__(self, train_data, train_data_metrics, train_labels, eval_data, eval_data_metrics, eval_labels, train_graph_data=None, eval_graph_data=None):
        self.train_data = train_data
        self.train_data_metrics = train_data_metrics
        self.train_labels = train_labels
        self.eval_data = eval_data
        self.eval_data_metrics = eval_data_metrics
        self.eval_labels = eval_labels
        self.train_graph_data = train_graph_data
        self.eval_graph_data = eval_graph_data

# data class for 2-d data
class Input_data2:
    def __init__(self, train_data, train_labels, eval_data, eval_labels, max_input_height, max_input_width):
        self.train_data = train_data
        self.train_labels = train_labels
        self.eval_data = eval_data
        self.eval_labels = eval_labels
        self.max_input_height = max_input_height
        self.max_input_width = max_input_width