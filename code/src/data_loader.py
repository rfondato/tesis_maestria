#%%
from torch.utils.data import Dataset, DataLoader
import os
from constants import LABELS
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split

class HARDataset (Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index) -> tuple:
        return self.X[index, :], self.y[index]
    
    def __len__(self):
        return len(self.X)
    
def load_sequences(data: np.ndarray, seq_length:int, stride:int):
    if len(data) < seq_length:
        return np.array([])

    sequences = []
    for i in range(0, len(data) - seq_length, stride):
        sequences.append(data[i:(i+seq_length), :])

    return np.array(sequences)

def get_loaders(seq_length=30, stride=5, batch_size=64, shuffle=True):
    X, y = None, None
    for l in tqdm.tqdm(LABELS, desc="Processing labels"):
        path = f'../preprocess/{l}'
        data = [load_sequences(np.load(f"{path}/{f}"), seq_length, stride) for f in os.listdir(path) if os.path.isfile(f"{path}/{f}")]
        temp_stack = None
        for d in filter(lambda x: len(x) > 0, data):
            temp_stack = d if temp_stack is None else np.vstack((temp_stack, d))
        
        labels = np.repeat(np.array([LABELS[l]]), len(temp_stack))
        X = temp_stack if X is None else np.vstack((X, temp_stack))
        y = labels if y is None else np.concatenate((y, labels))

    X = np.expand_dims(X, 1)
    y = y.astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=8, shuffle=shuffle)

    training_dataset = HARDataset(X_train, y_train)
    testing_dataset = HARDataset(X_test, y_test)

    return DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(testing_dataset, batch_size=batch_size)

if __name__ == "__main__":
    loader = get_loaders()

#model = HARModule(device="cuda")
#trainer = L.Trainer(max_epochs=100)


# %%
