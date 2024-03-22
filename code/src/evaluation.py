#%%
from har_model import HARModule
from data_loader import get_loaders
import torch.nn.functional as F
import torch as t
import lightning as L
from sklearn.metrics import confusion_matrix
from constants import LABELS

training_dl, testing_dl = get_loaders(seq_length=30, stride=5, batch_size=128, shuffle=True)

model = HARModule.load_from_checkpoint("../models/CNN-Res100-Seq30-epoch=6-val_loss=0.02.ckpt", device="cuda")
model.eval()

trainer = L.Trainer()
pred = t.cat(trainer.predict(model, testing_dl), dim=0)

y_pred = t.argmax(pred, dim=1)
y_true = t.from_numpy(testing_dl.dataset.y)

confusion_matrix(y_true, y_pred)

print(f"Accuracy: {len(y_true[y_pred == y_true]) / len(y_true) * 100:.2f}%")
# %%
