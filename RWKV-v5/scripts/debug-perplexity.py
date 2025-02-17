import torch
from torch.nn import functional as F
import numpy as np
import math
import pandas as pd

torch.set_printoptions(threshold=100)

real_logits = torch.load("real-logits.npy", weights_only=True).to("cpu")
fake_logits = torch.load("fake-logits.npy", weights_only=True)
cls = np.load("01b-x59-cls.npy")

df1 = pd.DataFrame(real_logits.numpy())
df2 = pd.DataFrame(fake_logits.numpy())
df3 = pd.DataFrame(cls)

with pd.ExcelWriter('logits.xlsx') as writer:
    df1.to_excel(writer, sheet_name='real')
    df2.to_excel(writer, sheet_name='fake')
    df3.to_excel(writer, sheet_name='cls')

dst = 57435
print(real_logits[0:dst+50])
print(fake_logits[0:dst+50])


print(f"real_logits_std: {real_logits.std()} fake_logits_std: {fake_logits.std()}")
print(f"real_logits_mean: {real_logits.mean()} fake_logits_mean: {fake_logits.mean()}")
print(math.log(F.softmax(real_logits, dim=-1)[dst]))
print(math.log(F.softmax(fake_logits, dim=-1)[dst]))
