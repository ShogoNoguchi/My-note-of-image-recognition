from typing import Callable

import torch
from torch import nn
from torch.utils.data import Dataset


"""
data_loader: 評価に使うデータを読み込むデータローダ
model      : 評価対象のモデル
loss_func  : 目的関数
"""


def evaluate(data_loader: Dataset, model: nn.Module, loss_func: Callable):
    model.eval()

    losses = []
    preds = []
    for x, y in data_loader:
        # 評価時は勾配計算をしない
        with torch.no_grad():
            x = x.to(model.get_device())
            y = y.to(model.get_device())

            y_pred = model(x)

            losses.append(loss_func(y_pred, y, reduction="none"))
            preds.append(y_pred.argmax(dim=1) == y)

    loss = torch.cat(losses).mean()
    accuracy = torch.cat(preds).float().mean()

    return loss, accuracy
