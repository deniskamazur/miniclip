import logging

import clip
import torch
from torch import nn, optim

img_loss = nn.CrossEntropyLoss()
txt_loss = nn.CrossEntropyLoss()


def evaluate_model(model, test_loader, device):
    model.eval()

    with torch.no_grad():
        eval_loss = 0
        for imgs, txts, _ in test_loader:
            imgs = imgs.to(device)
            txts = clip.tokenize(txts).to(device)

            logits_per_image, logits_per_text = model(imgs, txts)
            ground_truth = torch.arange(test_loader.batch_size).to(device)

            total_loss = (img_loss(logits_per_image, ground_truth) + txt_loss(logits_per_text, ground_truth)) / 2
            eval_loss += total_loss * imgs.size(0)

        eval_loss = eval_loss / len(test_loader.dataset)

    return eval_loss


def train_model(model, train_loader, test_loader, device, num_epochs=200, learning_rate=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        len(train_loader) * num_epochs,
    )

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0
        for imgs, txts, _ in train_loader:
            imgs = imgs.to(device)
            txts = clip.tokenize(txts).to(device)

            optimizer.zero_grad()

            logits_per_image, logits_per_text = model(imgs, txts)
            ground_truth = torch.arange(train_loader.batch_size).to(device)

            total_loss = (img_loss(logits_per_image, ground_truth) + txt_loss(logits_per_text, ground_truth)) / 2
            total_loss.backward()

            train_loss += total_loss.item()

            optimizer.step()
            scheduler.step()

        train_loss /= len(train_loader.dataset)

        model.eval()
        eval_loss = evaluate_model(model, test_loader, device)

        logging.info(f"Epoch: {epoch}, train loss: {train_loss}, eval loss: {eval_loss}")

    return model
