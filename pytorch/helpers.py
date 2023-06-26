def eval_model(model, dataloader, loss_fn):
    loss = 0
    acc = 0
    for x, y in dataloader:
        y_hat = model(x)
        loss += loss_fn(y_hat, y)
        acc += y_hat.argmax(dim=1).eq(y).float().mean()
    loss /= len(dataloader)
    acc /= len(dataloader)
    return loss, acc
