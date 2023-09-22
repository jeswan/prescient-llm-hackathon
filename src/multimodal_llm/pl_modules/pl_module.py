import pytorch_lightning as pl


class BasicLightningModule(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer_cls):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True)
        # return the validation targets and predictions as a dict
        return {"val_targets": y, "val_preds": y_hat}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss_fn(y_hat, y)
        self.log("test_loss", test_loss, prog_bar=True)
        # return the test targets and predictions as a dict
        return {"test_targets": y, "test_preds": y_hat}

    def configure_optimizers(self):
        return self.optimizer_cls(params=self.model.parameters())
