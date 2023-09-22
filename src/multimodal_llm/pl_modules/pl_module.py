import pytorch_lightning as pl


class BasicLightningModule(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer_cls):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls

    # def forward(self, idx, img):
    #     return self.model(idx=idx, img=img)

    def _common_step(self, batch, batch_idx):
        # batches:
        input_text_ids = batch["input_text_ids"]
        target_text_ids = batch["target_text_ids"]
        image_tensor = batch["image_tensor"]

        y_hat = self.model(idx=input_text_ids, img=image_tensor)
        loss = self.loss_fn(y_hat, target_text_ids)
        return loss

    def training_step(self, batch: dict, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self._common_step(batch, batch_idx)
        self.log("val_loss", val_loss, prog_bar=True)
        # return the validation targets and predictions as a dict
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss = self._common_step(batch, batch_idx)
        self.log("test_loss", test_loss, prog_bar=True)
        # return the test targets and predictions as a dict
        return test_loss

    def configure_optimizers(self):
        # only train the `transformer.linear` layer
        trained_params = self.model.transformer.linear.parameters()
        return self.optimizer_cls(params=trained_params, lr=1e-3)
