import torch
import torch.nn as nn
import torch.optim as optim
from transformers import TransfoXLConfig, TransfoXLModel
from models import BaseModel

class TransformerXL(BaseModel):
    def __init__(self, config):
        super(TransformerXL, self).__init__()  # Initialize the parent class
        self.config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Transformer-XL model from scratch
        model_config = TransfoXLConfig(
            d_model=config.d_model,
            n_head=config.n_head,
            n_layer=config.n_layer,
            n_positions=config.n_positions
        )
        self.model = TransfoXLModel(model_config).to(self._device)
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    # def train(self, train_dataloader, val_dataloader=None):
    #     self.model.train()
    #     for epoch in range(self.config.epochs):
    #         for batch in train_dataloader:
    #             inputs, labels = batch
    #             inputs = inputs.to(self._device)
    #             labels = labels.to(self._device)
                
    #             self.optimizer.zero_grad()
    #             outputs = self.model(inputs)
    #             logits = outputs.last_hidden_state
    #             loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    #             loss.backward()
    #             self.optimizer.step()
                
    #             print(f"Epoch {epoch}, Loss: {loss.item()}")

    #         if val_dataloader:
    #             self.evaluate(val_dataloader)

    def forward(self, inputs):
        """
        Forward pass for the TransformerXL model.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        inputs = inputs.to(self._device)
        outputs = self.model(inputs)
        return outputs.last_hidden_state

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                
                outputs = self.model(inputs)
                logits = outputs.last_hidden_state
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Validation Loss: {avg_loss}")

    def calculate_losses_and_metrics(self, input_dict, intermediate_dict):
        ys = input_dict["ys"]
        preds = intermediate_dict["preds"]
        if self.config.dataset_typ == "pendulum":
            ys[..., 0] = ys[..., 0] % (2 * torch.pi)
            preds[..., 0] = preds[..., 0] % (2 * torch.pi)
            diff = torch.abs(ys - preds)
            diff[..., 0] = torch.where(diff[..., 0] > torch.pi, diff[..., 0] - 2 * torch.pi, diff[..., 0])
            res_sq = diff ** 2
        else:
            res_sq = (preds - ys) ** 2

        output_dict = {"loss_mse": torch.mean(res_sq)}
        for i in range(ys.shape[1]):
            for j in range(ys.shape[2]):
                output_dict[f"metric_mse_ts{i}_dim_{j}"] = torch.mean(res_sq[:, i, j])

        return output_dict

    def predict_ar(self, ins, fix_window_len=True):
        ins = torch.from_numpy(ins).float().to(self._device)
        one_d = False
        if ins.ndim == 2:
            one_d = True
            ins = ins.unsqueeze(0)
        bsize, points, _ = ins.shape
        d_o = self.config.n_dims_out
        outs = torch.zeros(bsize, 1, d_o).to(self._device)
        with torch.no_grad():
            for i in range(1, points + 1):
                I = ins[:, :i]
                if fix_window_len and I.shape[1] > self.config.n_positions:
                    I = I[:, -self.config.n_positions:]
                _, interm = self.predict_step({"xs": I})
                pred = interm["preds"][:, -1:]  # b, 1, d_o
                outs = torch.cat([outs, pred], dim=1)
        outs = outs.detach().cpu().numpy()
        if one_d:
            outs = outs[0]
        return outs

    def predict_step(self, input_dict):
        xs = input_dict["xs"].to(self._device)
        outputs = self.model(xs)
        preds = outputs.last_hidden_state
        intermediate_dict = {"preds": preds}
        return intermediate_dict, intermediate_dict

# Example configuration
# class Config:
#     learning_rate = 5e-5
#     epochs = 3
#     dataset_typ = "pendulum"
#     n_dims_out = 10
#     n_positions = 512
#     d_model = 512
#     n_head = 8
#     d_inner = 2048
#     n_layer = 12
#     cutoffs = [20000, 40000, 200000]
#     div_val = 4
#     mem_len = 1600
#     same_length = True
#     clamp_len = 1000

# config = Config()
# trainer = TransformerXLTrainer(config)