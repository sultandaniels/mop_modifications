from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from tensordict import TensorDict

import utils


class CnnKF(nn.Module):
    def __init__(self, batch_shape: Sequence[int], ny: int, ir_length: int, ridge: float = 0.0):
        super().__init__()

        self.batch_shape = batch_shape
        self.O_D = ny
        self.ir_length = ir_length

        self.observation_IR = nn.Parameter(torch.zeros(*self.batch_shape, self.O_D, self.ir_length, self.O_D))  # [B... x O_D x R x O_D]

        r_ = self.ir_length * self.O_D
        self.XTX = ridge * torch.eye(r_).expand(*batch_shape, r_, r_)                           # [B... x RO_D x RO_D]
        self.XTy = torch.zeros((*batch_shape, r_, self.O_D))                                    # [B... x RO_D x O_D]

        self.X = torch.zeros((*batch_shape, 0, r_))                                             # [B... x k x RO_D]
        self.y = torch.zeros((*batch_shape, 0, self.O_D))                                       # [B... x k x O_D]

    def analytical_error(self, systems: TensorDict[str, torch.Tensor]) -> torch.Tensor:
        # Variable definition
        def to_complex(t: torch.Tensor) -> torch.Tensor:
            return torch.complex(t, torch.zeros_like(t))

        Q = to_complex(self.observation_IR.to(torch.float32))                                   # [B... x O_D x R x O_D]
        Q = Q.permute(*range(Q.ndim - 3), -2, -1, -3)                                           # [B... x R x O_D x O_D]

        F = to_complex(systems['F'])                                                            # [B... x S_D x S_D]
        H = to_complex(systems['H'])                                                            # [B... x O_D x S_D]
        sqrt_S_W = to_complex(systems['sqrt_S_W'])                                              # [B... x S_D x S_D]
        sqrt_S_V = to_complex(systems['sqrt_S_V'])                                              # [B... x O_D x O_D]

        R = Q.shape[-3]

        L, V = torch.linalg.eig(F)                                                              # [B... x S_D], [B... x S_D x S_D]
        Vinv = torch.linalg.inv(V)                                                              # [B... x S_D x S_D]

        Hs = H @ V                                                                              # [B... x O_D x S_D]
        sqrt_S_Ws = Vinv @ sqrt_S_W                                                             # [B... x S_D x S_D]

        # State evolution noise error
        # Highlight
        ws_current_err = (Hs @ sqrt_S_Ws).norm(dim=(-2, -1)) ** 2                               # [B...]

        L_pow_series = L.unsqueeze(-2) ** torch.arange(1, R + 1)[:, None]                       # [B... x R x S_D]
        L_pow_series_inv = 1. / L_pow_series                                                    # [B... x R x S_D]

        QlHsLl = (Q @ Hs.unsqueeze(-3)) * L_pow_series_inv.unsqueeze(-2)                        # [B... x R x O_D x S_D]
        Hs_cumQlHsLl = Hs.unsqueeze(-3) - torch.cumsum(QlHsLl, dim=-3)                          # [B... x R x O_D x S_D]
        Hs_cumQlHsLl_Lk = Hs_cumQlHsLl * L_pow_series.unsqueeze(-2)                             # [B... x R x O_D x S_D]

        # Highlight
        ws_recent_err = (Hs_cumQlHsLl_Lk @ sqrt_S_Ws.unsqueeze(-3)).flatten(-3, -1).norm(dim=-1) ** 2   # [B...]

        Hs_cumQlHsLl_R = Hs_cumQlHsLl.index_select(-3, torch.tensor([R - 1])).squeeze(-3)       # [B... x O_D x S_D]
        cll = L.unsqueeze(-1) * L.unsqueeze(-2)                                                 # [B... x S_D x S_D]

        # Highlight
        _ws_geometric = (Hs_cumQlHsLl_R.mT @ Hs_cumQlHsLl_R) * ((cll ** (R + 1)) / (1 - cll))   # [B... x S_D x S_D]
        ws_geometric_err = utils.batch_trace(sqrt_S_Ws.mT @ _ws_geometric @ sqrt_S_Ws)          # [B...]

        # Observation noise error
        # Highlight
        v_current_err = sqrt_S_V.norm(dim=(-2, -1)) ** 2                                        # [B...]

        # Highlight
        # TODO: Backward pass on the above one breaks when Q = 0 for some reason
        # v_recent_err = (Q @ sqrt_S_V.unsqueeze(-3)).flatten(-3, -1).norm(dim=-1) ** 2           # [B...]
        v_recent_err = utils.batch_trace(sqrt_S_V.mT @ (Q.mT @ Q).sum(dim=-3) @ sqrt_S_V)       # [B...]

        err = ws_current_err + ws_recent_err + ws_geometric_err + v_current_err + v_recent_err  # [B...]
        return err.real

    """ forward
        :parameter {
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns {
            'observation_estimation': [B x L x O_D]
        }
    """
    def forward(self,
                context: torch.Tensor   # [B... x R x O_D]
    ) -> torch.Tensor:                  # [B... x O_D]
        flipped_context = context.transpose(-2, -1).unsqueeze(-1).unsqueeze(-4).flip(-2)  # [B... x 1 x O_D x R x 1]
        # [5, 2, 5] [1, 5, 2, 1]
        print("flipped_context", flipped_context.shape)
        print("self.observation_IR", self.observation_IR.shape)
        return torch.vmap(Fn.conv2d)(
            self.observation_IR.flatten(0, -4),                                 # [B x O_D x R x O_D]
            flipped_context.flatten(0, -5),                                     # [B x 1 x O_D x R x 1]
        ).unflatten(0, self.batch_shape).squeeze(-2, -3)                        # [B... x O_D]

    """ forward
        :parameter {
            'input': [B x L x I_D],
            'observation': [B x L x O_D]
        }
        :returns None
    """
    def update(self,
               context: torch.Tensor,   # [B... x R x O_D]
               target: torch.Tensor     # [B... x O_D]
    ) -> None:
        # DONE: Implement online least squares for memory efficiency
        flattened_X = context.flip(-2).flatten(-2, -1).unsqueeze(-2)                                # [B... x 1 x RO_D]
        flattened_observations = target.unsqueeze(-2)                                               # [B... x 1 x O_D]

        self.XTX = self.XTX + (flattened_X.mT @ flattened_X)
        self.XTy = self.XTy + (flattened_X.mT @ flattened_observations)

        if torch.all(torch.linalg.matrix_rank(self.XTX) >= self.XTX.shape[0]):
            flattened_w = torch.linalg.inv(self.XTX) @ self.XTy                                     # [B... x RO_D x O_D]
        else:
            self.X = torch.cat([self.X, flattened_X], dim=-2)                                       # [B... x k x RO_D]
            self.y = torch.cat([self.y, flattened_observations], dim=-2)                            # [B... x k x O_D]
            flattened_w = torch.linalg.pinv(self.X) @ self.y

        self.observation_IR.data = flattened_w.unflatten(-2, (self.ir_length, self.O_D)).transpose(-3, -2).to(torch.float32)    # [B... x O_D x R x O_D]