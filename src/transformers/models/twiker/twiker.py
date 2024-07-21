"""
twiker model
"""

import math

import torch
import torch.nn as nn


class TwikerEmbedding(nn.Embedding):
    """ a wrapper of `nn.Embedding` to avoid random initialization of twiker """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                 sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                         scale_grad_by_freq, sparse, _weight, _freeze, device, dtype)


class TwikerModel(nn.Module):
    def __init__(self, vocab_size: int, kernel_size: int, n_head: int, n_layer: int,
                 to_be_convolved: str = "v",
                 softmax: bool = True, temperature: float = 1.0,
                 head_invariant: bool = True, layer_invariant: bool = True,
                 casual_handling: str = "shrink_near_boundary", ):
        super().__init__()
        self.vocab_size = vocab_size
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.n_layer = n_layer
        self.to_be_convolved = to_be_convolved
        self.softmax = softmax
        self.temperature = temperature
        self.head_invariant = head_invariant
        self.layer_invariant = layer_invariant
        self.casual_handling = casual_handling

        # verify arguments
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        assert to_be_convolved in ["kv", "k", "v"], \
            f"Unknown value for `to_be_convolved`: {to_be_convolved}."
        assert casual_handling in [
            "none",  # do not handle
            "only_left_half",  # always mask kernels by 11100
            "truncate_near_boundary",  # mask kernels by 11110, 11100 near boundary
            "shrink_near_boundary",  # mask kernels by 01110, 00100 near boundary
        ], f"Unknown value for `casual_handling`: {casual_handling}."

        # embedding layer
        weight_shape = [vocab_size, n_layer, 2, n_head, kernel_size]
        if head_invariant:
            weight_shape[3] = 1
        if layer_invariant:
            weight_shape[1] = 1
        self.embedding = TwikerEmbedding(vocab_size, math.prod(weight_shape[1:]))

        # initialize weights: 00100
        p = self.kernel_size // 2
        self.embedding.weight.data.fill_(0.)
        if self.softmax:
            self.embedding.weight.data.view(weight_shape)[..., p].fill_(10. * self.temperature)
        else:
            self.embedding.weight.data.view(weight_shape)[..., p].fill_(1.)

        # prepare casual mask
        if casual_handling == "none":
            self.casual_mask = None
        elif casual_handling == "only_left_half":
            self.casual_mask = torch.ones(self.kernel_size)
            self.casual_mask[p + 1:].zero_()
        else:
            self.casual_mask = torch.ones(p + 1, self.kernel_size)
            for i_mask in range(1, p + 1):
                self.casual_mask[i_mask, -i_mask:].zero_()  # 11111 => 11110
                if casual_handling == "shrink_near_boundary":
                    self.casual_mask[i_mask, :i_mask].zero_()  # 11110 => 01110

    def get_kernel(self, input_ids: torch.LongTensor):
        kernel = self.embedding(input_ids)

        # broadcast along head and layer
        n_batch, n_token = kernel.shape[:2]
        wanted_shape = [n_batch, n_token, self.n_layer, 2, self.n_head, self.kernel_size]
        actual_shape = [n_batch, n_token, self.n_layer, 2, self.n_head, self.kernel_size]
        if self.head_invariant:
            actual_shape[4] = 1
        if self.layer_invariant:
            actual_shape[2] = 1
        kernel = kernel.reshape(actual_shape).expand(wanted_shape)

        # handle kv, k, v
        t00100 = torch.zeros((n_batch, n_token, self.n_layer, 1, self.n_head, self.kernel_size),
                             device=kernel.device)
        if self.softmax:
            t00100[..., self.kernel_size // 2] = 10. * self.temperature
        else:
            t00100[..., self.kernel_size // 2] = 1.
        if self.to_be_convolved == "k":
            kernel = torch.cat((kernel[:, :, :, 0:1, :, :], t00100), dim=3)
        elif self.to_be_convolved == "v":
            kernel = torch.cat((t00100, kernel[:, :, :, 1:2, :, :]), dim=3)
        else:
            assert self.to_be_convolved == "kv"

        # merge kv and head dimensions
        kernel = kernel.reshape(n_batch, n_token, self.n_layer, 2 * self.n_head, self.kernel_size)
        return kernel

    def conv_key_value(self, key: torch.Tensor, value: torch.Tensor, kernel: torch.Tensor,
                       for_casual: bool = False):
        # shape of key, value: (B, H, N, F)
        # shape of kernel: (B, N, 2 * H, K)
        n_batch, n_head, n_token, n_feat = key.shape

        # merge kv: (B, 2 * H, N, F)
        kv = torch.cat((key, value), dim=1)
        # unfold: (B, 2 * H * K, N * F)
        p = self.kernel_size // 2
        kv = torch.nn.functional.unfold(kv, kernel_size=(self.kernel_size, 1), padding=(p, 0))
        # reshape: (B, 2 * H, K, N, F)
        kv = kv.reshape(n_batch, 2 * n_head, self.kernel_size, n_token, n_feat)

        # casual handling
        if for_casual:
            if self.casual_handling == "only_left_half":
                kernel = kernel * self.casual_mask.to(
                    dtype=kernel.dtype, device=kernel.device)[None, None, None, :]
            elif self.casual_handling in ["truncate_near_boundary", "shrink_near_boundary"]:
                # make copies of kernel: (B, N, 2 * H, p + 1, K)
                kernel = kernel.unsqueeze(-2).expand(-1, -1, -1, p + 1, -1)
                # apply masking
                kernel = kernel * self.casual_mask.to(
                    dtype=kernel.dtype, device=kernel.device)[None, None, None, :, :]
                # merge 2H and p to channel: (B, N, 2 * H * (p + 1), K)
                kernel = kernel.flatten(2, 3)
                # make copies of kv: (B, 2 * H, K, N, F) => (B, 2 * H * (p + 1), K, N, F)
                kv = kv.unsqueeze(2).expand(-1, -1, p + 1, -1, -1, -1).flatten(1, 2)
            else:
                pass  # self.casual_handling == "none"

        # softmax
        if self.softmax:
            kernel = torch.softmax(kernel / self.temperature, dim=-1)

        # mm
        kernel = kernel.to(dtype=kv.dtype, device=kv.device)
        kv = torch.einsum('BGkNF,BNGk->BGNF', kv, kernel)

        # fold
        kv = kv.reshape(kv.size(0), kv.size(1), -1)
        kv = torch.nn.functional.fold(kv, output_size=(n_token, n_feat), kernel_size=(1, 1))

        # rearrange output
        if for_casual and self.casual_handling in ["truncate_near_boundary", "shrink_near_boundary"]:
            kv = kv.reshape(n_batch, 2 * n_head, p + 1, n_token, n_feat).movedim(2, -1)
            key, value = kv[..., 0].split(n_head, dim=1)
            casual_boundary_keys, casual_boundary_values = kv[..., 1:].split(n_head, dim=1)
            return key, value, casual_boundary_keys, casual_boundary_values
        else:
            key, value = kv.split(n_head, dim=1)
            return key, value, None, None

    @staticmethod
    def correct_attn_weights_near_casual_boundary(attn_weights: torch.Tensor,
                                                  query: torch.Tensor,
                                                  casual_boundary_keys: torch.Tensor):
        n_past = attn_weights.size(-1) - attn_weights.size(-2)
        p = casual_boundary_keys.size(-1)
        for i_mask in range(0, p):
            offset = p - i_mask - 1
            cb_key = casual_boundary_keys[..., i_mask]  # 01110, 00100
            correct = torch.einsum("BHNF,BHNF->BHN",
                                   query[:, :, offset:, :],
                                   cb_key[:, :, :cb_key.size(2) - offset, :])
            if n_past == 0:
                attn_weights = attn_weights.diagonal_scatter(
                    correct, offset=-offset, dim1=2, dim2=3)
            else:
                attn_weights_square = attn_weights[..., n_past:].diagonal_scatter(
                    correct, offset=-offset, dim1=2, dim2=3)
                attn_weights = torch.cat((attn_weights[..., :n_past],
                                          attn_weights_square), dim=-1)
        return attn_weights

    @staticmethod
    def correct_attn_output_near_casual_boundary(attn_output: torch.Tensor,
                                                 attn_weights: torch.Tensor,
                                                 value: torch.Tensor,
                                                 casual_boundary_values: torch.Tensor):
        n_past = attn_weights.size(-1) - attn_weights.size(-2)
        p = casual_boundary_values.size(-1)
        for i_mask in range(0, p):
            offset = p - i_mask - 1
            cb_value = casual_boundary_values[..., i_mask]  # 01110, 00100
            diag_att_w = attn_weights[..., n_past:].diagonal(offset=-offset, dim1=2, dim2=3)
            diff_value = (cb_value[:, :, :cb_value.size(2) - offset, :]
                          - value[:, :, n_past:value.size(2) - offset, :])
            diff = torch.einsum("BHN,BHNF->BHNF", diag_att_w, diff_value)
            diff = torch.nn.functional.pad(diff, pad=(0, 0, offset, 0))
            attn_output = attn_output + diff
        return attn_output
