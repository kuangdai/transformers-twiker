"""
twiker model
"""

import math

import torch
import torch.nn as nn


class TwikerEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                 sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
        """ avoid random initialization for twiker Embedding """
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                         scale_grad_by_freq, sparse, _weight, _freeze, device, dtype)


class TwikerModel(nn.Module):
    def __init__(self, vocab_size: int, kernel_size: int, n_head: int, n_layer: int,
                 sum_to_one: bool = False, head_invariant: bool = True,
                 layer_invariant: bool = True, strict_on_casual: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.n_layer = n_layer
        self.sum_to_one = sum_to_one
        self.head_invariant = head_invariant
        self.layer_invariant = layer_invariant
        self.strict_on_casual = strict_on_casual

        # embedding layer
        weight_shape = [vocab_size, n_layer, 2, n_head, kernel_size]
        if head_invariant:
            weight_shape[3] = 1
        if layer_invariant:
            weight_shape[1] = 1
        self.embedding = TwikerEmbedding(vocab_size, math.prod(weight_shape[1:]))

        # initialize weights: 00100
        self.embedding.weight.data.fill_(0.)
        self.embedding.weight.data.view(weight_shape)[..., self.kernel_size // 2].fill_(1.)

        # prepare casual mask
        p = self.kernel_size // 2
        self.casual_mask = torch.ones(p + 1, self.kernel_size)
        for i in range(1, p + 1):
            self.casual_mask[i, (self.kernel_size - i):].zero_()

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

        # merge kv and head dimensions
        kernel = kernel.reshape(n_batch, n_token, self.n_layer, 2 * self.n_head, self.kernel_size)

        # sum to one
        if self.sum_to_one:
            kernel = kernel / kernel.sum(dim=-1)[:, :, :, :, None]
        return kernel

    def conv_key_value(self, key: torch.Tensor, value: torch.Tensor, kernel: torch.Tensor,
                       for_casual: bool = False):
        # shape of key, value: (B, H, N, F)
        # shape of kernel: (B, N, 2 * H, K)
        n_batch, n_head, n_token, n_feat = key.shape

        # merge kv: (B, 2 * H, N, F)
        kv = torch.cat((key, value), dim=1)
        # unfold: (B, 2 * H * K, N * F)
        kv = torch.nn.functional.unfold(kv, kernel_size=(self.kernel_size, 1),
                                        padding=(self.kernel_size // 2, 0))
        # reshape: (B, 2 * H, K, N, F)
        kv = kv.reshape(n_batch, 2 * n_head, self.kernel_size, n_token, n_feat)

        # casual
        p = self.kernel_size // 2
        if self.strict_on_casual and for_casual:
            # make copies of kernel: (B, N, 2 * H, p + 1, K)
            kernel = kernel.unsqueeze(-2).expand(-1, -1, -1, p + 1, -1)
            # mask future ones
            kernel = kernel * self.casual_mask.to(kernel.device)[None, None, None]
            # merge 2H and p: (B, N, 2 * H * (p + 1), K)
            kernel = kernel.flatten(2, 3)
            # make copies of kv: (B, 2 * H, K, N, F) => (B, 2 * H * (p + 1), K, N, F)
            kv = kv.unsqueeze(2).expand(-1, -1, p + 1, -1, -1, -1).flatten(1, 2)

        # mm
        kernel = kernel.to(dtype=kv.dtype, device=kv.device)
        kv = torch.einsum('BGkNF,BNGk->BGNF', kv, kernel)  # noqa

        # fold
        kv = kv.reshape(kv.size(0), kv.size(1), -1)
        kv = torch.nn.functional.fold(kv, output_size=(n_token, n_feat), kernel_size=(1, 1))

        # rearrange output
        if self.strict_on_casual and for_casual:
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
                                                  key: torch.Tensor,
                                                  casual_boundary_keys: torch.Tensor):
        n_past = key.size(-2) - query.size(-2)
        p = casual_boundary_keys.size(-1)
        for i in range(0, p):
            cb_key = casual_boundary_keys[..., i]  # 11110, 11100
            offset = p - i - 1
            correct = torch.einsum("BHNF,BHNF->BHN",
                                   query[:, :, offset:, :],
                                   cb_key[:, :, :cb_key.size(2) - offset, :])
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
        for i in range(0, p):
            cb_value = casual_boundary_values[..., i]  # 11110, 11100
            offset = p - i - 1
            diag_att_w = attn_weights[..., n_past:].diagonal(offset=-offset, dim1=2, dim2=3)
            diff_value = (cb_value[:, :, :cb_value.size(2) - offset, :]
                          - value[:, :, n_past:value.size(2) - offset, :])
            diff = torch.einsum("BHN,BHNF->BHNF", diag_att_w, diff_value)
            diff = torch.nn.functional.pad(diff, pad=(0, 0, offset, 0))
            attn_output = attn_output + diff
        return attn_output


if __name__ == "__main__":
    torch.manual_seed(0)
    # data
    V, B, H, N, F, K = 7, 8, 11, 9, 13, 5
    ker_uni_k = torch.randn(K)
    ker_uni_v = torch.randn(K)
    k = torch.randn(B, H, N, F)
    v = torch.randn(B, H, N, F)

    # by TwikerModel
    ker_k = ker_uni_k.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, K)
    ker_v = ker_uni_v.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, K)
    ker = torch.cat((ker_k, ker_v), dim=2)
    emb = TwikerModel(V, K, 1, 1)
    k1, v1, _, _ = emb.conv_key_value(k, v, ker, for_casual=False)

    # by conv
    ker_kc = ker_uni_k.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(H, 1, K, 1)
    ker_vc = ker_uni_v.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(H, 1, K, 1)
    k2 = torch.nn.functional.conv2d(k, ker_kc, padding='same', groups=H)
    v2 = torch.nn.functional.conv2d(v, ker_vc, padding='same', groups=H)
    print((k1 - k2).abs().max())
    print((v1 - v2).abs().max())

    # by TwikerModel
    k1, v1, ck, cv = emb.conv_key_value(k, v, ker, for_casual=True)
    print((k1 - k2).abs().max())
    print((v1 - v2).abs().max())

    # masked
    ker_uni_k[-1:] = 0
    ker_uni_v[-1:] = 0
    ker_kc = ker_uni_k.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(H, 1, K, 1)
    ker_vc = ker_uni_v.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(H, 1, K, 1)
    k3 = torch.nn.functional.conv2d(k, ker_kc, padding='same', groups=H)
    v3 = torch.nn.functional.conv2d(v, ker_vc, padding='same', groups=H)
    print((ck[..., 0] - k3).abs().max())
    print((cv[..., 0] - v3).abs().max())

    ker_uni_k[-2:] = 0
    ker_uni_v[-2:] = 0
    ker_kc = ker_uni_k.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(H, 1, K, 1)
    ker_vc = ker_uni_v.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(H, 1, K, 1)
    k3 = torch.nn.functional.conv2d(k, ker_kc, padding='same', groups=H)
    v3 = torch.nn.functional.conv2d(v, ker_vc, padding='same', groups=H)
    print((ck[..., 1] - k3).abs().max())
    print((cv[..., 1] - v3).abs().max())
    print("************")

    # correct attn_weights
    query_ = torch.randn(B, H, N, F)
    past = 2
    key_ = torch.cat((torch.randn(B, H, past, F), k1), dim=2)
    value_ = torch.cat((torch.randn(B, H, past, F), v1), dim=2)
    att_w = torch.matmul(query_, key_.transpose(-1, -2))

    # manual correction
    att_w_manual = att_w.clone()
    pp = K // 2
    for i_query in range(N):
        for i_dist, i_key in enumerate(range(i_query + 1 - pp, i_query + 1)):
            if i_key < 0:
                continue
            q = query_[:, :, i_query:i_query + 1, :]
            k = ck[:, :, i_key:i_key + 1, :, i_dist]
            att_w_manual[:, :, i_query, past + i_key] = torch.matmul(q, k.transpose(-1, -2)).squeeze()

    # fast correction
    att_w_fast = emb.correct_attn_weights_near_casual_boundary(att_w, query_, key_, ck)
    print((att_w_manual - att_w_fast).abs().max())

    # mask
    for b in range(B):
        for h in range(H):
            att_w_fast[b, h, :, past:] *= torch.ones(N, N).tril()

    # dot with value
    att_o = torch.matmul(att_w_fast, value_)

    # manual correction
    att_o_manual = att_o.clone()
    for i_query in range(N):
        for i_dist, i_value in enumerate(range(i_query + 1 - pp, i_query + 1)):
            if i_value < 0:
                continue
            value_right = cv[:, :, i_value, :, i_dist]
            value_wrong = value_[:, :, past + i_value, :]
            value_diff = value_right - value_wrong
            w = att_w_fast[:, :, i_query, past + i_value]
            att_o_manual[:, :, i_query, :] += w[:, :, None] * value_diff

    # fast correction
    att_o_fast = emb.correct_attn_output_near_casual_boundary(att_o, att_w_fast, value_, cv)
    print((att_o_manual - att_o_fast).abs().max())
