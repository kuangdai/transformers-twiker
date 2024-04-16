# if __name__ == "__main__":
#     torch.manual_seed(0)
#     # data
#     V, B, H, N, F, K = 7, 8, 11, 9, 13, 7
#     ker_uni_k = torch.randn(K)
#     ker_uni_v = torch.randn(K)
#     k = torch.randn(B, H, N, F)
#     v = torch.randn(B, H, N, F)
#
#     # by TwikerModel
#     ker_k = ker_uni_k.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, K)
#     ker_v = ker_uni_v.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, N, H, K)
#     ker = torch.cat((ker_k, ker_v), dim=2)
#     emb = TwikerModel(V, K, H, 1, casual_handling="shrink_near_boundary")
#     k1, v1, _, _ = emb.conv_key_value(k, v, ker, for_casual=False)
#
#     # by conv
#     ker_kc = ker_uni_k.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(H, 1, K, 1)
#     ker_vc = ker_uni_v.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(H, 1, K, 1)
#     k2 = torch.nn.functional.conv2d(k, ker_kc, padding='same', groups=H)
#     v2 = torch.nn.functional.conv2d(v, ker_vc, padding='same', groups=H)
#     print((k1 - k2).abs().max())
#     print((v1 - v2).abs().max())
#
#     # by TwikerModel
#     k1, v1, ck, cv = emb.conv_key_value(k, v, ker, for_casual=True)
#     print((k1 - k2).abs().max())
#     print((v1 - v2).abs().max())
#
#     # masked
#     pp = K // 2
#     for i in range(1, pp + 1):
#         ker_uni_k[-i:] = 0
#         ker_uni_v[-i:] = 0
#         ker_uni_k[:i] = 0
#         ker_uni_v[:i] = 0
#         ker_kc = ker_uni_k.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(H, 1, K, 1)
#         ker_vc = ker_uni_v.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(H, 1, K, 1)
#         k3 = torch.nn.functional.conv2d(k, ker_kc, padding='same', groups=H)
#         v3 = torch.nn.functional.conv2d(v, ker_vc, padding='same', groups=H)
#         print(i)
#         print((ck[..., i - 1] - k3).abs().max())
#         print((cv[..., i - 1] - v3).abs().max())
#     print("************")
#
#     # correct attn_weights
#     query_ = torch.randn(B, H, N, F)
#     past = 2
#     key_ = torch.cat((torch.randn(B, H, past, F), k1), dim=2)
#     value_ = torch.cat((torch.randn(B, H, past, F), v1), dim=2)
#     att_w = torch.matmul(query_, key_.transpose(-1, -2))
#
#     # manual correction
#     att_w_manual = att_w.clone()
#     for i_query in range(N):
#         for i_dist, i_key in enumerate(range(i_query + 1 - pp, i_query + 1)):
#             if i_key < 0:
#                 continue
#             q = query_[:, :, i_query:i_query + 1, :]
#             k = ck[:, :, i_key:i_key + 1, :, i_dist]
#             att_w_manual[:, :, i_query, past + i_key] = torch.matmul(q, k.transpose(-1, -2)).squeeze()
#
#     # fast correction
#     att_w_fast = emb.correct_attn_weights_near_casual_boundary(att_w, query_, key_, ck)
#     print((att_w_manual - att_w_fast).abs().max())
#
#     # mask
#     for b in range(B):
#         for h in range(H):
#             att_w_fast[b, h, :, past:] *= torch.ones(N, N).tril()
#
#     # dot with value
#     att_o = torch.matmul(att_w_fast, value_)
#
#     # manual correction
#     att_o_manual = att_o.clone()
#     for i_query in range(N):
#         for i_dist, i_value in enumerate(range(i_query + 1 - pp, i_query + 1)):
#             if i_value < 0:
#                 continue
#             value_right = cv[:, :, i_value, :, i_dist]
#             value_wrong = value_[:, :, past + i_value, :]
#             value_diff = value_right - value_wrong
#             w = att_w_fast[:, :, i_query, past + i_value]
#             att_o_manual[:, :, i_query, :] += w[:, :, None] * value_diff
#
#     # fast correction
#     att_o_fast = emb.correct_attn_output_near_casual_boundary(att_o, att_w_fast, value_, cv)
#     print((att_o_manual - att_o_fast).abs().max())
#     print("************")
#     print("************")
#
#     kernel_k = torch.randn(B, N, H, K)
#     kernel_v = torch.randn(B, N, H, K)
#     key_ = torch.randn(B, H, N, F)
#     value_ = torch.randn(B, H, N, F)
#
#     conv_key = torch.zeros(B, H, N, F)
#     conv_value = torch.zeros(B, H, N, F)
#     for b in range(B):
#         for h in range(H):
#             for n in range(N):
#                 for k in range(-pp, pp + 1):
#                     if n + k < 0 or n + k >= N:
#                         continue
#                     conv_key[b, h, n] += kernel_k[b, n, h, k + pp] * key_[b, h, n + k]
#                     conv_value[b, h, n] += kernel_v[b, n, h, k + pp] * value_[b, h, n + k]
#
#     conv_key1, conv_value1, cb_key1, cb_value1 = emb.conv_key_value(
#         key_, value_, torch.cat((kernel_k, kernel_v), dim=2), for_casual=True)
#     print((conv_key1 - conv_key).max().abs())
#     print((conv_value1 - conv_value).max().abs())
#     print("************")
#     print("************")
#     print("************")
#
#     query_ = torch.randn(B, H, N, F)
#     att_w = torch.zeros(B, H, N, N)
#     att_w_cb = torch.zeros(B, H, N, N)
#     for b in range(B):
#         for h in range(H):
#             for n in range(N):
#                 for nn in range(N):
#                     for k in range(-pp, pp + 1):
#                         if 0 <= nn + k < N:
#                             att_w[b, h, n, nn] += (query_[b, h, n] * kernel_k[b, nn, h, k + pp] *
#                                                    key_[b, h, nn + k]).sum()
#                         if nn <= n:
#                             if max(nn * 2 - n, 0) <= nn + k <= n:
#                                 att_w_cb[b, h, n, nn] += (query_[b, h, n] * kernel_k[b, nn, h, k + pp] *
#                                                           key_[b, h, nn + k]).sum()
#                         else:
#                             if 0 <= nn + k < N:
#                                 # correction not considered for nn > n
#                                 att_w_cb[b, h, n, nn] += (query_[b, h, n] * kernel_k[b, nn, h, k + pp] *
#                                                           key_[b, h, nn + k]).sum()
#
#     att_w1 = torch.matmul(query_, conv_key1.transpose(-1, -2))
#     att_w1_cb = emb.correct_attn_weights_near_casual_boundary(att_w1, query_, conv_key1,
#                                                               cb_key1)
#     print((att_w1 - att_w).max().abs())
#     print((att_w1_cb - att_w_cb).max().abs())
#
#     att_o = torch.zeros(B, H, N, F)
#     att_o_cb = torch.zeros(B, H, N, F)
#     for b in range(B):
#         for h in range(H):
#             for n in range(N):
#                 for nn in range(N):
#                     for k in range(-pp, pp + 1):
#                         if 0 <= nn + k < N:
#                             att_o[b, h, n, :] += (att_w[b, h, n, nn] * kernel_v[b, nn, h, k + pp] *
#                                                   value_[b, h, nn + k])
#                         if nn <= n:
#                             if max(nn * 2 - n, 0) <= nn + k <= n:
#                                 att_o_cb[b, h, n, :] += (att_w[b, h, n, nn] * kernel_v[b, nn, h, k + pp] *
#                                                          value_[b, h, nn + k])
#                         else:
#                             if 0 <= nn + k < N:
#                                 # correction not considered for nn > n
#                                 att_o_cb[b, h, n, :] += (att_w[b, h, n, nn] * kernel_v[b, nn, h, k + pp] *
#                                                          value_[b, h, nn + k])
#
#     att_o1 = torch.matmul(att_w1, conv_value1)
#     att_o1_cb = emb.correct_attn_output_near_casual_boundary(att_o1, att_w1, conv_value1, cb_value1)
#     print((att_o1 - att_o).max().abs())
#     print((att_o1_cb - att_o_cb).max().abs())
