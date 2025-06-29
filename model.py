import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
import torch.nn.functional as F

from networks.swinunet.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

class DocREModel(nn.Module):

    def __init__(self,
                 args,
                 config,
                 model,
                 emb_size=768,
                 block_size=64,
                 num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        
        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.relation_extractor = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 3),
        )

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        # self.mlp = MLP(in_channels=emb_size, out_channels=512, hidden_dim=256, num_heads=4)
        self.bilinear = nn.Linear(256, config.num_labels)

        self.segmentation_net = SwinTransformerSys(
            num_classes=256,
        )

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(
            self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"
                                                       ] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0),
                                                dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(
                            self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(
                            self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss
    

    def get_channel_map(self, sequence_output, attention, entity_pos, hts):
        batch_size = len(hts)
        map_rss = torch.zeros(batch_size, 224, 224, 3, device=sequence_output.device)

        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))

        bl = torch.cat([hs, ts], dim=1)
        bl = self.relation_extractor(bl)
        offset = 0
        for b_idx, ht in enumerate(hts):
            for pair_idx, (h_idx, t_idx) in enumerate(ht):
                map_rss[b_idx, h_idx, t_idx] = bl[offset + pair_idx]
            offset += len(ht)

        return map_rss
    
    def get_ht(self, rel_enco, hts):
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for (h_index, t_index) in ht_index:
                htss.append(rel_enco[i,h_index,t_index])
        htss = torch.stack(htss,dim=0)
        return htss
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        map_rss = self.get_channel_map(sequence_output, attention, entity_pos, hts)
        print("map_rss shape:", map_rss.shape)
        map_rss = map_rss.permute(0, 3, 1, 2)
        
        feature_map = self.segmentation_net(map_rss)
        feature_map = feature_map.permute(0, 2, 3, 1)  # [batch_size, 42, 42, 96]

        bl = self.get_ht(feature_map, hts)
        
        logits = self.bilinear(bl)

        output = (self.loss_fnt.get_label(logits,
                                          num_labels=self.num_labels), )
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]

            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output), ) + output
        return output

