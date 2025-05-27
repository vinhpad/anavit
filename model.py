import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss
import torch.nn.functional as F
from mlp import MLP
from attn_unet import AttentionUNet

class DocREModel(nn.Module):

    def __init__(self,
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
        self.relation_extractor = nn.Linear(emb_size*block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.relation_extractor_feature_map = nn.Linear(2 * config.hidden_size, emb_size)
        self.cv_unet = AttentionUNet(emb_size, config.num_labels)
        self.cv_mlp = MLP(emb_size, emb_size // 2, config.num_labels, num_heads=4)

        self.bilinear = nn.Linear(config.num_labels * 3, config.num_labels)
        

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
        hss, tss, mss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts, mention_embs = [], [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att, m_emb = [], [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        m_emb.extend(e_emb)
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        m_emb.extend(e_emb)
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        m_emb.extend(e_emb)
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        m_emb.extend(e_emb)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
                mention_embs.extend(m_emb)

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
            mss.append(mention_embs)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss, mss
    

    def get_channel_map(self, sequence_output, attention, entity_pos, hts, mention_idx):
       
        hs, rs, ts, ms = self.get_hrt(sequence_output, attention, entity_pos, hts)
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        r1 = self.relation_extractor(bl)

        n = len(hts)
        map_rss = torch.zeros(n, 256, 256, self.emb_size, device=sequence_output.device)
        offset = 0
        for b_idx, ht in enumerate(hts):
            for pair_idx, (h_idx, t_idx) in enumerate(ht):
               
                for mention_h_pos in entity_pos[b_idx][h_idx]:
                    for mention_t_pos in entity_pos[b_idx][t_idx]:

                        mention_h_idx = mention_idx[mention_h_pos[0]]
                        mention_t_idx = mention_idx[mention_t_pos[0]]

                        map_rss[b_idx, mention_h_idx, mention_t_idx] = torch.cat([ms[b_idx][mention_h_idx], ms[b_idx][mention_t_idx]], dim=-1)
            offset += len(ht)
        
        map_rss = self.relation_extractor_feature_map(map_rss)


        map_rss_enhance_unet = self.unet(map_rss)
        r2 = self.get_ht(map_rss_enhance_unet)

        map_rss_enhance_mlp = self.mlp(map_rss)
        r3 = self.get_ht(map_rss_enhance_mlp)

        return r1, r2, r3
    
    def get_ht(self, rel_enco, hts, mention_idx):
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for (h_index, t_index) in ht_index:
                mts_embs = []
                for mention_h_idx in mention_idx[i][h_index]:
                    for mention_t_idx in mention_idx[i][t_index]:
                        mts_embs.append(rel_enco[mention_h_idx][mention_t_idx])
                htss.append(torch.logsumexp(torch.stack(mts_embs, dim=0), dim=0))
        htss = torch.stack(htss, dim=0)
        return htss
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                mention_idx=None):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        r1, r2, r3 = self.get_channel_map(sequence_output, attention, entity_pos, hts, mention_idx)
        
        logits = self.bilinear(torch.cat([r1, r2, r3], dim = -1))

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels), )
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output), ) + output
        return output

