import torch
import torch.nn as nn
import random
import copy
from graph_walker import Attention

class Explicit_Recommender(nn.Module):
    def __init__(self,utterance_embed_size=64,graph_embed_size=64,bow_size=512,atten_hidden=20):
        super(Explicit_Recommender,self).__init__()
        self.cand_embed_size=graph_embed_size+bow_size
        self.graph_embed_size=graph_embed_size
        self.utterance_embed_size=utterance_embed_size
        self.attention=Attention(self.utterance_embed_size,self.cand_embed_size,atten_hidden)
        self.Wr=nn.Linear(self.cand_embed_size,self.cand_embed_size)



    def forward(self,utterance_embed,graph_embedding,user_portrait,bow_embedding,rec_index,rec_batch_index):
        graph_embed=graph_embedding.index_select(0,rec_index)
        bow_embed=bow_embedding.index_select(0,rec_index)
        cand_embed=torch.cat([graph_embed,bow_embed],dim=-1)
        context=utterance_embed
        tiled_context=self.tile_context(context)
        attended=self.Wr(self.attention.forward(tiled_context,cand_embed,cand_embed,rec_batch_index))
        tiled_attended=self.tile_context(attended)
        score=torch.sum(tiled_attended*cand_embed,dim=-1).view(-1,5)
        return score

    def tile_context(self,context):
        batch_size=context.size()[0]
        tile_ctx=[]
        for i in range(batch_size):
            repeated=context[i].repeat(5,1)
            tile_ctx.append(repeated)
        tile_ctx=torch.cat(tile_ctx,dim=0)
        return tile_ctx
    

    def prepare_data(self,rec_cand,device):
        sel_indices=[]
        sel_batch_indices=[]
        golden=[]
        sel=[]
        bat=[]
        for i,item in enumerate(rec_cand):
            correct=item[0]
            item_copy=copy.deepcopy(item)
            random.shuffle(item_copy)
            golden.append(item_copy.index(correct))
            for k in item_copy:
                sel.append(k)
                bat.append(i)
        rec_index=torch.Tensor(sel).long().to(device=device)
        rec_batch_index=torch.Tensor(bat).long().to(device=device)
        golden_rec=torch.Tensor(golden).long().to(device=device)
        return rec_index,rec_batch_index,golden_rec