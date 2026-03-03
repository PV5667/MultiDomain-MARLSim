import torch
import torch.nn as nn

# defining the neural networks behind the policy
# using centralized training, decentralized execution (CTDE)


class PatchEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, embed_dim)
        )
    def forward(self, patch):
        out = self.net(patch)
        return out


# going to use transformer encoder for entity + event reasoning (take advantage of self-attention)
# use separate instances for events and entities
class ContextGNN(nn.Module):
    def __init__(self, context_dim, embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(context_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.Linear(embed_dim, embed_dim)

    def forward(self, context, mask=None):
        # entities: (B, N, dim)
        # mask: (B, N) -- says which are padding/not
        x = self.input_proj(context)
        x = self.transformer(x, src_key_padding_mask=mask)
        context_embs = x # need to keep these for the engage head (only entity enc)

        # global emb is avg of all entity emb's
        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()
            sum_emb = torch.sum(x * valid, dim=1)
            count = torch.clamp(torch.sum(valid, dim=1), min=1)
            global_emb = sum_emb / count
        else:
            global_emb = torch.mean(x, dim=1)
        global_emb = self.pool(global_emb)
        return context_embs, global_emb
    
# simple MLP for both comms and internal state encoding (using separate instances)
class StateMLP(nn.Module):
    def __init__(self, input_dim, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
    def forward(self, x):
        return self.net(x)


class PolicyHeads(nn.Module):
    def __init__(self, fused_dim, entity_emb_dim, n_directions=8, motion_range=10):
        """
        direction head: 8 directions + NOOP
        magnitude head: discrete from 1 to max range of movement
        deploy head: only for cUAS deployment, will add later
        engage binary head: engage pr not
        engage score head: scores entity embeds from GNN
        """
        super().__init__()
        self.move_dir = nn.Linear(fused_dim, n_directions + 1)
        self.move_mag = nn.Linear(fused_dim, motion_range)
        #self.deploy = nn.Linear(fused_dim, 2) going to add this later
        self.engage_binary = nn.Linear(fused_dim, 2)  # engage or not
        # Engage target: score each entity embedding
        self.engage_score = nn.Linear(fused_dim + entity_emb_dim, 1) # applied to entity_embs

    def forward(self, fused_emb, entity_embs):
        move_dir_logits = self.move_dir(fused_emb)
        move_mag_logits = self.move_mag(fused_emb)
        #deploy_logits = self.deploy(fused_emb)
        engage_logits = self.engage_binary(fused_emb)

        # Entity scoring for target selection, concat fused emb with each entity emb
        B, N, D = entity_embs.shape
        fused_exp = fused_emb.unsqueeze(1).expand(-1, N, -1) # expanding to be able concat
        combined = torch.cat([entity_embs, fused_exp], dim=-1)
        target_scores = self.engage_score(combined).squeeze(-1)

        return {
            "move_dir":   move_dir_logits,
            "move_mag":   move_mag_logits,
            #"deploy":     deploy_logits,
            "engage_bin": engage_logits,
            "engage_tgt": target_scores,
        }
    

class ActorAgent(nn.Module):
    def __init__(self, patch_in_channels, entity_dim, event_dim, comm_dim, state_dim, n_directions, motion_range):
        super().__init__()
        self.patch_enc = PatchEncoder(patch_in_channels, embed_dim=128)
        self.entity_gnn = ContextGNN(entity_dim, embed_dim=128)
        self.event_gnn = ContextGNN(event_dim, embed_dim=64)
        self.comms_in = StateMLP(comm_dim, embed_dim=64)
        self.state_mlp = StateMLP(state_dim, embed_dim=64)

        fused_dim = 128 + 128 + 64 + 64 + 64
        
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.comms_out = nn.Linear(256, comm_dim)
        self.heads = PolicyHeads(256, entity_emb_dim=128, n_directions=n_directions, motion_range=motion_range)

    def forward(self, obs):
        patch_emb = self.patch_enc(obs["patch"])
        entity_embs, entity_glob_emb = self.entity_gnn(obs["entities"], obs["entity_mask"])
        _, event_emb = self.event_gnn(obs["events"], obs["event_mask"])
        state_emb = self.state_mlp(obs["internal_state"])
        comms_emb = self.comms_in(obs["comms_in"])
        
        #print(f"Patch Emb: {patch_emb.shape}, Entity Glob Emb: {entity_glob_emb.shape}, Event Emb: {event_emb.shape}, State Emb: {state_emb.shape}, Comms Emb: {comms_emb.shape}")
        fused = torch.cat([patch_emb, entity_glob_emb, event_emb, state_emb, comms_emb], dim=-1)
        fused = self.fusion(fused)

        comms_out = self.comms_out(fused)
        action_dists = self.heads(fused, entity_embs)
        return action_dists, comms_out, fused
    

class CentralizedCritic(nn.Module):
    def __init__(self, agent_emb_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        print(agent_emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=agent_emb_dim,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.value_head = nn.Sequential(
            nn.Linear(agent_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, agent_embeddings, agent_mask=None):
        x = self.transformer(agent_embeddings, src_key_padding_mask=agent_mask, )
        value = self.value_head(x)
        return value