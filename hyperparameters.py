import torch


def get_hyperparameters():
    return {
        'seed': 0,
        'lr': 1e-4,
        'num_epochs': 200,
        'dim_latent': 64,
        'node_features': 2,
        'edge_features': 2,
        'hidden_edge_features': 16,
        'hidden_edge_scores': 64,
        'num_gt_layers': 10,
        'nb_pos_enc': 0,
        'batch_size_train': 50,
        'batch_size_eval': 50,
        'num_decoding_paths': 150,
        'len_threshold': 20,
        'patience': 2,
        'decay': 0.95,
        'device': 'cuda:4' if torch.cuda.is_available() else 'cpu',
        # 'device': 'cpu',
        'batch_norm': True,
        'wandb_mode': 'disabled',  # switch between 'online' and 'disabled'

        'masking': True,
        'mask_frac_low': 80,
        'mask_frac_high': 100,
        'num_nodes_per_cluster': 25000,
        'npc_lower_bound': 1,
        'npc_upper_bound': 1,
        'alpha': 0.1,
        'k_extra_hops': 1,
        'type_pos_enc': 'none',
        'decode_with_labels': False,
        'strategy': 'greedy',
        'load_checkpoint': True,
        'num_threads': 32,

    }
