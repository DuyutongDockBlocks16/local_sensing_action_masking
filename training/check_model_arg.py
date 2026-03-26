if __name__ == "__main__":
    import torch
    
    model_path = "../results/MyEnv/MyEnv/mappo/check/run1/models/actor.pt"
    state_dict = torch.load(model_path, map_location='cpu')
    
    for name, param in state_dict.items():
        print(f"{name}: {param.shape}")
    
    actor_layers = [name for name in state_dict.keys() if 'actor' in name and 'weight' in name]
    if actor_layers:
        first_layer = actor_layers[0]
        hidden_size = state_dict[first_layer].shape[0]
        print(f"\n hidden_size: {hidden_size}")
    
    print(f"\obs_dim: {210}")  
    print(f"action_dim: {5}")    