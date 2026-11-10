from torch_geometric.nn import SAGEConv,GCNConv


# default_config={
#     
#     
#     "hidden_dim" : 64,
#     "latent_dim" : 64,
#     "conv_layer" :SAGEConv,
#     "epochs_cluster" : 300,
#     "pre_epoch":200,
#     "lr_cluster" : 0.001}

configs={
"default":{
        "hidden_dim" : 64,
        "latent_dim" : 64,
        "conv_layer" :GCNConv,
        "epochs_cluster" : 500,
        "pre_epoch":200,
        "lr_cluster" : 0.0001},
"Adam":{
    "hidden_dim" : 256,
    "latent_dim" : 32,
    "conv_layer" :SAGEConv,
    "epochs_cluster" : 8,
    "pre_epoch":5,
    "lr_cluster" : 0.0001},
    "Muraro":{
        "hidden_dim" : 64,
        "latent_dim" : 64,
        "conv_layer" :SAGEConv,
        "epochs_cluster" : 500,
        "pre_epoch":200,
        "lr_cluster" : 0.001},
    "Bach":{
        "hidden_dim" : 64,
        "latent_dim" : 128,
        "conv_layer" :SAGEConv,
        "epochs_cluster" : 800,
        "pre_epoch":200,
        "lr_cluster" : 0.001},

    "Cao_2020_Spleen":{
        "hidden_dim" : 256,
        "latent_dim" : 32,
        "conv_layer" :SAGEConv,
        "epochs_cluster" : 800,
        "pre_epoch":500,
        "lr_cluster" : 0.0001}, 
    "Quake_10x_Bladder":{
        "hidden_dim" : 64,
        "latent_dim" : 128,
        "conv_layer" :SAGEConv,
        "epochs_cluster" : 800,
        "pre_epoch":200,
        "lr_cluster" : 0.001},
    "Quake_10x_Limb_Muscle":{
        "hidden_dim" : 64,
        "latent_dim" : 512,
        "conv_layer" :SAGEConv,
        "epochs_cluster" : 800,
        "pre_epoch":500,
        "lr_cluster" : 0.0001},
    "Quake_10x_Spleen":{
        "hidden_dim" : 128,
        "latent_dim" : 32,
        "conv_layer" :SAGEConv,
        "epochs_cluster" : 300,
        "pre_epoch":200,
        "lr_cluster" : 0.005},
    "Quake_Smart-seq2_Diaphragm":{
        "hidden_dim" : 64,
        "latent_dim" : 64,
        "conv_layer" :GCNConv,
        "epochs_cluster" : 500,
        "pre_epoch":200,
        "lr_cluster" : 0.0001},
    "Quake_Smart-seq2_Limb_Muscle":{
        "hidden_dim" : 64,
        "latent_dim" : 32,
        "conv_layer" :SAGEConv,
        "epochs_cluster" : 300,
        "pre_epoch":200,
        "lr_cluster" : 0.001},
    "Quake_Smart-seq2_Lung":{
        "hidden_dim" : 256,
        "latent_dim" : 64,
        "conv_layer" :GCNConv,
        "epochs_cluster" : 800,
        "pre_epoch":200,
        "lr_cluster" : 0.001},
    "Quake_Smart-seq2_Trachea":{
        "hidden_dim" : 64,
        "latent_dim" : 32,
        "conv_layer" :GCNConv,
        "epochs_cluster" : 800,
        "pre_epoch":500,
        "lr_cluster" : 0.001},
    "Romanov":{
        "hidden_dim" : 64,
        "latent_dim" : 64,
        "conv_layer" :GCNConv,
        "epochs_cluster" : 300,
        "pre_epoch":200,
        "lr_cluster" : 0.0001},

    "Tosches_turtle":{
        "hidden_dim" : 128,
        "latent_dim" :64,
        "conv_layer" :SAGEConv,
        "epochs_cluster" : 800,
        "pre_epoch":500,
        "lr_cluster" : 0.001},
    "Wang_Large_Intestine":{
        "hidden_dim" : 512,
        "latent_dim" : 512,
        "conv_layer" :GCNConv,
        "epochs_cluster" : 300,
        "pre_epoch":200,
        "lr_cluster" : 0.0001} ,
    "Young": {
        "hidden_dim" : 256,
        "latent_dim" : 32,
        "conv_layer" :SAGEConv,
        "epochs_cluster" : 800,
        "pre_epoch":500,
        "lr_cluster" : 0.0001}
    }