cd benchmarking/cold_start

# Leroy
python main_cold_leroy.py  --data_name starcraft  --model Leroy --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.0 > output_starcraft_leroy_true

# ACCSLP
python main_cold_accslp.py  --data_name starcraft  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 2 --max_nodes 20 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.0 > output_starcraft_accslp_true
python main_cold_accslp.py  --data_name starcraft  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 2 --max_nodes 20 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.25 > output_starcraft_accslp_25_edge
python main_cold_accslp.py  --data_name starcraft  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 2 --max_nodes 20 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.25 --blind node > output_starcraft_accslp_25_node
python main_cold_accslp.py  --data_name starcraft  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 2 --max_nodes 20 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.75 > output_starcraft_accslp_75_edge
python main_cold_accslp.py  --data_name starcraft  --model ACCSLP  --alpha 20 --beta 27 --rank 5 --groups 2 --max_nodes 20 --epochs 1 --kill_cnt 1 --eval_steps 1  --runs 1 --cold_perc 0.75 --blind node > output_starcraft_accslp_75_node

## CN
python main_cold_heuristic.py --data_name starcraft --use_heuristic CN --cold_perc 0.0 > output_starcraft_cn_true
python main_cold_heuristic.py --data_name starcraft --use_heuristic CN --cold_perc 0.25 > output_starcraft_cn_25_edge
python main_cold_heuristic.py --data_name starcraft --use_heuristic CN --cold_perc 0.25 --blind node > output_starcraft_cn_25_node
python main_cold_heuristic.py --data_name starcraft --use_heuristic CN --cold_perc 0.75 > output_starcraft_cn_75_edge
python main_cold_heuristic.py --data_name starcraft --use_heuristic CN --cold_perc 0.75 --blind node > output_starcraft_cn_75_node

## AA
python main_cold_heuristic.py --data_name starcraft --use_heuristic AA --cold_perc 0.0 > output_starcraft_aa_true
python main_cold_heuristic.py --data_name starcraft --use_heuristic AA --cold_perc 0.25 > output_starcraft_aa_25_edge
python main_cold_heuristic.py --data_name starcraft --use_heuristic AA --cold_perc 0.25 --blind node > output_starcraft_aa_25_node
python main_cold_heuristic.py --data_name starcraft --use_heuristic AA --cold_perc 0.75 > output_starcraft_aa_75_edge
python main_cold_heuristic.py --data_name starcraft --use_heuristic AA --cold_perc 0.75 --blind node > output_starcraft_aa_75_node

## RA
python main_cold_heuristic.py --data_name starcraft --use_heuristic RA --cold_perc 0.0 > output_starcraft_ra_true
python main_cold_heuristic.py --data_name starcraft --use_heuristic RA --cold_perc 0.25 > output_starcraft_ra_25_edge
python main_cold_heuristic.py --data_name starcraft --use_heuristic RA --cold_perc 0.25 --blind node > output_starcraft_ra_25_node
python main_cold_heuristic.py --data_name starcraft --use_heuristic RA --cold_perc 0.75 > output_starcraft_ra_75_edge
python main_cold_heuristic.py --data_name starcraft --use_heuristic RA --cold_perc 0.75 --blind node > output_starcraft_ra_75_node

## shortest path
python main_cold_heuristic.py --data_name starcraft --use_heuristic shortest_path --cold_perc 0.0 > output_starcraft_shortest_true
python main_cold_heuristic.py --data_name starcraft --use_heuristic shortest_path --cold_perc 0.25 > output_starcraft_shortest_25_edge
python main_cold_heuristic.py --data_name starcraft --use_heuristic shortest_path --cold_perc 0.25 --blind node > output_starcraft_shortest_25_node
python main_cold_heuristic.py --data_name starcraft --use_heuristic shortest_path --cold_perc 0.75 > output_starcraft_shortest_75_edge
python main_cold_heuristic.py --data_name starcraft --use_heuristic shortest_path --cold_perc 0.75 --blind node > output_starcraft_shortest_75_node

## katz
python main_cold_heuristic.py --data_name starcraft --use_heuristic katz_close --cold_perc 0.0 > output_starcraft_katz_true
python main_cold_heuristic.py --data_name starcraft --use_heuristic katz_close --cold_perc 0.25 > output_starcraft_katz_25_edge
python main_cold_heuristic.py --data_name starcraft --use_heuristic katz_close --cold_perc 0.25 --blind node > output_starcraft_katz_25_node
python main_cold_heuristic.py --data_name starcraft --use_heuristic katz_close --cold_perc 0.75 > output_starcraft_katz_75_edge
python main_cold_heuristic.py --data_name starcraft --use_heuristic katz_close --cold_perc 0.75 --blind node > output_starcraft_katz_75_node

#GCN
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_starcraft_gcn_true
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_starcraft_gcn_25_edge
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_starcraft_gcn_25_node
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_starcraft_gcn_75_edge
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model GCN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_starcraft_gcn_75_node

#GAT
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_starcraft_gat_true
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_starcraft_gat_25_edge
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_starcraft_gat_25_node
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_starcraft_gat_75_edge
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model GAT  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_starcraft_gat_75_node

#SAGE
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model SAGE  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_starcraft_sage_true
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model SAGE  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_starcraft_sage_25_edge
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model SAGE  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_starcraft_sage_25_node
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model SAGE  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_starcraft_sage_75_edge
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model SAGE  --lr 0.001 --dropout 0.3 --l2 1e-7 --num_layers 2 --hidden_channels 256  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_starcraft_sage_75_node

#GAE
python main_cold_gae.py  --data_name starcraft  --input_size 25 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_starcraft_gae_true
python main_cold_gae.py  --data_name starcraft  --input_size 25 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_starcraft_gae_25_edge
python main_cold_gae.py  --data_name starcraft  --input_size 25 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_starcraft_gae_25_node
python main_cold_gae.py  --data_name starcraft  --input_size 25 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_starcraft_gae_75_edge
python main_cold_gae.py  --data_name starcraft  --input_size 25 --gnn_model GCN  --with_loss_weight --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 3 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_starcraft_gae_75_node

#mlp_model
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model mlp_model  --lr 0.001 --dropout 0.1 --l2 1e-4 --num_layers 1 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_starcraft_mlp_true

#MF
python main_cold_gnn.py  --data_name starcraft  --input_size 25 --gnn_model MF --max_nodes 20  --lr 0.001 --dropout 0.1 --l2 0 --num_layers 3 --hidden_channels 128  --num_layers_predictor 1  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_starcraft_mf_true

#NeoGNN
python main_cold_neognn.py  --data_name starcraft  --input_size 25 --gnn_model NeoGNN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.0 > output_starcraft_neognn_true
python main_cold_neognn.py  --data_name starcraft  --input_size 25 --gnn_model NeoGNN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_starcraft_neognn_25_edge
python main_cold_neognn.py  --data_name starcraft  --input_size 25 --gnn_model NeoGNN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_starcraft_neognn_25_node
python main_cold_neognn.py  --data_name starcraft  --input_size 25 --gnn_model NeoGNN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_starcraft_neognn_75_edge
python main_cold_neognn.py  --data_name starcraft  --input_size 25 --gnn_model NeoGNN  --lr 0.001 --dropout 0.1 --l2 1e-7 --num_layers 2 --hidden_channels 128  --num_layers_predictor 2  --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_starcraft_neognn_75_node

#NCN
python main_cold_ncn.py  --dataset starcraft  --input_size 25  --gnnlr 0.001 --prelr 0.001 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 2 --nnlayers 2 --hiddim 256 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5 --cold_perc 0.0 > output_starcraft_ncn_true
python main_cold_ncn.py  --dataset starcraft  --input_size 25  --gnnlr 0.001 --prelr 0.001 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 2 --nnlayers 2 --hiddim 256 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 > output_starcraft_ncn_25_edge
python main_cold_ncn.py  --dataset starcraft  --input_size 25  --gnnlr 0.001 --prelr 0.001 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 2 --nnlayers 2 --hiddim 256 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.25 --blind node > output_starcraft_ncn_25_node
python main_cold_ncn.py  --dataset starcraft  --input_size 25  --gnnlr 0.001 --prelr 0.001 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 2 --nnlayers 2 --hiddim 256 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 > output_starcraft_ncn_75_edge
python main_cold_ncn.py  --dataset starcraft  --input_size 25  --gnnlr 0.001 --prelr 0.001 --l2 1e-7  --predp 0.1 --gnndp 0.1  --mplayers 2 --nnlayers 2 --hiddim 256 --testbs 512 --xdp 0.7 --tdp 0.3 --pt 0.75 --gnnedp 0.0 --preedp 0.4 --probscale 4.3 --proboffset 2.8 --alpha 1.0 --ln --lnnn --predictor cn1 --runs 10 --model puregcn --maskinput --jk --use_xlin --tailact --epochs 9999 --kill_cnt 3 --eval_steps 5  --cold_perc 0.75 --blind node > output_starcraft_ncn_75_node