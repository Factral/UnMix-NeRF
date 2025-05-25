CUDA_VISIBLE_DEVICES=0 ns-train unmixnerf --steps_per_save 1000 \
 --save_only_latest_checkpoint False  --machine.seed 42 --log-gradients True --pipeline.num_classes 6 \
 --pipeline.model.far-plane 1000 --pipeline.model.near_plane 0.05 --pipeline.model.background-color random \
 --pipeline.model.spectral_loss_weight 5.0 --pipeline.model.temperature 0.7 --pipeline.model.pred_dino False \
 --pipeline.model.pred_specular True --pipeline.model.load_vca True --pipeline.datamanager.images-on-gpu True \
 --pipeline.model.implementation tcnn \
 --pipeline.datamanager.patch-size 1 --pipeline.datamanager.train-num-rays-per-batch 4096 --pipeline.model.method rgb+spectral \
 --data data/processed/cbox_sphere --experiment-name cbox_sphere-t0.4-k7 --vis wandb --viewer.websocket-port 7007
