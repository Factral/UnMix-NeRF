CUDA_VISIBLE_DEVICES=0,1,2,3 ns-train unmixnerf \
 --machine.seed 42 \
 --machine.num-devices 4 \
 --log-gradients True \
 --pipeline.num_classes 4 \
 --pipeline.model.far-plane 1000 \
 --pipeline.model.near_plane 0.05 \
 --pipeline.model.background-color random \
 --pipeline.model.spectral_loss_weight 5.0 \
 --pipeline.model.temperature 0.5 \
 --pipeline.model.pred_dino False \
 --pipeline.model.pred_specular True \
 --pipeline.model.load_vca True \
 --pipeline.datamanager.images-on-gpu False \
 --pipeline.datamanager.patch-size 1 \
 --pipeline.datamanager.train-num-rays-per-batch 2048 \
 --pipeline.datamanager.eval-num-rays-per-batch 256 \
 --pipeline.model.method rgb+spectral \
 --data  data/hsnerf/bayspec/pinecone/processed_dataset  \
 --experiment-name "pinecone-t0.7-k4" \
 --vis wandb --viewer.websocket-port 7007 \

 #3.8.18