CHECKPOINT=${CHECKPOINT:-'ecg-ts/7616tiff/checkpoints/epoch=9-step=250020.ckpt'}

for subset in diagnostic subdiagnostic supdiagnostic form rhythm; do
    WANDB_RUN_GROUP=ptbxl_${subset} TORCH_LOGS=recompiles,cudagraphs uv run python downstream_trainer.py --pretrained_model_path=${CHECKPOINT} --subset=$subset --base_lr 1e-5 --weight_decay 1e-5 --num_epochs 100 --num_workers 4 --full_finetune
done