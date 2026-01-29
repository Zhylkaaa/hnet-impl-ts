

for subset in diagnostic subdiagnostic supdiagnostic form rhythm; do
    WANDB_RUN_GROUP=ptbxl_${subset} TORCH_LOGS=recompiles,cudagraphs uv run python downstream_trainer.py --pretrained_model_path='ecg-ts/yse7h8ci/checkpoints/epoch=9-step=250020.ckpt' --subset=$subset --base_lr 1e-3 --weight_decay 1e-5 --num_epochs 500 --num_workers 4
done