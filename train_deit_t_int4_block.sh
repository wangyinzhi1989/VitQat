SHNAME=$(basename "$0"  | awk -F . '{print $1}')
echo $SHNAME
DATE=$(date +%Y%m%d%H%M%S)
LOG_DIR_NMAE=$SHNAME
echo $LOG_DIR_NMAE
cd ../exp
mkdir $LOG_DIR_NMAE
cd $LOG_DIR_NMAE
mkdir $DATE
cd ../../solution
SAVE_DIR=../exp/$LOG_DIR_NMAE/$DATE
echo $SAVE_DIR

python train_VVTQ.py \
--gpu 4 5 6 7 \
--rank 0 \
--model deit_tiny_patch16_224_quant \
--batch-size 64 \
--lr 5e-4 \
--warmup-epochs 5 \
--min-lr 1e-5 \
--wbits 4 \
--abits 4 \
--workers 4 \
--reg \
--softlabel_path /data/wangyinzhi/FKD_soft_label_500_crops_marginal_smoothing_k_5/imagenet \
--finetune /data4022/wangyinzhi/quantization/Quantization_variation_pre_ckpt/deit_tiny_fp/deit_tiny_fp/ckpt/current_checkpoint.pth \
--save_checkpoint_path $SAVE_DIR \
--data /datassd/Inet1K
