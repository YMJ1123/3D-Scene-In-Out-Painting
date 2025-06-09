# python remove_anything_video.py \
#     --input_video ./example/video/paragliding/original_video.mp4 \
#     --coords_type key_in \
#     --point_coords 652 162 \
#     --point_labels 1 \
#     --dilate_kernel_size 15 \
#     --output_dir ./results \
#     --sam_model_type "vit_t" \
#     --sam_ckpt ./weights/mobile_sam.pt \
#     --lama_config lama/configs/prediction/default.yaml \
#     --lama_ckpt ./pretrained_models/big-lama \
#     --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
#     --vi_ckpt ./pretrained_models/sttn.pth \
#     --mask_idx 2 \
#     --fps 25

# python remove_anything_video.py \
#     --input_video ./example/video/mouse/20250601_232814.mp4 \
#     --coords_type click \
#     --point_coords 652 162 \
#     --point_labels 1 \
#     --dilate_kernel_size 15 \
#     --output_dir ./results \
#     --sam_model_type "vit_h" \
#     --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
#     --lama_config lama/configs/prediction/default.yaml \
#     --lama_ckpt ./pretrained_models/big-lama \
#     --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
#     --vi_ckpt ./pretrained_models/sttn.pth \
#     --mask_idx 2 \
#     --fps 25

python remove_anything_video.py \
    --input_video ./example/video/mouse/20250601_232814.mp4\
    --coords_type key_in \
    --point_coords 550 980 \
    --point_labels 1 \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
    --lama_config lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama \
    --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
    --vi_ckpt ./pretrained_models/sttn.pth \
    --mask_idx 2 \
    --fps 25