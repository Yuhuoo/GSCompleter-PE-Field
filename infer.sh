python ./infer_viewchanger_single_v2.py \
  --moge_checkpoint_path "./moge-2-vitl-normal/model.pt" \
  --transformer_checkpoint_path "./checkpoints" \
  --flux_kontext_path "./FLUX.1-Kontext-dev" \
  --input_image "/home/gaoao-p/code/GSInpainter/depthsplat/experiments/12.15/baseline/dl3dv_ctx_1v_tgt_2v_n10/before/0bfdd020cf475b9c68e4b469d1d1a2d0cad303eefe8b78fb23078/color/000004.png" \
  --output_dir "outputs/12.15/0bfd-test-10" \
  --phi 5 --theta 5