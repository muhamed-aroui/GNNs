python run_infer.py \
--gpu='1' \
--nr_types=6 \
--batch_size=64 \
--model_mode=fast \
--model_path=../pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=archive/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/malignant/SOB/ductal_carcinoma/SOB_M_DC_14-10926/40X/ \
--output_dir=./Output \
--mem_usage=0.1 \
--draw_dot \
--save_qupath
