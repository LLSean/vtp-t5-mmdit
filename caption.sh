python3 -m mmdit.scripts.caption_images_t5gemma2 --image_dir danbooru_surtr_solo_100k --out_root data/danbooru_surtr_solo_100k_t2i_smoke --model_name_or_path t5gemma-2-270m-270m --max_images 1 --batch_size 1 --device cpu --dtype fp32 --caption_mode single --length_policy short


CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m mmdit.scripts.caption_images_t5gemma2 \
  --image_dir danbooru_surtr_solo_100k \
  --out_root data/danbooru_surtr_solo_100k_t2i \
  --model_name_or_path t5gemma-2-270m-270m \
  --caption_mode single \
  --length_policy long \
  --device cuda \
  --dtype auto \
  --batch_size 4 \
  --resume


CUDA_VISIBLE_DEVICES=4 python3 -m mmdit.scripts.caption_images_t5gemma2 \
  --backend qwen3vl \
  --image_dir danbooru_surtr_solo_100k \
  --out_root data/danbooru_surtr_solo_100k_t2i \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --caption_mode single \
  --length_policy random \
  --batch_size 4 \
  --resume