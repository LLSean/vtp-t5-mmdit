python3 -m mmdit.scripts.prepare_edges2shoes_t2i --max_samples 32

python3 -m mmdit.scripts.cache_vtp_images \
  --config configs/t2i_vtp_256_smoke.yaml \
  --meta_jsonl data/edges2shoes_t2i/train_images.jsonl \
  --out_dir cache/edges2shoes_t2i_latents

python3 -m mmdit.scripts.cache_vtp_images \
  --config configs/t2i_vtp_256_flow_matching_anime_buckets.yaml \
  --meta_jsonl data/danbooru_surtr_solo_100k_t2i/train_images.jsonl \
  --out_dir cache/danbooru_surtr_solo_100k_t2i_latents_buckets



python3 -m mmdit.scripts.cache_vtp_images \
  --config configs/t2i_vtp_256_flow_matching_anime_buckets_big.yaml \
  --meta_jsonl data/danbooru_surtr_solo_100k_t2i/train_images.jsonl \
  --out_dir cache/danbooru_surtr_solo_100k_t2i_latents

# small dataset
python3 -m mmdit.scripts.cache_vtp_images \
  --config configs/t2i_vtp_256_flow_matching_anime_buckets.yaml \
  --meta_jsonl data/danbooru_surtr_solo_100k_t2i_small/train_images.jsonl \
  --out_dir cache/danbooru_surtr_solo_100k_t2i_latents_small

python3 -m mmdit.scripts.estimate_latent_scale \
  --cache_dir cache/danbooru_surtr_solo_100k_t2i_latents_small \
  --target_std 1.0 \
  --max_files 0 \
  --seed 42 \
  --print_yaml


# normal dataset
python3 -m mmdit.scripts.estimate_latent_scale \
  --cache_dir cache/danbooru_surtr_solo_100k_t2i_latents \
  --target_std 1.0 \
  --max_files 0 \
  --seed 42 \
  --print_yaml
torchrun --master_port 29501 --nproc_per_node=1 -m mmdit.scripts.train_t2v   --config configs/t2i_vtp_256_flow_matching_anime.yaml   --train_cache_dir cache/danbooru_surtr_solo_100k_t2i_latents



python3 -m mmdit.scripts.train_t2v   --config configs/t2i_vtp_256_flow_matching_anime_buckets.yaml   --train_cache_dir cache/danbooru_surtr_solo_100k_t2i_latents --resume


python3 -m mmdit.scripts.train_t2v   --config configs/t2i_vtp_256_flow_matching_anime_buckets_big.yaml   --train_cache_dir cache/danbooru_surtr_solo_100k_t2i_latents --resume