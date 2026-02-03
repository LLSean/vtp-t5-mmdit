from cheesechaser.datapool import Danbooru2024SfwDataPool
from cheesechaser.query import DanbooruIdQuery

pool = Danbooru2024SfwDataPool()
#my_waifu_ids = DanbooruIdQuery(['surtr_(arknights)', 'solo']) 
# above is only available when Danbooru is accessible, if not, use following:
import pandas as pd

# read parquet file
df = pd.read_parquet('metadata.parquet', 
                     columns=['id', 'tag_string']) # read only necessary columns
                     
#surtr_(arknights) -> gets interpreted as regex so we need to escape the brackets
subdf = df[
           df['tag_string'].str.contains('solo')]
ids = subdf.index.tolist()
print(len(ids))
print(ids[:5]) # check the first 5 ids

# download danbooru images with surtr+solo, to directory ./danbooru_surtr_solo_100k
pool.batch_download_to_directory(
    resource_ids=ids[10000:20000],
    dst_dir='./danbooru_surtr_solo_100k',
    max_workers=12,
)