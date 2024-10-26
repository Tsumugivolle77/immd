import dataclasses


@dataclasses.dataclass
class config:
    max_rows: int = int(1e5)
    download_dir: str = "/Users/offensive77/Code/mmdb-example/24WS-mmd-code-public/rec_sys/file/"
    unzipped_dir: str = download_dir + "ml-25m/"
    dowload_url: str = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    file_path: str = download_dir + "ml-25m/ratings.csv"
