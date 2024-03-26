from pathlib import Path

path_data_reprojected = Path("data_reprojected")
path_data_reprojected.mkdir(exist_ok=True, parents=True)

path_temp = Path(".tmp_reproject")
path_temp.mkdir(exist_ok=True, parents=True)
