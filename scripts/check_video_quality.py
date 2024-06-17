import pandas as pd

from mbp.preprocessing.loader import Loader
from scripts.datasets import sit_datasets
from mbp.utils.media_metadata import VideoMetadataExtractor
from scripts.utils import extract_id_clip


def analyse_metadata(metadata: pd.DataFrame):
    metadata[['id', 'clip']] = metadata.T.apply(
        lambda row: extract_id_clip(row['path'].stem)
    ).T
    metadata['clip'] = metadata['clip'].astype(int)
    clips = metadata['clip'].unique()

    for clip in sorted(clips):
        sub_metadata = metadata[metadata['clip'] == clip][
            [# 'id',
             'duration', 'fps', 'frames', 'mtime']]
        rows_shown = 9
        print(f'Information for videos with clip {clip}:\n'
              f'{sub_metadata.describe()}')
        print(f'Videos with the lowest duration: \n'
              f'{sub_metadata.sort_values(by="duration").head(rows_shown)}')
        print(f'Videos with the lowest frame count: \n'
              f'{sub_metadata.sort_values(by="frames").head(rows_shown)}')
        print(f'Videos with the lowest FPS: \n'
              f'{sub_metadata.sort_values(by="fps").head(rows_shown)}')


def main():
    for dataset in sit_datasets.values():
        loader = Loader(dataset.raw, dataset.video_extension)
        data = loader.load()
        extractor = VideoMetadataExtractor(data)
        metadata = extractor.run()

        analyse_metadata(metadata)


if __name__ == '__main__':
    main()
