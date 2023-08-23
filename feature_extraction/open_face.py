import subprocess
import pandas as pd
import numpy as np
import imageio

from tqdm import tqdm
from pathlib import Path
from typing import Union


# External libs paths
OPEN_FACE_PATH = Path('/opt/OpenFace/build/bin/')
OPEN_FACE_EXECUTABLE_PATH = OPEN_FACE_PATH / 'FeatureExtraction'


def get_video_frames_count_duration(path: Path):
    reader = imageio.get_reader(path, 'ffmpeg')
    frames = reader.count_frames()
    duration = reader._meta['duration']
    assert duration != 0, f'Video has no duration, please check: {path}'
    return frames, duration


class OpenFaceFeatureExtractor:
    def __init__(self, data: pd.DataFrame, output_path: Union[str, Path],
                 executable_path: Union[str, Path] = OPEN_FACE_EXECUTABLE_PATH,
                 correct_timespan: bool = True, **kwargs):
        self._executable_path = executable_path
        self._data = data
        self._output_path = Path(output_path) / 'open_face_features'
        self._output_path.mkdir(exist_ok=True, parents=True)
        self._correct_timespan = correct_timespan
        self._parameters = kwargs

    def _get_command(self, video_path):
        executable = self._executable_path
        command = [executable, '-f', video_path, '-out_dir', self._output_path]
        additional_parameters = [f'-{k}' for k, v in self._parameters.items() if v]
        command.extend(additional_parameters)
        return command

    def run(self):
        for index, video in tqdm(self._data.iterrows(), total=len(self._data)):
            output_filename = self._output_path / f'{video["Path"].stem}.csv'
            if already_processed(video, output_filename):
                continue
            command = self._get_command(video['Path'])
            process = subprocess.run(command, capture_output=True)

            if process.returncode:
                print(f"An error occurred while processing video {video['ID']}:")
                print(process.stdout, process.stderr)
            if self._correct_timespan:
                _, video_duration = get_video_frames_count_duration(video['Path'])
                features = pd.read_csv(output_filename)
                difference = abs(features['timestamp'].max() - video_duration)
                if difference > 2:  # 2 seconds
                    features['timestamp'] = np.linspace(0.0, video_duration, num=(len(features)))
                    features.to_csv(output_filename, index=False)  # todo check index=False


def already_processed(video, output_filename) -> bool:
    if not output_filename.is_file():
        return False

    try:
        processed_frames_count = len(pd.read_csv(output_filename))
        video_frames_count, video_duration = get_video_frames_count_duration(video['Path'])

        if (missing_frames := video_frames_count - processed_frames_count) < 25:
            # todo implement image extraction?
            print(f'Missing frames count: {missing_frames}, {processed_frames_count}: {output_filename}')
            return True
    except:  # file is corrupted
        return False

    return False
