import pandas as pd

from pathlib import Path
from typing import Union
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

from mbp.preprocessing.loader import Loader
from mbp.transformers.eye_gaze import EyeGazeFeatures
from mbp.transformers.head_pose import HeadPoseFeatures

from scripts.data_objects import SITDataset, SITVersion, SITCondition
from scripts.utils import extract_id_clip


eye_gaze_columns = ['gaze_angle_x', 'gaze_angle_y']
head_columns = ['pose_Rx', 'pose_Ry', 'pose_Rz']
au_intensity_columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r',
                        'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r',
                        'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r',
                        'AU26_r', 'AU45_r']
au_presense_columns = ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c',
                       'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c',
                       'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c',
                       'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c',
                       'AU28_c', 'AU45_c']
general_info_columns = ['frame', 'timestamp', 'confidence', 'success']

def calculate_cumulative(dataframe: pd.DataFrame,
                         column: str) -> pd.DataFrame:
    cumulative_column = f'{column}s'
    dataframe[cumulative_column] = dataframe[column]
    clips = sorted(dataframe['clip'].unique())
    for index, clip in enumerate(clips[1:], 1):
        max_value = dataframe[dataframe['clip'] == clips[index - 1]][
            cumulative_column].max()
        dataframe.loc[
            dataframe['clip'] == clip, cumulative_column] += max_value
    return dataframe

def add_part_information(dataframe, sit_version):
    def helper(condition, part, speaker):
        dataframe.loc[condition, 'part'] = part
        dataframe.loc[condition, 'speaker'] = speaker

    # todo rewrite based on SITParts
    if sit_version == SITVersion.ONLINE:
        helper((dataframe['clip'] == '1'), 'neutral', 'actress')
        helper((dataframe['clip'] == '2') & (dataframe['timestamp'] <= 16),
               'neutral', 'actress')  # neut_instr
        helper((dataframe['clip'] == '2') & (dataframe['timestamp'] > 16),
               'neutral', 'participant')
        helper((dataframe['clip'] == '3'), 'joy', 'actress')
        helper((dataframe['clip'] == '4') & (dataframe['timestamp'] <= 6),
               'joy', 'actress')  # joy_instr
        helper((dataframe['clip'] == '4') & (dataframe['timestamp'] > 6),
               'joy', 'participant')
        helper((dataframe['clip'] == '5'), 'disgust', 'actress')
        helper((dataframe['clip'] == '6') & (dataframe['timestamp'] <= 6),
               'disgust', 'actress')  # disgust_instr
        helper((dataframe['clip'] == '6') & (dataframe['timestamp'] > 6),
               'disgust', 'participant')
    elif sit_version == SITVersion.PC:
        neutral_p_start = 40
        joy_a_start, joy_p_start = 66, 97
        disgust_a_start, disgust_p_start = 123, 156

        helper((dataframe['timestamp'] < neutral_p_start), 'neutral',
               'actress')
        helper((dataframe['timestamp'] >= neutral_p_start) & (
                dataframe['timestamp'] < joy_a_start), 'neutral',
               'participant')
        helper((dataframe['timestamp'] >= joy_a_start) & (
                    dataframe['timestamp'] < joy_p_start), 'joy',
               'actress')
        helper((dataframe['timestamp'] >= joy_p_start) & (
                dataframe['timestamp'] < disgust_a_start), 'joy',
               'participant')
        helper((dataframe['timestamp'] >= disgust_a_start) & (
                dataframe['timestamp'] < disgust_p_start), 'disgust',
               'actress')
        helper((dataframe['timestamp'] >= disgust_p_start), 'disgust',
               'participant')

    return dataframe

class OpenFaceFeaturePreparator:
    def __init__(self, dataset: SITDataset, output_filepath: Union[str, Path]):
        self._dataset = dataset
        self._output_filepath = output_filepath

        loader = Loader(self._dataset.processed / 'open_face_features',
                        ['*.csv'])
        open_face_feature_files = loader.load()

        self._datafiles = pd.DataFrame(open_face_feature_files, columns=['path'])
        self._datafiles[['id', 'clip']] = self._datafiles.T.apply(
            lambda row: extract_id_clip(row['path'].stem)).T
        self._datafiles['clip'] = pd.to_numeric(self._datafiles['clip'])
        self._data = None

    def _exclusion_helper(self, participants_to_exclude, msg):
        if participants_to_exclude.any():
            print(f"{msg} {list(self._datafiles.loc[participants_to_exclude]['id'].unique())}")
            self._datafiles = self._datafiles.drop(participants_to_exclude)

    def _exclude_incomplete_participants(self):
        participants_to_exclude = pd.Series()
        if self._dataset.SIT_version == SITVersion.PC:
            # including only last recorded clip, except clip 1
            max_clip_idx = self._datafiles.groupby('id')['clip'].idxmax()
            self._datafiles = self._datafiles.loc[max_clip_idx]
            participants_to_exclude = self._datafiles[self._datafiles['clip'] < 2]['id'].index
        elif self._dataset.SIT_version == SITVersion.ONLINE:
            participants_to_exclude = []
            for participant_id in self._datafiles['id'].unique():
                clips = set(self._datafiles[self._datafiles['id'] == participant_id]['clip'])
                if not {1, 2, 3, 4, 5, 6}.issubset(clips):
                    participants_to_exclude.append(participant_id)
            participants_to_exclude = self._datafiles[self._datafiles.T.apply(
                lambda row: row['id'] in participants_to_exclude).T].index
        else:
            print(f'{self._dataset.SIT_version=} is not supported...')

        self._exclusion_helper(participants_to_exclude,
                               msg='Participants without complete video clips:')

    def _get_participant_features(self, participant_id):
        participant_features = []
        columns_to_include = [
            *general_info_columns,
            *eye_gaze_columns, *head_columns,
            *au_intensity_columns, *au_presense_columns,
        ]

        for index, file in self._datafiles[
            self._datafiles['id'] == participant_id].iterrows():
            file_features = pd.read_csv(file['path'])
            file_features = file_features[columns_to_include].copy()
            file_features['id'] = file['id']
            file_features['clip'] = file['clip']
            participant_features.append(file_features)
        participant_features = pd.concat(participant_features)
        participant_features = participant_features.reset_index(drop=True)

        return participant_features

    # open all csv files with needed columns and combine into one big csv
    def _get_combined_frame(self):
        frames = []

        ids = self._datafiles['id'].unique()
        for participant_id in tqdm(ids):
            participant_features = self._get_participant_features(participant_id)
            participant_features['successful'] = participant_features.T.apply(
                lambda x: x['confidence'] >= 0.75 and x['success'])
            indices_to_exclude = participant_features[
                participant_features['success'] != 1].index
            if len(indices_to_exclude) < len(participant_features) / 10:
                participant_features = participant_features.drop(indices_to_exclude).reset_index(drop=True)
                participant_features.loc[participant_features['timestamp'] == 0, 'timestamp'] = 0.001
                participant_features = calculate_cumulative(participant_features, 'frame')
                participant_features = calculate_cumulative(participant_features, 'timestamp')
                participant_features = add_part_information(participant_features, self._dataset.SIT_version)
                participant_features = participant_features.drop(columns=['successful', *general_info_columns])
                frames.append(participant_features)
            else:
                print(f'\nUnsuccessfully extracted frame count exceeds the threshold. \n'
                      f'Excluding participant with {participant_id=}')

        return pd.concat(frames).reset_index(drop=True)

    def run(self):
        # Exclude all participants with incomplete recorded video clips
        self._exclude_incomplete_participants()

        participants_with_no_features = set(self._dataset.labels['id'].unique()).difference(self._datafiles['id'].unique())
        if participants_with_no_features:
            print(f'Participants with missing features: {participants_with_no_features}')

        # Exclude all participants from defined blacklist
        participants_to_exclude = set(self._dataset.blacklist).intersection(self._datafiles['id'].unique())
        participants_to_exclude = self._datafiles[self._datafiles.T.apply(
            lambda row: row['id'] in participants_to_exclude).T].index
        self._exclusion_helper(participants_to_exclude,
                               msg='Excluding participants from blacklist with ids:')

        # Exclude all participants without labels
        participants_to_exclude = set(self._datafiles['id'].unique()).difference(self._dataset.labels['id'].unique())
        participants_to_exclude = self._datafiles[self._datafiles.T.apply(
            lambda row: row['id'] in participants_to_exclude).T].index
        self._exclusion_helper(participants_to_exclude,
                               msg='Excluding participants without labels:')

        self._data = self._get_combined_frame()

        self._data.to_csv(self._output_filepath, index=False)


class ActionUnitFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        item = {}
        for au in au_presense_columns:
            item.update({
                f'{au}_mean': X[au].mean()
            })
        for au in au_intensity_columns:
            item.update({
                f'{au}_mean': X[au].mean(),
                f'{au}_var': X[au].var(),
                f'{au}_median': X[au].median(),
            })

        return item


def get_transformer_functionals(transformer, data: pd.DataFrame, **params) -> dict:
    transformer = transformer(**params)
    features = transformer.fit_transform(data)
    return features

def functionals_calculation_helper(item, data):
    item.update(
        get_transformer_functionals(
            transformer=ActionUnitFeatures,
            data=data
        )
    )

    item.update(
        get_transformer_functionals(
            transformer=EyeGazeFeatures,
            data=data[[*eye_gaze_columns, 'timestamps']])
    )

    item.update(
        get_transformer_functionals(
            transformer=HeadPoseFeatures,
            data=data[[*head_columns, 'timestamps']])
    )

    return item


class OpenFaceFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, total: bool = False):
        self._total = total

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        indices = X['id'].unique() if self._total else X[
            ['id', 'part', 'speaker']].groupby(
            ['id', 'part', 'speaker']).mean().index

        new_data = []
        for index in indices:
            if self._total:
                item = {'id': index}
                condition = (X['id'] == index)
            else:
                p_id, part, speaker = index
                item = {'id': p_id, 'part': part, 'speaker': speaker}
                condition = (X['id'] == p_id) & (X['part'] == part) & (X['speaker'] == speaker)

            item = functionals_calculation_helper(item, X[condition])
            new_data.append(item)
        new_data = pd.DataFrame(new_data)

        return new_data


class AudioVisualFeatureCombiner:
    def __init__(self, visual_features, audio_feature_files,
                 labels, output_filepath):
        self._visual_features = visual_features
        self._audio_feature_files = audio_feature_files
        self._labels = labels
        self._output_filepath = output_filepath

    def _get_audio_features(self):
        print('Transforming audio features...')
        all_features = pd.DataFrame()
        for file in tqdm(self._audio_feature_files):
            features = pd.read_csv(file)
            pid_features = {'id': features['id'][0]}
            # features = features.drop(columns=['file', 'start', 'end', 'id'])
            features = features.drop(columns=['id'])
            for j, row in features.iterrows():
                part = row['part']
                columns = row.drop(['part']).index
                for column in columns:
                    pid_features[f'audio_{column}_{part}'] = row[column]
            all_features = pd.concat(
                [all_features, pd.DataFrame([pid_features])])
        return all_features.reset_index(drop=True)

    def _transform_visual_features(self):
        print('Transforming visual features...')
        ids = self._visual_features['id'].unique()
        parts = self._visual_features['part'].unique()
        speakers = self._visual_features['speaker'].unique()
        columns = self._visual_features.drop(columns=['id', 'part', 'speaker']).columns
        all_features = pd.DataFrame()
        for pid in tqdm(ids):
            pid_features = {'id': pid}
            for part in parts:
                for speaker in speakers:
                    for column in columns:
                        condition = ((self._visual_features['id'] == pid) &
                                     (self._visual_features['part'] == part) &
                                     (self._visual_features['speaker'] == speaker))
                        if column.startswith('AU'):
                            pid_features[
                                f'facial_{column}_{part}_{speaker}'
                            ] = self._visual_features[condition][column].mean()
                        else:
                            pid_features[
                                f'{column}_{part}_{speaker}'
                            ] = self._visual_features[condition][column].mean()
            all_features = pd.concat(
                [all_features, pd.DataFrame([pid_features])])
        return all_features.reset_index(drop=True)

    def run(self):
        audio_features = self._get_audio_features()
        visual_features = self._transform_visual_features()
        # labels = self._labels[(self._labels[str(SITCondition.ASD)] == 1) |
        #                       (self._labels[str(SITCondition.HG)] == 1)]
        features = pd.merge(self._labels[['id', str(SITCondition.ASD)]],
                            visual_features, on='id', how='inner')
        features = pd.merge(features, audio_features, on='id', how='inner')
        print('Saving audiovisual features...')
        features.to_csv(self._output_filepath, index=False)


if __name__ == '__main__':
    from scripts.datasets import sit_datasets
    for dataset in list(sit_datasets.values())[1:]:
        output_filepath = Path('../data/sit_test.csv')
        p = OpenFaceFeaturePreparator(dataset, output_filepath)
        p.run()

        data = pd.read_csv(output_filepath)
        output_filepath.unlink()
        p = OpenFaceFeatureTransformer()
        visual_features = p.fit_transform(data)

        loader = Loader(dataset.processed / 'open_smile_features', ['*.csv'])
        audio_feature_files = loader.load()
        VERSION = 'test'
        output_filepath = Path(dataset.processed / f'classification_features_v{VERSION}.csv',)

        feature_combiner = AudioVisualFeatureCombiner(visual_features, audio_feature_files, dataset.labels, output_filepath)
        feature_combiner.run()

