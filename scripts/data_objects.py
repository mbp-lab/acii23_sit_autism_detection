import pandas as pd

from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from collections import namedtuple


class SITVersion(Enum):
    PC = 1
    ONLINE = 2


class SITCondition(Enum):
    ASD = 1  # Autistic
    MDD = 2  # Depression
    ADHD = 3
    SAD = 4  # Social Anxiety
    HG = 5  # Control group


@dataclass
class SITDataset:
    path: Path
    raw: Path
    processed: Path
    corrected: Path
    SIT_version: SITVersion
    labels: pd.DataFrame
    blacklist: []

    @property
    def video_extension(self) -> list:
        return ['*.webm', '*.mkv'] if self.SIT_version == SITVersion.ONLINE else ['*.mkv']


Part = namedtuple('Part', ['name', 'speaker', 'start', 'end'])
SIT_PARTS = {
    SITVersion.ONLINE: {
        '0': [],
        '1': [
            Part(name='neutral', speaker='actress', start=None, end=None),
        ],
        '2': [
            Part(name='neutral', speaker='actress', start=None, end=16),
            Part(name='neutral', speaker='participant', start=16, end=None),
        ],
        '3': [
            Part(name='joy', speaker='actress', start=None, end=None),
        ],
        '4': [
            Part(name='joy', speaker='actress', start=None, end=6),
            Part(name='joy', speaker='participant', start=6, end=None),
        ],
        '5': [
            Part(name='disgust', speaker='actress', start=None, end=None),
        ],
        '6': [
            Part(name='disgust', speaker='actress', start=None, end=6),
            Part(name='disgust', speaker='participant', start=6, end=None),
        ],
    },
    SITVersion.PC: {
        '2': [
            Part(name='neutral', speaker='actress', start=None, end=40),
            Part(name='neutral', speaker='participant', start=40, end=66),
            Part(name='joy', speaker='actress', start=66, end=97),
            Part(name='joy', speaker='participant', start=97, end=123),
            Part(name='disgust', speaker='actress', start=123, end=156),
            Part(name='disgust', speaker='participant', start=156, end=None),
        ]
    },
}
