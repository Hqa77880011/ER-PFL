import argparse
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', default='data/voice.npz')
    args = parser.parse_args()
    path = Path(args.manifest)
    data = pd.read_csv(path)
    required = {'audio', 'y', 'group', 'corpus', 'age', 'sex', 'split'}
    if not required <= set(data):
        raise ValueError('Manifest requires ' + ','.join(sorted(required)))
    extractor = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                feature_level=opensmile.FeatureLevel.Functionals)
    features = []
    for row in data.itertuples():
        signal, rate = sf.read(path.parent / row.audio, always_2d=True)
        signal = signal.mean(1)
        if rate != 16000:
            divisor = np.gcd(rate, 16000)
            signal = resample_poly(signal, 16000 // divisor, rate // divisor)
        if len(signal) < 16000:
            raise ValueError('Recording shorter than one second: ' + row.audio)
        features.append(extractor.process_signal(signal, 16000).iloc[0].to_numpy())
    arrays = {key: data[key].to_numpy() for key in required - {'audio'}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=np.array(features, dtype=np.float32), **arrays)


if __name__ == '__main__':
    main()
