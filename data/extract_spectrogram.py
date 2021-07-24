import os
import sys
import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.io import wavfile

sys.path.append("utils")
from audio_utils import waveform_to_examples


def get_audio_list(root_dir="audio", ext=".wav"):
    audio_list = glob.glob(os.path.join(root_dir, "*"+ext))
    return audio_list


def change_root_ext(ori_path_list, 
                    new_root, 
                    new_ext=".npy"):
    new_path_list = [os.path.join(new_root, os.path.split(ori_path)[-1]) 
                        for ori_path in ori_path_list]
    new_path_list = [new_path.split(".")[0] + "_" + "{}" + new_ext 
                        for new_path in new_path_list]
    return new_path_list


def get_length_dict(length_file):
    sample_rate = 8
    length_table = pd.read_table(length_file)
    audio_length_dict = dict(
        zip(
            [name.split(".")[0]+".wav" for name in length_table["name"]],
            [l//sample_rate for l in length_table["length"]]
    ))
    return audio_length_dict


def extract_spect(audio_list, spect_list, audio_length_dict):
    print("Number: {}".format(len(audio_list)))
    for audio, spect in zip(audio_list, spect_list):
        length = audio_length_dict[audio.split(os.path.sep)[-1]]
        try:
            print("processing {}".format(audio))
            sr, wav_data = wavfile.read(audio)
            assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
            samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
            steps = samples.shape[0]
            for i in range(length):
                center = int((steps - 16000) / (length - 1) * i) + 8000
                feature = waveform_to_examples(samples[center-8000:center+8000], sr).astype(np.float32)
                np.save(spect.format(i), feature)
        except:
            print("Error: {}".format(audio))


def extract_spect_batch(audio_list, spect_list, audio_length_dict, n_jobs=4):
    p = Pool(n_jobs)
    [p.apply_async(extract_spect, (audio_list[i::n_jobs], \
        spect_list[i::n_jobs], audio_length_dict))
        for i in range(n_jobs)]
    p.close()
    p.join()


def main():
    audio_root = "audio"
    spect_root = "spectrograms"
    if not os.path.exists(spect_root):
        os.mkdir(spect_root)

    audio_length_dict = get_length_dict("video_length_table.txt")

    audio_list = get_audio_list(audio_root)
    spect_list = change_root_ext(audio_list, spect_root, ".npy")
    extract_spect_batch(audio_list, spect_list, audio_length_dict)
    #extract_spect(audio_list, spect_list, audio_length_dict)


if __name__ == '__main__':
    main()