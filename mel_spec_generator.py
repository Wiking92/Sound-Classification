import os
from os import walk
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import warnings


class CreateMelSpec:
    warnings.filterwarnings('ignore')

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 54

    def data_set_loader(self):
        new_dirs = []
        files_paths = []
        for (dir_path, dir_names, file_names) in walk(self.dataset_path):
            for file in file_names:
                filename_ext = ['.flac', '.wav', '.mp3']
                if file.endswith(tuple(filename_ext)):
                    if not file.startswith("._"):
                        new_dirs.append(os.path.join(dir_path, file).split(self.dataset_path + '/')[1])
                        files_paths.append(os.path.join(dir_path, file))

        return files_paths

    def create_img_dataset(self):
        if not os.path.isdir('images'):
            os.mkdir('images')

        files_paths = self.data_set_loader()
        for file_path in files_paths:
            subdir = file_path.split('/')[-2]
            if not os.path.isdir('images/' + subdir):
                os.mkdir('images/' + subdir)

    def create_mel_spec(self):

        files_paths = self.data_set_loader()
        # print(files_paths)
        self.create_img_dataset()
        counter = 0
        for file_path in files_paths:
            file_name = file_path.split('/')[-1].split('.')[0]
            subdir = file_path.split('/')[-2]
            sample, sr = librosa.load(file_path)
            #print(f"{sample =}, ' in file: ', {file_path=}")

            # create mel spectrogram
            ax = plt.axes([0, 0, 1, 1], frameon=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            mel_s = librosa.feature.melspectrogram(sample, sr=sr, hop_length=self.hop_length, n_fft=self.n_fft,
                                                   n_mels=self.n_mels)
            mel_s_db = librosa.power_to_db(mel_s, ref=np.max)
            librosa.display.specshow(mel_s_db, sr=sr, ax=ax)
            plt.plot()
            plt.savefig('images/' + subdir + '/' + file_name + '.jpg')

            counter += 1
            print(f'Generate file: {file_name}, percent of generate images from dataset: '
                  f'{round(counter / len(files_paths) * 100, 2)}%')


create_melsp = CreateMelSpec(dataset_path='audio_balanced')
create_melsp.create_mel_spec()
