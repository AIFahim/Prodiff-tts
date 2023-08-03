# Prodiff-tts

## Preprocessing Stages:

- **Step 1** : 
    - In the data folder, There are **3 Sub folders** - `binary , processed & raw.`
        - Inside of the raw folder, `LJSpeech-1.1` folder need to contains the training datasets as: \
        `wavs` folder(audio wave files) & `metadata.csv`
- **Step 2** :
    - Create `Textgrid` of the dataset using **Dfa Repo**: [https://github.com/as-ideas/DeepForcedAligner](https://github.com/as-ideas/DeepForcedAligner) & moved `Textgrid` files inside of the `processed/ljspeech` folder with name of the folder `mfa_outputs`.
- **Step 3** :
    - Run **binarization steps(for faster I/O)** as follow:
        ```sh
        export PYTHONPATH=.
        CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config configs/tts/lj/fs2.yaml
        ```
        **This will populate the training files inside of the `data/processed/ljspeech` folder**
- **Step 4** :
    - Then need to create metadata_phone.csv & dict.txt in the `data/processed/ljspeech` folder
        `metadata_phone.csv creation code:`
        ```sh
        ############################## metadata_phone.csv -> creation from metadata.csv #####################
        import pandas as pd

        # load the original data
        df = pd.read_csv('data/raw/LJSpeech-1.1/metadata.csv', sep='|')

        # define the base path for the audio files
        base_path = 'data/raw/LJSpeech-1.1/wavs/'

        # transform the data
        df['item_name'] = df['ID']
        df['spk'] = 'SPK1'  # or use appropriate speaker information if available
        df['txt'] = df['grapheme']
        df['txt_raw'] = df['grapheme']
        df['ph'] = '<BOS> ' + df['phoneme'] + ' <EOS>'
        df['wav_fn'] = base_path + df['ID'] + '.wav'

        # select the required columns and save to new csv file
        df = df[['item_name', 'spk', 'txt', 'txt_raw', 'ph', 'wav_fn']]
        df.to_csv('metadata_phone.csv', index=False)
        ```
        `dict.txt creation code:`
        ```sh
        import csv
        def get_unique_chars(csv_file, output_file):
            unique_chars = set()
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ph = row['ph'].split()
                    unique_chars.update(set(ph))
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for char in sorted(unique_chars):
                    f.write(f'{char} {char}\n')

        csv_file = 'data/processed/ljspeech/metadata_phone.csv'
        output_file = 'dict.txt'
        get_unique_chars(csv_file, output_file)
        ```
## Training Stages:
- **Step 1** : 
    - Move fastdiff vocoder weight ( [Link of the fastdiff trained 22k vocoder from Rongjiehuang huggingface space](https://huggingface.co/spaces/Rongjiehuang/ProDiff/tree/main/checkpoints) ) to the folder `checkpoints`
- **Step 2** :
    - Train the prodiff teacher in the cmd: 
        ```sh
        CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config modules/ProDiff/config/prodiff_teacher.yaml --exp_name ProDiff_Teacher --reset
        ```
    **Train weight will be save in the `checkpoints` folder**