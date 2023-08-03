# Prodiff-tts

## Preprocessing Stages:

- **Step 1** : 
    - In the data folder, There are **3 Sub folders** - `binary , processed & raw.`
        - Inside of the raw folder, `LJSpeech-1.1` folder need to contains the training datasets as: \
        `wavs` folder(audio wave files) & `metadata.csv`
- **Step 2** :
    - Create `Textgrid` of the dataset using **Dfa Repo**: [https://github.com/as-ideas/DeepForcedAligner](https://github.com/as-ideas/DeepForcedAligner) & moved `Textgrid` inside of the `processed/ljspeech` folder with name `mfa_outputs`.
- **Step 3** :
    - Run **binarization steps(for faster I/O)** as follow:
    ```sh
    export PYTHONPATH=.
    CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config configs/tts/lj/fs2.yaml
    ```