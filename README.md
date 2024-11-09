# [BAMM: Bidirectional Autoregressive Motion Model](https://exitudio.github.io/BAMM-page/)  (ECCV 2024)

[![arXiv](https://img.shields.io/badge/arXiv-<2403.19435>-<COLOR>.svg)](https://arxiv.org/abs/2403.19435)

The official PyTorch implementation of the paper [**"BAMM: Bidirectional Autoregressive Motion Model"**](https://arxiv.org/abs/2403.19435).

Please visit our [**webpage**](https://exitudio.github.io/BAMM-page/) for more details.


If our project is helpful for your research, please consider citing :
``` 
@inproceedings{pinyoanuntapong2024bamm,
  title={BAMM: Bidirectional Autoregressive Motion Model}, 
  author={Ekkasit Pinyoanuntapong and Muhammad Usama Saleem and Pu Wang and Minwoo Lee and Srijan Das and Chen Chen}, 
  booktitle="Computer Vision -- ECCV 2024",
  year={2024},
}
```
## 1. Setup Env & Download Pre-train MoMask
Our model is devloped based on [MoMask](https://github.com/EricGuo5513/momask-codes). You can follow the setup instructions provided there, or refer to the steps below. (The environment setup is the same but we change conda env name to "BAMM")
<details>
  
### 1. Conda Environment
```
conda env create -f environment.yml
conda activate BAMM
pip install git+https://github.com/openai/CLIP.git
```
We test our code on Python 3.7.13 and PyTorch 1.7.1

#### Alternative: Pip Installation
<details>
We provide an alternative pip installation in case you encounter difficulties setting up the conda environment.

```
pip install -r requirements.txt
```
We test this installation on Python 3.10

</details>

### 2. Models and Dependencies

#### Download Pre-trained Models
```
bash prepare/download_models.sh
```

#### Download Evaluation Models and Gloves
For evaluation only.
```
bash prepare/download_evaluator.sh
bash prepare/download_glove.sh
```

#### Troubleshooting
To address the download error related to gdown: "Cannot retrieve the public link of the file. You may need to change the permission to 'Anyone with the link', or have had many accesses". A potential solution is to run `pip install --upgrade --no-cache-dir gdown`, as suggested on https://github.com/wkentaro/gdown/issues/43. This should help resolve the issue.

#### (Optional) Download Manually
Visit [[Google Drive]](https://drive.google.com/drive/folders/1b3GnAbERH8jAoO5mdWgZhyxHB73n23sK?usp=drive_link) to download the models and evaluators mannually.

### 3. Get Data

You have two options here:
* **Skip getting data**, if you just want to generate motions using *own* descriptions.
* **Get full data**, if you want to *re-train* and *evaluate* the model.

**(a). Full data (text + motion)**

**HumanML3D** - Follow the instruction in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then copy the result dataset to our repository:
```
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```
**KIT**-Download from [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then place result in `./dataset/KIT-ML`

#### 

</details>

## 2. Download Pre-train BAMM and Move MoMask to "log" folder
```
bash prepare/download_models_BAMM.sh
```

## 3. Generate

```
python gen_t2m.py \
    --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw \
    --name 2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd \
    --text_prompt "the person crouches and walks forward." \
    --motion_length -1 \
    --repeat_times 3 \
    --gpu_id 0 \
    --seed 1 \
    --ext generation_name_nopredlen
```
- --text_prompt: text
- --motion_length: -1 the model will automatically predict length otherwise it will use the input as a length
- --seed: Sets the random seed for generation, enabling reproducibility of results across different runs. With a fixed seed, you will get the same generation in each run, making results consistent.
**Note:**
If you'd like to see variation in the generation results, see the generation from other "repeat_times" or change the seed value.
- --ext: generation name
It will generate to "generation" directory.


## 4. Evaluation
<details>



```
python eval_t2m_trans_res.py \
    --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw \
    --name 2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd \
    --gpu_id 1 \
    --ext LOG_NAME
```
</details>