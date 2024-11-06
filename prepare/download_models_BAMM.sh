mkdir log
cd log
mkdir t2m
cd t2m
mkdir t2m
cd t2m

# echo -e "Downloading pretrained BAMM"
# gdown --fuzzy https://drive.google.com/file/d/1vo0PcYOHzCdPoDk5SpBA1llvK_50GTlv/view?usp=sharing
# echo -e "Unzipping 2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd.zip"
# unzip 2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd.zip

# echo -e "Cleaning 2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd.zip"
# rm 2024-02-14-14-27-29_8_GPT_officialTrans_2iterPrdictEnd.zip

# echo -e "Downloading done!"

cd ../../
mkdir vq
cd vq
mkdir t2m
cd ../../
cp -r ./checkpoints/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2 ./log/vq/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2
echo -e "copy pretrain momask to log!"