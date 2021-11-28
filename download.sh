wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Py2nadGXyeNBwPDzrrrPaRsYyy-UVZE9' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Py2nadGXyeNBwPDzrrrPaRsYyy-UVZE9" -O Models.zip && rm -rf /tmp/cookies.txt
unzip Models.zip
rm Models.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1SQlYawHkBggs6smS3FGH4nLrfGrp5A-d' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1SQlYawHkBggs6smS3FGH4nLrfGrp5A-d" -O RPN_results.zip && rm -rf /tmp/cookies.txt
unzip RPN_results.zip
rm RPN_results.zip

cd Dataset

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hx6eErHtuR7TCGlQMyl2_03gXDTQw3Qm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hx6eErHtuR7TCGlQMyl2_03gXDTQw3Qm" -O Occluded_Vehicles.zip && rm -rf /tmp/cookies.txt
unzip Occluded_Vehicles.zip
rm Occluded_Vehicles.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1DeaVbE_CwdIjogIPS3jCRKMOSqTGVloA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DeaVbE_CwdIjogIPS3jCRKMOSqTGVloA" -O kitti.zip && rm -rf /tmp/cookies.txt
unzip kitti.zip
rm kitti.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1n1vvOaT701dAttxxGeMKQa7k9OD_Ds51' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1n1vvOaT701dAttxxGeMKQa7k9OD_Ds51" -O COCO.zip && rm -rf /tmp/cookies.txt
unzip COCO.zip
rm COCO.zip

cd COCO
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
rm train2014.zip

wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm val2014.zip