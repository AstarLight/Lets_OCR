# CTPN_Pytorch

A simple and light-weight implementation of CTPN, a classical text detection network in OCR area. This project is built in Pytorch, 
this greatly simplified the coding complexity and it makes easier to understand the CTPN inner mechanism.


## Train
1. Download this project, and prepare the OCR dataset(MSRA_TD500 & tianchi_icpr)
2. ```python train.py```，then training will be setup. Before that, you can modify the file ```config```to adjust some parameters.
3. The final training model will be save in ```model```directory.



## Inference
I propose two modes to inference. You can modify the infer mode in file ```test.sh```. We can use command ```sh test.sh```to start text detection.
1. "python infer.py random" means taht we will inference a batch of pictures in a specific directory path offered in file ```infer.py```. 
2. "python infer.py your_pic_path" means that we will only infer one picture according to your picture path.


## Reference
1. https://github.com/eragonruan/text-detection-ctpn
2. https://github.com/zwenwang/CTPN_Pytorch


## More details
