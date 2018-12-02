# CTPN_Pytorch

A simple and light-weight implementation of CTPN, a classical text detection network in OCR area. This project is built in Pytorch, 
this greatly simplified the coding complexity and it makes easier to understand the CTPN inner mechanism.

## Environment
1. ```cd ctpn directory```
2. ```make```

## Train
1. Download this project, and prepare the OCR dataset([MSRA_TD500 & tianchi_icpr](https://pan.baidu.com/s/1kHTI5fg_oTOlQIVjOpFC8A))
2. ```python train.py```，then training will be setup. Before that, you can modify the file ```config```to adjust some parameters.
3. The final training model will be save in ```model```directory.



## Inference
I propose two modes to inference. You can modify the infer mode in file ```test.sh```. We can use command ```sh test.sh```to start text detection.
1. "python infer.py random" means taht we will inference a batch of pictures in a specific directory path offered in file ```infer.py```. 
2. "python infer.py your_pic_path" means that we will only infer one picture according to your picture path.


# Text detetcion show of this model
![](./image_show/999.png)

![](./image_show/998.png)

## Reference
1. https://github.com/eragonruan/text-detection-ctpn
2. https://github.com/zwenwang/CTPN_Pytorch


## More details
Pls visit my blog to get more details:https://www.cnblogs.com/skyfsm/p/10054386.html
