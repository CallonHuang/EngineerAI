import numpy as np
import cv2
from rknn.api import RKNN
import torchvision.models as models
import torch

if __name__ == '__main__':
    model = './logistic.pt'
    input_size_list = [[28 * 28]]

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> Config model')
    rknn.config(target_platform='rv1109')
    print('done')

    # Load Pytorch model
    print('--> Loading model')
    ret = rknn.load_pytorch(model=model, input_size_list=input_size_list)
    if ret != 0:
        print('Load Pytorch model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    # ret = rknn.build(do_quantization=False, dataset='./dataset.txt', pre_compile=True)
    ret = rknn.build(do_quantization=False, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export RKNN model')
    # ret = rknn.export_rknn('./logistic_pre_compile.rknn')
    ret = rknn.export_rknn('./logistic.rknn')
    if ret != 0:
        print('Export logistic.rknn failed!')
        exit(ret)
    print('done')
    # exit(0)
    ret = rknn.load_rknn('./logistic.rknn')

    # Set inputs
    img = cv2.imread('./8.png')
    img = img[:, :, 0]
    img.reshape(-1)

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print(outputs)
    re = np.where(outputs == np.max(outputs))
    print(re[2])
    print('done')

    rknn.release()
