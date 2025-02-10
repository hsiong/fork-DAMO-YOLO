
```shell
 python --version #3.12
 pip install D:\app\torch-2.3.1+cu121-cp312-cp312-win_amd64.whl
 pip uninstall torchvision torchaudio
 pip install torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
 
 pip uninstall numpy
 pip install numpy==1.26.4 # 模型使用的老版本 numpy
 
 pip install -r requirements.txt # pip install -e .

```

+ run [demo_infer_single.py](demo_infer_single.py)

>  pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI # for Linux

# Error
```shell
| __main__:<module>:356 - An error has been caught in function '<module>', process 'MainProcess' (15252), thread 'MainThread' (26204):
Traceback (most recent call last):

> File "tools\demo.py", line 356, in <module>
    main()
    e<function main at 0x0000023A43000280>

  File "tools\demo.py", line 321, in main
    vis_res = infer_engine.visualize(origin_img, bboxes, scores, cls_inds, conf=args.conf, save_name=os.path.basename(args.path), save_result=args.save_result)
              │            │         │           │       │       │              │    │               │  │    │        │    │                  │    └ True
              │            │         │           │       │       │              │    │               │  │    │        │    │                  └ Namespace(camid=0, conf=0.6, config_file='./configs/damoyolo_tinynasL45_L.py', device='cuda', end2end=False, engine='./config...
              │            │         │           │       │       │              │    │               │  │    │        │    └ './assets/dog.jpg'
              │            │         │           │       │       │              │    │               │  │    │        └ Namespace(camid=0, conf=0.6, config_file='./configs/damoyolo_tinynasL45_L.py', device='cuda', end2end=False, engine='./config...
              │            │         │           │       │       │              │    │               │  │    └ <function basename at 0x0000023A15D8F4C0>
              │            │         │           │       │       │              │    │               │  └ <module 'ntpath' from 'C:\\Users\\lenovo\\AppData\\Local\\Programs\\Python\\Python38\\lib\\ntpath.py'>
              │            │         │           │       │       │              │    │               └ <module 'os' from 'C:\\Users\\lenovo\\AppData\\Local\\Programs\\Python\\Python38\\lib\\os.py'>
              │            │         │           │       │       │              │    └ 0.6
              │            │         │           │       │       │              └ Namespace(camid=0, conf=0.6, config_file='./configs/damoyolo_tinynasL45_L.py', device='cuda', end2end=False, engine='./config...
              │            │         │           │       │       └ tensor([ 1., 16.,  7.,  2., 58.,  3., 58.,  2.,  3., 58.,  0., 58., 58., 58.,
              │            │         │           │       │                 58., 58.,  5., 72., 60., 58., 58.,  2.,...
              │            │         │           │       └ tensor([0.9152, 0.8831, 0.7081, 0.4480, 0.3059, 0.2483, 0.2356, 0.2217, 0.2134,
              │            │         │           │                 0.1960, 0.1719, 0.1706, 0.1617, 0.155...
              │            │         │           └ tensor([[ 1.2609e+02,  1.3322e+02,  5.6658e+02,  4.2027e+02],
              │            │         │                     [ 1.3133e+02,  2.2133e+02,  3.0998e+02,  5.4202e+02],
              │            │         │              ...
              │            │         └ array([[[ 57,  58,  50],
              │            │                   [ 58,  59,  51],
              │            │                   [ 60,  61,  53],
              │            │                   ...,
              │            │                   [142,  89,  47],
              │            │                   [ 88...
              │            └ <function Infer.visualize at 0x0000023A43000160>
              └ <__main__.Infer object at 0x0000023A42FF2EE0>

  File "tools\demo.py", line 249, in visualize
    vis_img = vis(image, bboxes, scores, cls_inds, conf, self.class_names)
              │   │      │       │       │         │     │    └ ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'sto...
              │   │      │       │       │         │     └ <__main__.Infer object at 0x0000023A42FF2EE0>
              │   │      │       │       │         └ 0.6
              │   │      │       │       └ tensor([ 1., 16.,  7.,  2., 58.,  3., 58.,  2.,  3., 58.,  0., 58., 58., 58.,
              │   │      │       │                 58., 58.,  5., 72., 60., 58., 58.,  2.,...
              │   │      │       └ tensor([0.9152, 0.8831, 0.7081, 0.4480, 0.3059, 0.2483, 0.2356, 0.2217, 0.2134,
              │   │      │                 0.1960, 0.1719, 0.1706, 0.1617, 0.155...
              │   │      └ tensor([[ 1.2609e+02,  1.3322e+02,  5.6658e+02,  4.2027e+02],
              │   │                [ 1.3133e+02,  2.2133e+02,  3.0998e+02,  5.4202e+02],
              │   │         ...
              │   └ array([[[ 57,  58,  50],
              │             [ 58,  59,  51],
              │             [ 60,  61,  53],
              │             ...,
              │             [142,  89,  47],
              │             [ 88...
              └ <function vis at 0x0000023A42F5AA60>

  File "d:\project\python\damo-yolo\damo\utils\visualize.py", line 30, in vis
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    │   │         │     │   │     │   │    └ [216, 82, 24]
    │   │         │     │   │     │   └ 420
    │   │         │     │   │     └ 566
    │   │         │     │   └ 133
    │   │         │     └ 126
    │   │         └ array([[[ 57,  58,  50],
    │   │                   [ 58,  59,  51],
    │   │                   [ 60,  61,  53],
    │   │                   ...,
    │   │                   [142,  89,  47],
    │   │                   [ 88...
    │   └ <built-in function rectangle>
    └ <module 'cv2' from 'D:\\project\\python\\DAMO-YOLO\\venv38\\lib\\site-packages\\cv2\\__init__.py'>

cv2.error: OpenCV(4.10.0) :-1: error: (-5:Bad argument) in function 'rectangle'
> Overload resolution failed:
>  - img marked as output argument, but provided NumPy array marked as readonly
>  - Expected Ptr<cv::UMat> for argument 'img'
>  - img marked as output argument, but provided NumPy array marked as readonly
>  - Expected Ptr<cv::UMat> for argument 'img'
```

```python
    # 确保图像数组是可写的
    img = np.copy(img)
```

+ ModuleNotFoundError: No module named 'distutils' (python3.12)
```
pip install setuptools
```

+ ​	size mismatch xxxx

```
config file mismatch
```





# License

This project is licensed under the Apache-2.0 license. See the [LICENSE](LICENSE) file for details.

# Acknowledgements

This project includes software from [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO) by TinyVision.

# Citation

If you use DAMO-YOLO in your research, please cite our work by using the following BibTeX entry:

```bibtex
@article{damoyolo,
  title={DAMO-YOLO: A Report on Real-Time Object Detection Design},
  author={Xianzhe Xu, Yiqi Jiang, Weihua Chen, Yilun Huang, Yuan Zhang and Xiuyu Sun},
  journal={arXiv preprint arXiv:2211.15444v2},
  year={2022},
}

@inproceedings{sun2022mae,
  title={Mae-det: Revisiting maximum entropy principle in zero-shot nas for efficient object detection},
  author={Sun, Zhenhong and Lin, Ming and Sun, Xiuyu and Tan, Zhiyu and Li, Hao and Jin, Rong},
  booktitle={International Conference on Machine Learning},
  pages={20810--20826},
  year={2022},
  organization={PMLR}
}

@inproceedings{jiang2022giraffedet,
 title={GiraffeDet: A Heavy-Neck Paradigm for Object Detection},
 author={Yiqi Jiang and Zhiyu Tan and Junyan Wang and Xiuyu Sun and Ming Lin and Hao Li},
 booktitle={International Conference on Learning Representations},
 year={2022},
}

```

+ 
