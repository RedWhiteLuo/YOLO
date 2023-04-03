# 确定目标：
    在 Convert 文件夹中生成：两个文件夹+四个文件:
    {images labels [test.txt] [train.txt] [trainval.txt] [val.txt]}
    # 目录结构：
    labelimg
        └── data
            ├── Convert
            |   ├──images       * 执行完_数据整理_后把 Picture 中的复制进来就行
            |   └──labels       * 这里生成了 XML 转换出来的 TXT 格式的数据标注
            ├──Picture          # 这里放用来训练的图片
            └──Picture_XML      # 这里放标注获得的XML 格式文件


# 标注数据
    1.打开labelimg.exe
    2.标注数据，记得选择好保存的标注数据的位置

# 数据整理
    1.将获取的图片放入./data/Picture/ 中, 将标注后获得的XML格式标签放入./data/Picture_XML/ 中
    2.利用 [del_space.py] 将Picture 中的文件名中的空格替换成下划线
    3.利用 [into_different_data.py] 将Picture 中的文件按照比例划分为不同的数据集
        执行完第三步后会在./data/Convert/ 中获得
        {train.txt      val.txt     trainval.txt    test.txt} 四个文件
    4.利用 [xml2txt.py] 将./Picture_XML/ 中的XML 格式的文件转换为 txt 格式
        这一步生成的文件都会在./data/Convert/labels/ 中

# 数据集配置
    1. 确认数据集
        ./data/Convert/ 下一共有: 两个文件夹,四个文件
        {images labels [test.txt] [train.txt] [trainval.txt] [val.txt]}
        将所有的图片复制进 images 中；labels 中应该是每个图片对应的标签
    2. 自定义配置yaml文件
        1. yolov5 默认使用的是 coco128.yaml 默认是在 'YOLO根目录'/data/ 下面, 复制一份
        2. 更改这份复制的coco128.yaml 中的2个参数：
            path: D:/0_AI_Learning/ApexDataset  # 更改为 ./data/Convert/ 文件夹的绝对路径
                比如  'D:\AI_Learning\labelImg\data.yaml'
            train: train.txt  # train images (relative to 'path') 128 images 不用更改
            val: val.txt  # val images (relative to 'path') 128 images  不用更改
            test:  # test images (optional) 不用更改

            # Classes   # 更改成训练时的类别， 注意保持顺序
            names:
              0: A
              1: B
              2: C
       或者使用现有的  data.yaml 本文件的同级目录下， 不要移动到别的文件夹

# 使用数据集：
    更改train.py 中的参数：
        parser.add_argument('--data', type=str, default=ROOT / '***.yaml', help='dataset.yaml path')
        将'***.yaml'改为 #数据集配置 中获得的 yaml 文件的绝对路径，如：
        'D:\AI_Learning\labelImg\data.yaml'

