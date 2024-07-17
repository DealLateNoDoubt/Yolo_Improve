1-基于官方ultralytics内容，改进基础Yolov8结构 ------ (yolov8m-pose-bifpn.yaml)
  (1)调整Yolov8-Neck网络: 将PANFPN调整为BiFPN，加强特征提取；
  (2)优化卷积操作，使用Ghostconv来进行轻量化调整；

2-使用coco-pose官方数据集来进行关键点检测预训练，训练50epoch获取相关权重信息；

3-使用labelme来对动漫图片进行数据标注(标注格式符合coco-pose数据集规范格式)、编辑批处理，将json标签文件转化为yolo格式，且要因用于预训练权重后进行二次训练，则需要标签格式与coco-pose官方格式一致；

4-基于coco-pose预训练权重对自定义数据集进行再次训练，训练300epoch；通过result(mAP50、mAP50-95)等数据信息观察，模型已达到收敛情况；

5-根据训练完毕的best-self.pt权重数据，对动漫图片进行predict,能符合获取到头部关键点数据信息以及其他关键点信息。
