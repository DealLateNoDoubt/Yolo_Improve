import argparse

from ultralytics import YOLO


def train(opt):
    model = YOLO(opt.model)
    model.load(opt.init_weights)  # 加载初始化权重
    # model.load(opt.last_weights)
    model.train(
        data=opt.data,
        imgsz=opt.imgsz,
        epochs=opt.epochs,
        optimizer=opt.optimizer,
        name="Improve_1"
    )


def val(opt):
    model = YOLO(opt.best_weights)
    model.val(
        data=opt.data,
        split="val",
        imgsz=opt.imgsz,
        batch=opt.batch,
        name="Improve_1"
    )


def predict(opt):
    model = YOLO(opt.best_weights)
    model.predict(
        source=opt.predict_source,
        imgsz=opt.imgsz,
        save=True,  # 保存预测结果
        name="Improve_1"
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 权重相关
    parser.add_argument("--init_weights", type=str, default="yolov8m-pose.pt")
    parser.add_argument("--last_weights", type=str, default="../runs/pose/train/weights/last.pt")
    parser.add_argument("--best_weights", type=str, default="../runs/pose/train/weights/best.pt")

    # other
    parser.add_argument("--model", type=str, default="yolov8m-pose-bifpn.yaml")
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--data", type=str, default="coco-pose.yaml")
    parser.add_argument("--predict_source", type=str, default="../datasets/pose_test")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--resume", type=bool, default=True)   # 断点训练，Yolo初始化时选择last.pt
    parser.add_argument("--cache", type=bool, default=True)

    opt = parser.parse_args()


    train(opt)
    # val(opt)
    # predict(opt)