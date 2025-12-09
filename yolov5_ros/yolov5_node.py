import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge

import torch
import cv2


class Yolov5Node(Node):
    def __init__(self):
        super().__init__('yolov5_node')

        # --------------------
        # Parameters
        # --------------------
        self.declare_parameter('image_topic', '/oakd/rgb/preview/image_raw')
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('device', '')  # ''=auto, 'cpu', 'cuda:0'
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('debug_topic', '/yolov5/debug_rgb')

        self.image_topic = self.get_parameter('image_topic').value
        self.conf_thres = float(self.get_parameter('conf_thres').value)
        self.device = self.get_parameter('device').value
        self.publish_debug = bool(self.get_parameter('publish_debug').value)
        self.debug_topic = self.get_parameter('debug_topic').value

        # --------------------
        # ROS I/O
        # --------------------
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            10
        )

        self.pub = self.create_publisher(
            Detection2DArray,
            '/yolov5/detections',
            10
        )

        # 디버그 이미지 퍼블리셔
        self.debug_pub = self.create_publisher(
            Image,
            self.debug_topic,
            10
        )

        # --------------------
        # YOLOv5n Model Load
        # --------------------
        self.get_logger().info('Loading YOLOv5n...')
        self.model = torch.hub.load(
            'ultralytics/yolov5',
            'yolov5n',
            pretrained=True
        )

        if self.device:
            self.model.to(self.device)

        self.model.conf = self.conf_thres
        self.get_logger().info('YOLOv5n loaded.')

        self._frame = 0

    def image_cb(self, msg: Image):
        # ROS Image -> OpenCV(BGR)
        img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 디버그용 이미지 (원본 복사)
        debug_bgr = img_bgr.copy()

        # YOLO expects RGB
        img_rgb = img_bgr[:, :, ::-1]

        # inference
        results = self.model(img_rgb)
        det_array = results.xyxy[0].cpu().numpy()  # (x1,y1,x2,y2,conf,cls)

        out_msg = Detection2DArray()
        out_msg.header = msg.header

        names = results.names  # class id -> name

        for x1, y1, x2, y2, conf, cls_id in det_array:
            # ---------------------------
            # (A) Detection2DArray 만들기
            # ---------------------------
            det = Detection2D()
            det.header = msg.header

            bbox = BoundingBox2D()
            bbox.center.position.x = float((x1 + x2) / 2.0)
            bbox.center.position.y = float((y1 + y2) / 2.0)
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)
            det.bbox = bbox

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(cls_id))
            hyp.hypothesis.score = float(conf)
            det.results.append(hyp)

            out_msg.detections.append(det)

            # ---------------------------
            # (B) 디버그 이미지에 bbox/라벨 그리기
            # ---------------------------
            if self.publish_debug:
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                class_name = names[int(cls_id)] if names else str(int(cls_id))
                label = f"{class_name} {conf:.2f}"

                cv2.rectangle(debug_bgr, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                cv2.putText(
                    debug_bgr, label, (x1i, max(y1i - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
                )

        # 검출 publish
        self.pub.publish(out_msg)

        # ---------------------------
        # (C) 디버그 RGB 이미지 publish
        # ---------------------------
        if self.publish_debug and self.debug_pub.get_subscription_count() > 0:
            debug_rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB)
            debug_msg = self.bridge.cv2_to_imgmsg(debug_rgb, encoding='rgb8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)


def main():
    rclpy.init()
    node = Yolov5Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

