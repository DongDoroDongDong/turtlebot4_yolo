#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import PointStamped


class BoxDistanceNode(Node):
    def __init__(self):
        super().__init__('box_distance_node')

        # --------------------
        # Parameters
        # --------------------
        self.declare_parameter('detections_topic', '/yolov5/detections')
        self.declare_parameter('target_class', '')      # '' 이면 모든 클래스 허용
        self.declare_parameter('min_conf', 0.3)

        # 구(old) 1/d 모델용 파라미터 (호환성 유지 차원에서만 남겨둠)
        self.declare_parameter('ref_distance', 0.6)     # [m] 기준 거리
        self.declare_parameter('ref_height_px', 120.0)  # [px] 기준 픽셀 높이

        # 새 모델 d = a + b / S (S: bbox height[px])
        # 실측 데이터(1~3m) 기반으로 피팅한 기본값:
        #   a ≈ -0.235, b ≈ 102.376
        self.declare_parameter('model_a', -0.235)
        self.declare_parameter('model_b', 102.376)

        self.detections_topic = self.get_parameter('detections_topic').value
        self.target_class = self.get_parameter('target_class').value.strip()
        self.min_conf = float(self.get_parameter('min_conf').value)

        # 호환성용 (현재 계산에는 사용하지 않음)
        self.ref_distance = float(self.get_parameter('ref_distance').value)
        self.ref_height_px = float(self.get_parameter('ref_height_px').value)

        # 보정된 모델 계수
        self.model_a = float(self.get_parameter('model_a').value)
        self.model_b = float(self.get_parameter('model_b').value)

        # --------------------
        # ROS I/O
        # --------------------
        self.sub = self.create_subscription(
            Detection2DArray,
            self.detections_topic,
            self.detection_cb,
            10
        )

        self.dist_pub = self.create_publisher(
            PointStamped,
            '/box_distance',
            10
        )

        self.get_logger().info(
            'BoxDistanceNode started.\n'
            f'  detections_topic = {self.detections_topic}\n'
            f'  min_conf        = {self.min_conf}\n'
            f'  target_class    = "{self.target_class or "ANY"}"\n'
            f'  model: d = a + b / S, a = {self.model_a:.4f}, b = {self.model_b:.4f}'
        )

    # ---------------------------------------------------------------
    # Detection 선택 로직
    # ---------------------------------------------------------------
    def _select_best_detection(self, det_array: Detection2DArray) -> Detection2D | None:
        """
        Detection2DArray 중에서:
        - target_class 가 지정되어 있으면 해당 class_id만
        - min_conf 이상
        - 그 중 score 가장 높은 detection 하나 반환
        """
        best_det = None
        best_score = -1.0
        target_class = self.target_class

        for det in det_array.detections:
            if not det.results:
                continue

            # 결과 중 가장 score 높은 hypothesis 하나
            best_hyp = max(det.results, key=lambda h: h.hypothesis.score)
            score = float(best_hyp.hypothesis.score)
            cls_id = best_hyp.hypothesis.class_id

            if score < self.min_conf:
                continue

            if target_class and cls_id != target_class:
                continue

            if score > best_score:
                best_score = score
                best_det = det

        return best_det

    # ---------------------------------------------------------------
    # 콜백: 거리 추정 및 publish
    # ---------------------------------------------------------------
    def detection_cb(self, msg: Detection2DArray):
        if not msg.detections:
            return

        det = self._select_best_detection(msg)
        if det is None:
            # 조건에 맞는 detection 없음
            return

        # bbox 높이 (px)
        height_px = float(det.bbox.size_y)
        if height_px <= 1.0:
            # 의미 없는 값
            return

        # 보정된 거리 추정 모델:
        #   d_est = a + b / S,  S = bbox height [px]
        distance = self.model_a + self.model_b / height_px

        # 보호막: 음수 거리는 없으니까 0으로 컷
        if distance < 0.0:
            distance = 0.0

        # publish
        out = PointStamped()
        out.header = msg.header
        out.point.x = float(distance)   # 추정 거리 [m]
        out.point.y = height_px         # bbox 높이 [px]
        out.point.z = 0.0

        self.dist_pub.publish(out)

        self.get_logger().debug(
            f'dist_est={distance:.3f} m (h_px={height_px:.1f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = BoxDistanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()