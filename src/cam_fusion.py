#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import message_filters

# TF2 imports
from tf2_ros import Buffer, TransformListener, TransformException

# Point cloud and math utilities
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class CameraFusionNode(Node):
    def __init__(self):
        super().__init__('camera_fusion_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', 10)

        self.sub_l = message_filters.Subscriber(self, PointCloud2, '/lidar_l/points')
        self.sub_r = message_filters.Subscriber(self, PointCloud2, '/lidar_r/points')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_l, self.sub_r], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("Camera Fusion Node Started. Waiting for point clouds...")

    def extract_xyz(self, cloud_msg):
        """Safely extracts x, y, z from a PointCloud2 message into a standard Nx3 numpy array."""
        # Read the points (without forcing float32 yet)
        pts = np.array(list(pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)))
        
        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float32)
            
        # Check if ROS returned a structured array (with labels 'x', 'y', 'z')
        if pts.dtype.names is not None:
            # Strip the labels and stack them side-by-side into raw numbers
            pts = np.column_stack((pts['x'], pts['y'], pts['z']))
        
        # Ensure it is standard N x 3 float array
        return pts.astype(np.float32).reshape(-1, 3)

    def transform_numpy_array(self, points, transform_msg):
        """Applies a ROS TransformStamped to an Nx3 numpy array manually."""
        t = transform_msg.transform.translation
        q = transform_msg.transform.rotation

        translation = np.array([t.x, t.y, t.z])

        x, y, z, w = q.x, q.y, q.z, q.w
        rotation_matrix = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

        transformed_points = np.dot(points, rotation_matrix.T) + translation
        return transformed_points

    def sync_callback(self, msg_l, msg_r):
        target_frame = 'base_link'

        try:
            # 1. Look up the transforms
            transform_l = self.tf_buffer.lookup_transform(
                target_frame, msg_l.header.frame_id, rclpy.time.Time()
            )
            transform_r = self.tf_buffer.lookup_transform(
                target_frame, msg_r.header.frame_id, rclpy.time.Time()
            )

            # 2. Safely extract only X, Y, Z into standard numpy arrays
            points_l = self.extract_xyz(msg_l)
            points_r = self.extract_xyz(msg_r)

            if points_l.size == 0 or points_r.size == 0:
                return

            # 3. Apply the transforms
            points_l_tf = self.transform_numpy_array(points_l, transform_l)
            points_r_tf = self.transform_numpy_array(points_r, transform_r)

            # 4. Append the two point clouds together
            all_points = np.vstack((points_l_tf, points_r_tf))

            # 5. Filter out "repeating" points (1cm resolution grid)
            precision_decimals = 2 
            rounded_points = np.round(all_points, decimals=precision_decimals)
            _, unique_indices = np.unique(rounded_points, axis=0, return_index=True)
            filtered_points = all_points[unique_indices]

            # 6. Create the new fused PointCloud2 message
            header = Header()
            header.stamp = msg_l.header.stamp  # <-- Inherit the exact time from the Lidar
            header.frame_id = target_frame
            
            fused_cloud_msg = pc2.create_cloud_xyz32(header, filtered_points.tolist())

            # 7. Publish
            self.pc_pub.publish(fused_cloud_msg)

        except TransformException as ex:
            self.get_logger().warn(f'Could not transform point clouds: {ex}')

def main(args=None):
    rclpy.init(args=args)
    node = CameraFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()