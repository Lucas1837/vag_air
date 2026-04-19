#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import message_filters

# TF2 imports
from tf2_ros import Buffer, TransformListener, TransformException

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

        self.get_logger().info("Lightning Camera Fusion Node Started. Waiting for point clouds...")

    def extract_xyz(self, cloud_msg):
        """Extracts X, Y, Z instantly using native memory buffers."""
        if len(cloud_msg.data) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Dynamically find the byte offsets for safety
        x_offset = next((f.offset for f in cloud_msg.fields if f.name == 'x'), 0)
        y_offset = next((f.offset for f in cloud_msg.fields if f.name == 'y'), 4)
        z_offset = next((f.offset for f in cloud_msg.fields if f.name == 'z'), 8)

        # Create a dtype to instantly map the raw binary array
        dtype = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': ['<f4', '<f4', '<f4'],
            'offsets': [x_offset, y_offset, z_offset],
            'itemsize': cloud_msg.point_step
        })

        pts = np.frombuffer(cloud_msg.data, dtype=dtype)
        
        # Filter out NaN values efficiently
        valid_mask = ~np.isnan(pts['x']) & ~np.isnan(pts['y']) & ~np.isnan(pts['z'])
        pts = pts[valid_mask]

        return np.column_stack((pts['x'], pts['y'], pts['z']))

    def transform_numpy_array(self, points, transform_msg):
        """Applies a ROS TransformStamped to an Nx3 numpy array."""
        t = transform_msg.transform.translation
        q = transform_msg.transform.rotation

        translation = np.array([t.x, t.y, t.z])
        x, y, z, w = q.x, q.y, q.z, q.w
        
        rotation_matrix = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

        return np.dot(points, rotation_matrix.T) + translation

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

            # 2. Extract instantly via memory buffer
            points_l = self.extract_xyz(msg_l)
            points_r = self.extract_xyz(msg_r)

            if points_l.size == 0 or points_r.size == 0:
                return

            # 3. Apply the transforms
            points_l_tf = self.transform_numpy_array(points_l, transform_l)
            points_r_tf = self.transform_numpy_array(points_r, transform_r)

            # 4. Append the two point clouds together
            all_points = np.vstack((points_l_tf, points_r_tf))

            # 5. Fast Voxel Grid Downsampling (Much faster than float rounding)
            voxel_size = 0.02 # 2cm resolution
            voxels = np.floor(all_points / voxel_size).astype(np.int32)
            _, unique_indices = np.unique(voxels, axis=0, return_index=True)
            filtered_points = all_points[unique_indices].astype(np.float32)

            # 6. Create the fused message instantly via .tobytes()
            fused_msg = PointCloud2()
            fused_msg.header.stamp = msg_l.header.stamp
            fused_msg.header.frame_id = target_frame
            fused_msg.height = 1
            fused_msg.width = filtered_points.shape[0]
            fused_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            fused_msg.is_bigendian = False
            fused_msg.point_step = 12  # 3 floats * 4 bytes
            fused_msg.row_step = fused_msg.point_step * fused_msg.width
            fused_msg.is_dense = True
            
            # Map raw binary array straight to message payload
            fused_msg.data = filtered_points.tobytes()

            # 7. Publish
            self.pc_pub.publish(fused_msg)

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