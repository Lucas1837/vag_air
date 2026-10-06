from argparse import ArgumentParser
from typing import Optional
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, Header
import numpy as np
from threading import Thread

# Import ROS 2 QoS Policy classes
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Dynamic Parameter Imports
from rcl_interfaces.msg import SetParametersResult

from ArducamDepthCamera import (
    ArducamCamera,
    Connection,
    DeviceType,
    FrameType,
    Control,
    DepthData,
)

class Option:
    cfg: Optional[str]

class DualTOFPublisher(Node):
    def __init__(self, options: Option):
        super().__init__("dual_arducam_node")

        # --- DYNAMIC PARAMETER SETUP FOR BOTH CAMERAS ---
        for i in range(2):
            self.declare_parameter(f'cam{i}_confidence_threshold', 100)
            self.declare_parameter(f'cam{i}_tx', 0.0)
            self.declare_parameter(f'cam{i}_ty', 0.27 if i == 0 else -0.27)
            self.declare_parameter(f'cam{i}_tz', 0.6)
            self.declare_parameter(f'cam{i}_roll', 0.0)
            self.declare_parameter(f'cam{i}_pitch', 0.7853982)
            self.declare_parameter(f'cam{i}_yaw', -0.7 if i == 0 else 0.7)

            # Retrieve initial values and assign as class attributes
            setattr(self, f'cam{i}_confidence_threshold', self.get_parameter(f'cam{i}_confidence_threshold').value)
            setattr(self, f'cam{i}_tx', self.get_parameter(f'cam{i}_tx').value)
            setattr(self, f'cam{i}_ty', self.get_parameter(f'cam{i}_ty').value)
            setattr(self, f'cam{i}_tz', self.get_parameter(f'cam{i}_tz').value)
            setattr(self, f'cam{i}_roll', self.get_parameter(f'cam{i}_roll').value)
            setattr(self, f'cam{i}_pitch', self.get_parameter(f'cam{i}_pitch').value)
            setattr(self, f'cam{i}_yaw', self.get_parameter(f'cam{i}_yaw').value)

        # Register callback for dynamic updates
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Initialize data structures for two cameras
        self.tofs = []
        self.widths = []
        self.heights = []
        self.points = [None, None]
        self.depth_msgs = [Float32MultiArray(), Float32MultiArray()]
        
        # Initialize hardware
        self.__init_cameras()

        # Create Best Effort QoS Profile
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create Publishers
        self.pc_pubs = []
        self.depth_pubs = []
        for i in range(2):
            self.pc_pubs.append(self.create_publisher(PointCloud2, f"point_cloud_cam{i}", best_effort_qos))
            self.depth_pubs.append(self.create_publisher(Float32MultiArray, f"depth_frame_cam{i}", best_effort_qos))

        self.running_ = True
        self.timer_ = self.create_timer(1 / 10, self.update)
        
        # Start independent threads to prevent frame blocking
        self.process_point_cloud_thrs = []
        for i in range(2):
            thr = Thread(target=self.__generateSensorPointCloud, args=(i,), daemon=True)
            self.process_point_cloud_thrs.append(thr)
            thr.start()

    def parameter_callback(self, params):
        """Catches live parameter changes from rqt and updates variables for both cameras."""
        for param in params:
            if hasattr(self, param.name):
                setattr(self, param.name, param.value)
        return SetParametersResult(successful=True)

    def __init_cameras(self):
        print("Initializing Dual Cameras...")
        csi_ports = [0, 8]  # Map camera 0 to CSI 0, and camera 1 to CSI 8
        
        for i in range(2):
            port = csi_ports[i]
            tof = ArducamCamera()
            
            # Open directly on the hardware port
            ret = tof.open(Connection.CSI, port)
            if ret != 0:
                print(f"Failed to open camera on CSI {port}. Error code: {ret}")
                raise Exception(f"Failed to initialize camera {i}")

            ret = tof.start(FrameType.DEPTH)
            if ret != 0:
                print(f"Failed to start camera {i}. Error code: {ret}")
                tof.close()
                raise Exception(f"Failed to start camera {i}")

            # Apply Range / Frequency separation unconditionally to prevent interference
            range_val = 2000 if i == 0 else 4000
            tof.setControl(Control.RANGE, range_val)

            info = tof.getCameraInfo()
            if info.device_type == DeviceType.HQVGA:
                width = info.width
                height = info.height
            elif info.device_type == DeviceType.VGA:
                width = info.width
                height = info.height // 10 - 1
            else:
                width = info.width
                height = info.height

            self.widths.append(width)
            self.heights.append(height)
            self.tofs.append(tof)
            
            freq_str = "2000" if i == 0 else "4000"
            print(f"Camera {i} success -> CSI {port}, width: {width}, height: {height}, Range: {freq_str}")

    def __generateSensorPointCloud(self, cam_idx):
        tof = self.tofs[cam_idx]
        width = self.widths[cam_idx]
        height = self.heights[cam_idx]
        
        while self.running_:
            frame = tof.requestFrame(200)
            if frame is not None and isinstance(frame, DepthData):
                fx = tof.getControl(Control.INTRINSIC_FX) / 100
                fy = tof.getControl(Control.INTRINSIC_FY) / 100
                
                # Force memory copy for safe NumPy slicing
                depth_buf = np.array(frame.depth_data, copy=True)
                confidence_buf = np.array(frame.confidence_data, copy=True)

                # Confidence threshold filtering
                conf_thresh = getattr(self, f"cam{cam_idx}_confidence_threshold")
                depth_buf[confidence_buf < conf_thresh] = 0

                # Health check / Obstruction detection
                valid_depths = depth_buf[depth_buf > 0]
                if len(valid_depths) > 0:
                    avg_depth = np.mean(valid_depths) / 1000.0
                else:
                    avg_depth = 0.0

                BLOCKAGE_THRESHOLD = 0.1  # Distance in meters

                if avg_depth < BLOCKAGE_THRESHOLD:
                    # Camera blocked -> output empty arrays to keep synchronizer active
                    self.points[cam_idx] = np.array([])
                    self.depth_msgs[cam_idx].data = []
                    tof.releaseFrame(frame)
                    continue 

                self.depth_msgs[cam_idx].data = (depth_buf.flatten() / 1000).tolist()

                # Convert depth values from millimeters to meters
                z = depth_buf / 1000.0
                z[z <= 0] = np.nan

                # Calculate pixel coordinate grid
                u = np.arange(width)
                v = np.arange(height)
                u, v = np.meshgrid(u, v)

                opt_x = (u - width / 2) * z / fx
                opt_y = (v - height / 2) * z / fy
                opt_z = z

                # Convert optical frame to standard ROS frame (X-forward, Y-left, Z-up)
                ros_x = opt_z
                ros_y = -opt_x
                ros_z = -opt_y

                local_points = np.stack((ros_x, ros_y, ros_z), axis=-1)

                # Direct map frame transformation via rotation & translation matrices
                roll = getattr(self, f"cam{cam_idx}_roll")
                pitch = getattr(self, f"cam{cam_idx}_pitch")
                yaw = getattr(self, f"cam{cam_idx}_yaw")
                tx = getattr(self, f"cam{cam_idx}_tx")
                ty = getattr(self, f"cam{cam_idx}_ty")
                tz = getattr(self, f"cam{cam_idx}_tz")

                cx, sx = np.cos(roll), np.sin(roll)
                cy, sy = np.cos(pitch), np.sin(pitch)
                cz, sz = np.cos(yaw), np.sin(yaw)

                Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
                R = Rz @ Ry @ Rx

                flat_points = local_points.reshape(-1, 3)
                map_points = flat_points @ R.T + np.array([tx, ty, tz])
                points_transformed = map_points.reshape(local_points.shape)

                self.points[cam_idx] = points_transformed[~np.isnan(points_transformed).any(axis=-1)]

                tof.releaseFrame(frame)

    def update(self):
        # Shared timestamp for accurate synchronization across both topics
        shared_stamp = self.get_clock().now().to_msg()

        for i in range(2):
            if self.points[i] is None:
                continue
            
            hdr = Header()
            hdr.stamp = shared_stamp
            hdr.frame_id = "map"

            if len(self.points[i]) == 0:
                pc2_msg_ = point_cloud2.create_cloud_xyz32(hdr, [])
            else:
                pc2_msg_ = point_cloud2.create_cloud_xyz32(hdr, self.points[i])

            self.pc_pubs[i].publish(pc2_msg_)
            
            if len(self.depth_msgs[i].data) > 0:
                self.depth_pubs[i].publish(self.depth_msgs[i]) 

    def stop(self):
        self.running_ = False
        for thr in self.process_point_cloud_thrs:
            thr.join()
        for tof in self.tofs:
            tof.stop()
            tof.close()

def main(args=None):
    rclpy.init(args=args)
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to camera configuration file")
 
    ns = parser.parse_args()
 
    options = Option()
    options.cfg = ns.cfg
 
    tof_publisher = DualTOFPublisher(options)

    try:
        rclpy.spin(tof_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        tof_publisher.stop()
        rclpy.shutdown()

if __name__ == "__main__":
    main()