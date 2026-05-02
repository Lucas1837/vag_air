#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace std::chrono_literals;

class CameraFusionNode : public rclcpp::Node {
public:
    CameraFusionNode() : Node("camera_fusion_node") {
        // QoS Profile for high-bandwidth sensor data
        rclcpp::QoS qos = rclcpp::SensorDataQoS();

        pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/point_cloud", qos);

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        sub_l_.subscribe(this, "/lidar_l/points", qos.get_rmw_qos_profile());
        sub_r_.subscribe(this, "/lidar_r/points", qos.get_rmw_qos_profile());

        // Setup Approximate Time Synchronizer
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), sub_l_, sub_r_);
        sync_->registerCallback(std::bind(&CameraFusionNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2));
    }

private:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2> SyncPolicy;
    
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_l_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_r_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    void sync_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg_l, 
                       const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg_r) {
        
        try {
            // 1. Get Transforms (tf2_ros handles caching natively in C++)
            auto tf_l = tf_buffer_->lookupTransform("base_link", msg_l->header.frame_id, tf2::TimePointZero);
            auto tf_r = tf_buffer_->lookupTransform("base_link", msg_r->header.frame_id, tf2::TimePointZero);

            // 2. Transform the raw ROS messages directly (Highly optimized by pcl_ros)
            sensor_msgs::msg::PointCloud2 msg_l_tf, msg_r_tf;
            pcl_ros::transformPointCloud("base_link", *msg_l, msg_l_tf, *tf_buffer_);
            pcl_ros::transformPointCloud("base_link", *msg_r, msg_r_tf, *tf_buffer_);

            // 3. Convert to PCL objects for instant merging
            pcl::PointCloud<pcl::PointXYZI> pcl_l, pcl_r, pcl_fused;
            pcl::fromROSMsg(msg_l_tf, pcl_l);
            pcl::fromROSMsg(msg_r_tf, pcl_r);

            // 4. Merge the point clouds (This is practically instantaneous in C++)
            pcl_fused = pcl_l;
            pcl_fused += pcl_r; 

            // 5. Convert back and publish
            sensor_msgs::msg::PointCloud2 fused_msg;
            pcl::toROSMsg(pcl_fused, fused_msg);
            fused_msg.header.stamp = msg_l->header.stamp;
            fused_msg.header.frame_id = "base_link";

            pc_pub_->publish(fused_msg);

        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "TF Error: %s", ex.what());
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraFusionNode>());
    rclcpp::shutdown();
    return 0;
}