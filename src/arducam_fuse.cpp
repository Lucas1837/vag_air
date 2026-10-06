#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

using namespace std::chrono_literals;

class CameraFusionNode : public rclcpp::Node {
public:
    CameraFusionNode() : Node("camera_fusion_node") {
        this->declare_parameter<bool>("enable_downsampling", true);
        this->declare_parameter<double>("voxel_leaf_size", 0.025); 

        rclcpp::QoS qos = rclcpp::SensorDataQoS();

        pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/point_cloud", qos);
        
        sub_pi5_.subscribe(this, "/point_cloud_pi5", qos.get_rmw_qos_profile());
        sub_pi4_.subscribe(this, "/point_cloud_pi4", qos.get_rmw_qos_profile());

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(50), sub_pi5_, sub_pi4_);
        sync_->registerCallback(std::bind(&CameraFusionNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2));
        
        RCLCPP_INFO(this->get_logger(), "Camera Fusion Node Started.");
    }

private:
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2> SyncPolicy;
    
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_pi5_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_pi4_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;

    void sync_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg_pi5, 
                       const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg_pi4) {
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pi5(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pi4(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg_pi5, *pcl_pi5);
        pcl::fromROSMsg(*msg_pi4, *pcl_pi4);

        // Append
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_fused(new pcl::PointCloud<pcl::PointXYZ>());
        *pcl_fused = *pcl_pi5;
        *pcl_fused += *pcl_pi4; 

        sensor_msgs::msg::PointCloud2 fused_msg;
        bool enable_downsampling = this->get_parameter("enable_downsampling").as_bool();

        if (enable_downsampling) {
            double leaf_size = this->get_parameter("voxel_leaf_size").as_double();
            pcl::PointCloud<pcl::PointXYZ> pcl_downsampled;
            pcl::VoxelGrid<pcl::PointXYZ> vg;
            vg.setInputCloud(pcl_fused);
            vg.setLeafSize(leaf_size, leaf_size, leaf_size);
            vg.filter(pcl_downsampled);
            pcl::toROSMsg(pcl_downsampled, fused_msg);
        } else {

            pcl::toROSMsg(*pcl_fused, fused_msg);
        }
        
        fused_msg.header.stamp = msg_pi5->header.stamp;
        fused_msg.header.frame_id = "map";

        pc_pub_->publish(fused_msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraFusionNode>());
    rclcpp::shutdown();
    return 0;
}