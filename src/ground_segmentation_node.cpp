#include <memory>
#include <limits>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

// TF2 Headers
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "visualization_msgs/msg/marker.hpp" 
#include <tf2/exceptions.h>
#include <tf2_eigen/tf2_eigen.hpp> 

// PCL Headers
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h> 
#include <pcl/filters/statistical_outlier_removal.h> 
#include <pcl/common/transforms.h>   

#include <Eigen/Dense>
#include <Eigen/Geometry> 

class GroundSegmentationNode : public rclcpp::Node
{
public:
    enum class Axis { X, Y, Z };

    // Standard ROS conventions for base_link
    Axis height_axis_ = Axis::Z;  
    Axis lateral_axis_ = Axis::Y; 
    
    std::string target_frame_ = "base_link"; 

    GroundSegmentationNode() : Node("ground_segmentation_node")
    {   
        // Parameters
        this->declare_parameter<double>("distance_threshold", 0.03);
        this->declare_parameter<double>("eps_angle_deg", 3.0);
        this->declare_parameter<double>("max_acceptable_angle_deg", 2.0);
        this->declare_parameter<int>("max_iterations", 200);
        this->declare_parameter<double>("sor_mean_k", 10);
        this->declare_parameter<double>("sor_std", 1);

        rclcpp::QoS best_effort_qos = rclcpp::QoS(10).best_effort();

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);


        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/point_cloud", best_effort_qos,
            std::bind(&GroundSegmentationNode::cloud_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/processed_pcd", 10);
        ground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_pcd", 10);
        tf_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/flat_pcd", 10);
        coeff_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/plane_coefficients", 10);
        plane_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/ground_plane_marker", 10);    
        
        // --- NEW PUBLISHER FOR DEBUGGING ---
        flat_2d_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug_flat_2d_pcd", 10);
        
        RCLCPP_INFO(this->get_logger(), "Optimized Ground Segmentation Node Started.");
    }

private:
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    inline float get_axis_val(const pcl::PointXYZ& pt, Axis axis) const {
        if (axis == Axis::X) return pt.x;
        if (axis == Axis::Y) return pt.y;
        return pt.z;
    }

    inline void set_axis_val(pcl::PointXYZ& pt, Axis axis, float value) const {
        if (axis == Axis::X) pt.x = value;
        else if (axis == Axis::Y) pt.y = value;
        else if (axis == Axis::Z) pt.z = value;
    }

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) 
    {
        if (msg->data.empty() || msg->width * msg->height == 0) return;

        double distance_threshold = this->get_parameter("distance_threshold").as_double();
        double eps_angle_deg = this->get_parameter("eps_angle_deg").as_double();
        double max_acceptable_angle_deg = this->get_parameter("max_acceptable_angle_deg").as_double();
        double sor_mean_k = this->get_parameter("sor_mean_k").as_double();
        double sor_std = this->get_parameter("sor_std").as_double();
        int max_iterations = this->get_parameter("max_iterations").as_int();

        // 1. Convert raw ROS msg directly to PCL FIRST
        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *raw_cloud);
        if (raw_cloud->points.empty()) return;

        // 2. Transform the raw PCL cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        try {
            geometry_msgs::msg::TransformStamped t = tf_buffer_->lookupTransform(
                target_frame_, msg->header.frame_id, tf2::TimePointZero, tf2::durationFromSec(0.1));
            Eigen::Isometry3d eigen_transform = tf2::transformToEigen(t);
            pcl::transformPointCloud(*raw_cloud, *cloud, eigen_transform.cast<float>());
        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN(this->get_logger(), "TF Error: %s", ex.what());
            return; 
        }

        // Publish transformed cloud for debugging
        sensor_msgs::msg::PointCloud2 flat_msg;
        pcl::toROSMsg(*cloud, flat_msg);
        flat_msg.header.frame_id = target_frame_;
        flat_msg.header.stamp = msg->header.stamp;
        tf_publisher_->publish(flat_msg);

        // 3. PassThrough Filter: Isolate the area of interest (Crop high structures)
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        if (height_axis_ == Axis::X) pass.setFilterFieldName("x");
        else if (height_axis_ == Axis::Y) pass.setFilterFieldName("y");
        else pass.setFilterFieldName("z");
        pass.setFilterLimits(-1.5, 0.8); // Adjust based on your sensor height to exclude canopy
        pass.filter(*cloud);

        if (cloud->points.size() < 10) return;

        // 4. Dual-RANSAC implementation
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(false);
        seg.setModelType(pcl::SACMODEL_PARALLEL_PLANE); 
        seg.setMethodType(pcl::SAC_RANSAC);    
        seg.setMaxIterations(max_iterations); 
        seg.setDistanceThreshold(distance_threshold); 
        seg.setEpsAngle(eps_angle_deg * (M_PI / 180.0)); 

        Eigen::Vector3f parallel_axis(1.0f, 1.0f, 1.0f);
        Eigen::Vector3f perpendicular_axis(0.0f, 0.0f, 0.0f);
        if (height_axis_ == Axis::Z) { parallel_axis[2] = 0.0f; perpendicular_axis[2] = 1.0f; }
        else if (height_axis_ == Axis::Y){ parallel_axis[1] = 0.0f; perpendicular_axis[1] = 1.0f; }
        else if (height_axis_ == Axis::X){ parallel_axis[0] = 0.0f; perpendicular_axis[0] = 1.0f; }
        seg.setAxis(parallel_axis);
        
        pcl::ModelCoefficients::Ptr coefficients1(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers1(new pcl::PointIndices());
        seg.setInputCloud(cloud);
        seg.segment(*inliers1, *coefficients1);

        if (inliers1->indices.empty()) return;

        float height1 = 0.0f;
        for (int idx : inliers1->indices) height1 += get_axis_val(cloud->points[idx], height_axis_);
        height1 /= inliers1->indices.size();

        std::vector<bool> is_inlier1(cloud->points.size(), false);
        for (int idx : inliers1->indices) is_inlier1[idx] = true;

        pcl::IndicesPtr remaining_indices(new std::vector<int>);
        for (size_t i = 0; i < cloud->points.size(); ++i) {
            if (!is_inlier1[i]) remaining_indices->push_back(i);
        }

        pcl::ModelCoefficients::Ptr coefficients2(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers2(new pcl::PointIndices());
        
        if (remaining_indices->size() >= 10) {
            seg.setIndices(remaining_indices);
            seg.segment(*inliers2, *coefficients2);
        }

        pcl::ModelCoefficients::Ptr coefficients = coefficients1;
        pcl::PointIndices::Ptr ground_inliers = inliers1;

        if (!inliers2->indices.empty()) {
            float height2 = 0.0f;
            for (int idx : inliers2->indices) height2 += get_axis_val(cloud->points[idx], height_axis_);
            height2 /= inliers2->indices.size();

            if (height2 < height1) {
                ground_inliers = inliers2;
                coefficients = coefficients2;
            }
        }
        
        // Manual Angle Validation
        Eigen::Vector3f plane_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        plane_normal.normalize(); 
        float dot_product = plane_normal.dot(perpendicular_axis);
        float angle_rad = std::acos(std::abs(dot_product));
        float angle_deg = angle_rad * (180.0f / M_PI);

        if (angle_deg > max_acceptable_angle_deg) return; 

        std_msgs::msg::Float32MultiArray coeff_msg;
        coeff_msg.data = { coefficients->values[0], coefficients->values[1], coefficients->values[2], coefficients->values[3] };
        coeff_publisher_->publish(coeff_msg);

        // RViz Plane Visualization
        visualization_msgs::msg::Marker plane_marker;
        plane_marker.header.frame_id = target_frame_;
        plane_marker.header.stamp = msg->header.stamp;
        plane_marker.ns = "ground_plane";
        plane_marker.id = 0;
        plane_marker.type = visualization_msgs::msg::Marker::CUBE;
        plane_marker.action = visualization_msgs::msg::Marker::ADD;

        float d = coefficients->values[3];
        plane_marker.pose.position.x = -d * plane_normal.x();
        plane_marker.pose.position.y = -d * plane_normal.y();
        plane_marker.pose.position.z = -d * plane_normal.z();

        Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), plane_normal);
        plane_marker.pose.orientation.x = q.x();
        plane_marker.pose.orientation.y = q.y();
        plane_marker.pose.orientation.z = q.z();
        plane_marker.pose.orientation.w = q.w();

        plane_marker.scale.x = 10.0; plane_marker.scale.y = 10.0; plane_marker.scale.z = 0.01; 
        plane_marker.color.r = 0.0f; plane_marker.color.g = 1.0f; plane_marker.color.b = 0.0f; plane_marker.color.a = 0.4f; 
        plane_marker_publisher_->publish(plane_marker);

        // Extract Ground Inliers
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(ground_inliers);
        extract.setNegative(false); 
        extract.filter(*ground_cloud);

        if (!ground_cloud->points.empty()) {
            sensor_msgs::msg::PointCloud2 ground_msg;
            pcl::toROSMsg(*ground_cloud, ground_msg);
            ground_msg.header.frame_id = target_frame_;
            ground_msg.header.stamp = msg->header.stamp; 
            ground_publisher_->publish(ground_msg);
        }

        // Extract Seedbeds / Objects (Outliers)
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_objects(new pcl::PointCloud<pcl::PointXYZ>());
        extract.setNegative(true); 
        extract.filter(*cloud_objects);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZ>());
        *cloud_2d = *cloud_objects; 
        for (auto& pt : cloud_2d->points) { set_axis_val(pt, height_axis_, 0.0f); }

        // --- NEW PUBLISH BLOCK: Debugging the flattened 2D cloud ---
        if (!cloud_2d->points.empty()) {
            sensor_msgs::msg::PointCloud2 debug_2d_msg;
            pcl::toROSMsg(*cloud_2d, debug_2d_msg);
            debug_2d_msg.header.frame_id = target_frame_;
            debug_2d_msg.header.stamp = msg->header.stamp; 
            flat_2d_publisher_->publish(debug_2d_msg);
        }
        // -------------------------------------------------------------

        pcl::PointCloud<pcl::PointXYZ>::Ptr pruned_cloud_2d(new pcl::PointCloud<pcl::PointXYZ>());
        if (!cloud_2d->points.empty()) {
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(cloud_2d);
            sor.setMeanK(sor_mean_k); 
            sor.setStddevMulThresh(sor_std); 
            sor.filter(*pruned_cloud_2d);
        }

        // Global Gap Filling Algorithm (Clustering Skipped)
        pcl::PointCloud<pcl::PointXYZ>::Ptr filled_gaps_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (!pruned_cloud_2d->points.empty()) {
            Axis axis_forward, axis_lateral;
            if (height_axis_ == Axis::X) { axis_forward = Axis::Y; axis_lateral = Axis::Z; }
            else if (height_axis_ == Axis::Y) { axis_forward = Axis::X; axis_lateral = Axis::Z; }
            else { axis_forward = Axis::X; axis_lateral = Axis::Y; }

            const float FILL_RESOLUTION = 0.01f; 
            std::map<int, std::pair<float, float>> boundary_map;

            for (const auto& pt : pruned_cloud_2d->points) {
                float fwd_val = get_axis_val(pt, axis_forward);
                float lat_val = get_axis_val(pt, axis_lateral);
                int fwd_bin = std::round(fwd_val / FILL_RESOLUTION);

                if (boundary_map.find(fwd_bin) == boundary_map.end()) boundary_map[fwd_bin] = {lat_val, lat_val};
                else {
                    if (lat_val < boundary_map[fwd_bin].first)  boundary_map[fwd_bin].first = lat_val;
                    if (lat_val > boundary_map[fwd_bin].second) boundary_map[fwd_bin].second = lat_val;
                }
            }

            for (const auto& kv : boundary_map) {
                int fwd_bin = kv.first;
                float min_lat = kv.second.first, max_lat = kv.second.second;

                if ((max_lat - min_lat) < 0.10f) continue; 

                float current_lat = min_lat + FILL_RESOLUTION;
                while (current_lat < max_lat) {
                    pcl::PointXYZ fill_pt;
                    set_axis_val(fill_pt, axis_forward, fwd_bin * FILL_RESOLUTION);
                    set_axis_val(fill_pt, axis_lateral, current_lat);
                    set_axis_val(fill_pt, height_axis_, 0.0f); 
                    filled_gaps_cloud->points.push_back(fill_pt);
                    current_lat += FILL_RESOLUTION;
                }
            }
            filled_gaps_cloud->width = filled_gaps_cloud->points.size();
            filled_gaps_cloud->height = 1;
            filled_gaps_cloud->is_dense = true;
        }

        // Final Publish
        if (!filled_gaps_cloud->points.empty()) {
            sensor_msgs::msg::PointCloud2 output_msg;
            pcl::toROSMsg(*filled_gaps_cloud, output_msg);
            output_msg.header.frame_id = target_frame_;
            output_msg.header.stamp = msg->header.stamp; 
            publisher_->publish(output_msg);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tf_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr coeff_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_marker_publisher_; 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr flat_2d_publisher_; // Debug Publisher
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GroundSegmentationNode>());
    rclcpp::shutdown();
    return 0;
}