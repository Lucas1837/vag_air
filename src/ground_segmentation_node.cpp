#include <memory>
#include <limits>
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

// TF2 Headers for PointCloud Transformation
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "visualization_msgs/msg/marker.hpp" 
#include <tf2/exceptions.h>

// PCL Headers
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h> 
#include <pcl/filters/statistical_outlier_removal.h> 

#include <Eigen/Dense>
#include <Eigen/Geometry> 

class GroundSegmentationNode : public rclcpp::Node
{
public:
    // ==========================================
    // MODULAR CONFIGURATION
    // ==========================================
    enum class Axis { X, Y, Z };

    // Standard ROS conventions for base_link:
    Axis height_axis_ = Axis::Z;  
    Axis lateral_axis_ = Axis::Y; 
    
    std::string target_frame_ = "base_link"; 

    GroundSegmentationNode() : Node("ground_segmentation_node")
    {   
        // --- DECLARE ROS 2 PARAMETERS ---
        this->declare_parameter<double>("distance_threshold", 0.03);
        this->declare_parameter<double>("eps_angle_deg", 3.0);
        this->declare_parameter<double>("max_acceptable_angle_deg", 2.0);
        this->declare_parameter<int>("max_iterations", 200);
        // --------------------------------

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/point_cloud", 10,
            std::bind(&GroundSegmentationNode::cloud_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/processed_pcd", 10);
        ground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_pcd", 10);
        tf_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/flat_pcd", 10);
        coeff_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/plane_coefficients", 10);
        plane_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/ground_plane_marker", 10);    
        
        RCLCPP_INFO(this->get_logger(), "Ground Segmentation Node Started.");
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

        // --- FETCH LATEST PARAMETER VALUES ---
        double distance_threshold = this->get_parameter("distance_threshold").as_double();
        double eps_angle_deg = this->get_parameter("eps_angle_deg").as_double();
        double max_acceptable_angle_deg = this->get_parameter("max_acceptable_angle_deg").as_double();
        int max_iterations = this->get_parameter("max_iterations").as_int();
        // -------------------------------------

        sensor_msgs::msg::PointCloud2 flat_msg;
        try {
            flat_msg = tf_buffer_->transform(*msg, target_frame_, tf2::durationFromSec(0.1));
        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN(this->get_logger(), "TF Error: %s", ex.what());
            return; 
        }

        tf_publisher_->publish(flat_msg);

        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(flat_msg, *raw_cloud);

        if (raw_cloud->points.empty()) return;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(raw_cloud);
        voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
        voxel_filter.filter(*cloud);

        if (cloud->points.size() < 10) return;

        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(false);
        seg.setModelType(pcl::SACMODEL_PARALLEL_PLANE); 
        seg.setMethodType(pcl::SAC_RANSAC);    
        
        // APPLY LIVE PARAMETERS
        seg.setMaxIterations(max_iterations); 
        seg.setDistanceThreshold(distance_threshold); 
        seg.setEpsAngle(eps_angle_deg * (M_PI / 180.0)); 

        Eigen::Vector3f parallel_axis(1.0f, 1.0f, 1.0f);
        Eigen::Vector3f perpendicular_axis(0.0f, 0.0f, 0.0f);
        if (height_axis_ == Axis::Z) {
            parallel_axis[2] = 0.0f;
            perpendicular_axis[2] = 1.0f;
        }
        else if (height_axis_ == Axis::Y){ 
            parallel_axis[1] = 0.0f;
            perpendicular_axis[1] = 1.0f;
        }
        else if (height_axis_ == Axis::X){ 
            parallel_axis[0] = 0.0f;
            perpendicular_axis[0] = 1.0f;
        }
        seg.setAxis(parallel_axis);

        // ==========================================
        // DUAL-RANSAC IMPLEMENTATION
        // ==========================================
        
        // --- PASS 1: Find the first largest plane ---
        pcl::ModelCoefficients::Ptr coefficients1(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers1(new pcl::PointIndices());
        seg.setInputCloud(cloud);
        seg.segment(*inliers1, *coefficients1);

        if (inliers1->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "Could not estimate a planar model. Skipping frame.");
            return;
        }

        float height1 = 0.0f;
        for (int idx : inliers1->indices) {
            height1 += get_axis_val(cloud->points[idx], height_axis_);
        }
        height1 /= inliers1->indices.size();

        // --- PASS 2: Find the second largest plane ---
        std::vector<bool> is_inlier1(cloud->points.size(), false);
        for (int idx : inliers1->indices) {
            is_inlier1[idx] = true;
        }

        pcl::IndicesPtr remaining_indices(new std::vector<int>);
        for (size_t i = 0; i < cloud->points.size(); ++i) {
            if (!is_inlier1[i]) {
                remaining_indices->push_back(i);
            }
        }

        pcl::ModelCoefficients::Ptr coefficients2(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers2(new pcl::PointIndices());
        
        if (remaining_indices->size() >= 10) {
            seg.setIndices(remaining_indices);
            seg.segment(*inliers2, *coefficients2);
        }

        // --- COMPARE HEIGHTS AND ASSIGN GROUND ---
        pcl::ModelCoefficients::Ptr coefficients = coefficients1;
        pcl::PointIndices::Ptr ground_inliers = inliers1;

        if (!inliers2->indices.empty()) {
            float height2 = 0.0f;
            for (int idx : inliers2->indices) {
                height2 += get_axis_val(cloud->points[idx], height_axis_);
            }
            height2 /= inliers2->indices.size();

            // The plane with the lower height is definitively the ground
            if (height2 < height1) {
                ground_inliers = inliers2;
                coefficients = coefficients2;
            } else {
                ground_inliers = inliers1;
                coefficients = coefficients1;
            }
        }
        
        // ==========================================
        // MANUAL ANGLE VALIDATION
        // ==========================================
        Eigen::Vector3f plane_normal(
            coefficients->values[0], 
            coefficients->values[1], 
            coefficients->values[2]
        );

        plane_normal.normalize(); 

        float dot_product = plane_normal.dot(perpendicular_axis);
        float angle_rad = std::acos(std::abs(dot_product));
        float angle_deg = angle_rad * (180.0f / M_PI);

        // USE LIVE PARAMETER
        if (angle_deg > max_acceptable_angle_deg) {
            return; 
        }

        std_msgs::msg::Float32MultiArray coeff_msg;
        coeff_msg.data = {
            coefficients->values[0], 
            coefficients->values[1], 
            coefficients->values[2], 
            coefficients->values[3]  
        };
        coeff_publisher_->publish(coeff_msg);

        // ==========================================
        // RVIZ2 PLANE VISUALIZATION MARKER
        // ==========================================
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

        plane_marker.scale.x = 10.0; 
        plane_marker.scale.y = 10.0; 
        plane_marker.scale.z = 0.01; 
        
        plane_marker.color.r = 0.0f;
        plane_marker.color.g = 1.0f;
        plane_marker.color.b = 0.0f;
        plane_marker.color.a = 0.4f; 

        plane_marker_publisher_->publish(plane_marker);

        // ==========================================
        // Extract and Publish the GROUND (Inliers)
        // ==========================================
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

        // ==========================================
        // Extract the SEEDBEDS / OBJECTS (Outliers)
        // ==========================================
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_objects(new pcl::PointCloud<pcl::PointXYZ>());
        extract.setNegative(true); 
        extract.filter(*cloud_objects);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZ>());
        *cloud_2d = *cloud_objects; 
        for (auto& pt : cloud_2d->points) { set_axis_val(pt, height_axis_, 0.0f); }

        pcl::PointCloud<pcl::PointXYZ>::Ptr pruned_cloud_2d(new pcl::PointCloud<pcl::PointXYZ>());
        if (!cloud_2d->points.empty()) {
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(cloud_2d);
            sor.setMeanK(10); 
            sor.setStddevMulThresh(1); 
            sor.filter(*pruned_cloud_2d);
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr final_target_object(new pcl::PointCloud<pcl::PointXYZ>());

        if (!pruned_cloud_2d->points.empty()) {
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
            tree->setInputCloud(pruned_cloud_2d); 

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(0.05); 
            ec.setMinClusterSize(5); 
            ec.setMaxClusterSize(250000);
            ec.setSearchMethod(tree);
            ec.setInputCloud(pruned_cloud_2d); 
            ec.extract(cluster_indices);

            if (!cluster_indices.empty()) {
                const double MIN_AREA_THRESHOLD = 0.01; 
                for (size_t i = 0; i < cluster_indices.size(); ++i) {
                    float min_u = std::numeric_limits<float>::max();
                    float max_u = -std::numeric_limits<float>::max();
                    float min_v = std::numeric_limits<float>::max();
                    float max_v = -std::numeric_limits<float>::max();

                    for (const auto& idx : cluster_indices[i].indices) {
                        const auto& pt = pruned_cloud_2d->points[idx];
                        float u, v;
                        if (height_axis_ == Axis::X) { u = pt.y; v = pt.z; }
                        else if (height_axis_ == Axis::Y) { u = pt.x; v = pt.z; }
                        else { u = pt.x; v = pt.y; }

                        if (u < min_u) min_u = u;
                        if (u > max_u) max_u = u;
                        if (v < min_v) min_v = v;
                        if (v > max_v) max_v = v;
                    }
                    double area = (max_u - min_u) * (max_v - min_v);
                    if (area >= MIN_AREA_THRESHOLD) {
                        for (const auto& idx : cluster_indices[i].indices) {
                            final_target_object->points.push_back(pruned_cloud_2d->points[idx]); 
                        }
                    }
                }
            }
        }

        // ==========================================
        // GLOBAL GAP FILLING ALGORITHM
        // ==========================================
        pcl::PointCloud<pcl::PointXYZ>::Ptr filled_gaps_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        if (!final_target_object->points.empty()) {
            Axis axis_forward, axis_lateral;
            if (height_axis_ == Axis::X) { axis_forward = Axis::Y; axis_lateral = Axis::Z; }
            else if (height_axis_ == Axis::Y) { axis_forward = Axis::X; axis_lateral = Axis::Z; }
            else { axis_forward = Axis::X; axis_lateral = Axis::Y; }

            const float FILL_RESOLUTION = 0.01f; 
            
            std::map<int, std::pair<float, float>> boundary_map;

            for (const auto& pt : final_target_object->points) {
                float fwd_val = get_axis_val(pt, axis_forward);
                float lat_val = get_axis_val(pt, axis_lateral);
                int fwd_bin = std::round(fwd_val / FILL_RESOLUTION);

                if (boundary_map.find(fwd_bin) == boundary_map.end()) {
                    boundary_map[fwd_bin] = {lat_val, lat_val};
                } else {
                    if (lat_val < boundary_map[fwd_bin].first)  boundary_map[fwd_bin].first = lat_val;
                    if (lat_val > boundary_map[fwd_bin].second) boundary_map[fwd_bin].second = lat_val;
                }
            }

            for (const auto& kv : boundary_map) {
                int fwd_bin = kv.first;
                float min_lat = kv.second.first;
                float max_lat = kv.second.second;

                if ((max_lat - min_lat) < 0.10f) {
                    continue; 
                }

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

        // ==========================================
        // FINAL PUBLISH (ONLY THE FILLED GAPS)
        // ==========================================
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
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GroundSegmentationNode>());
    rclcpp::shutdown();
    return 0;
}