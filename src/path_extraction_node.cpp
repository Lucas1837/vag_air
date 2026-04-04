#include <memory>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense> 
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/float64.hpp"

// PCL Headers
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h> 

class PathExtractionNode : public rclcpp::Node
{
public:
    // ==========================================
    // MODULAR CONFIGURATION
    // ==========================================
    enum class Axis { X, Y, Z };

    // Standard ROS 2 base_link frame conventions:
    Axis forward_axis_ = Axis::X; // The axis pointing forward into the distance
    Axis lateral_axis_ = Axis::Y; // The horizontal left/right axis
    Axis height_axis_  = Axis::Z; // The vertical up/down axis (flattened to 0)

    PathExtractionNode() : Node("path_extraction_node")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/processed_pcd", 10,
            std::bind(&PathExtractionNode::cloud_callback, this, std::placeholders::_1));

        marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/seed_bed_centerline", 10);
        lookahead_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/lookahead_error_visualization", 10);
        roi_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/roi_boundaries", 10);
        
        error_publisher_ = this->create_publisher<std_msgs::msg::Float64>("/error", 10);
        
        RCLCPP_INFO(this->get_logger(), "Path Extraction Node Started. Listening to /processed_pcd...");
    }

private:
    // --- Helper Functions to make axes modular ---
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

    inline void set_axis_val(geometry_msgs::msg::Point& pt, Axis axis, float value) const {
        if (axis == Axis::X) pt.x = value;
        else if (axis == Axis::Y) pt.y = value;
        else if (axis == Axis::Z) pt.z = value;
    }
    // ---------------------------------------------

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) 
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->points.empty()) return;

        // Step 1: Find the nearest and farthest limits of the seed bed
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);
        
        float min_forward = get_axis_val(min_pt, forward_axis_);
        float max_forward = get_axis_val(max_pt, forward_axis_);

        // ==========================================
        // Visualize Region of Interest (ROI) Boundaries
        // ==========================================
        const float ROI_HALF_WIDTH = 0.6f;

        visualization_msgs::msg::Marker roi_marker;
        roi_marker.header = msg->header;
        if (roi_marker.header.frame_id.empty()) roi_marker.header.frame_id = "base_link";
        
        roi_marker.ns = "roi_bounds";
        roi_marker.id = 2;
        roi_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        roi_marker.action = visualization_msgs::msg::Marker::ADD;
        roi_marker.pose.orientation.w = 1.0;
        roi_marker.scale.x = 0.02; // 2cm thick lines
        roi_marker.color.r = 0.0f;
        roi_marker.color.g = 1.0f; // Bright green
        roi_marker.color.b = 0.0f;
        roi_marker.color.a = 0.8f;

        // Left Boundary Line
        geometry_msgs::msg::Point left_start, left_end;
        set_axis_val(left_start, lateral_axis_, ROI_HALF_WIDTH);
        set_axis_val(left_start, forward_axis_, min_forward);
        set_axis_val(left_start, height_axis_, 0.0f);
        
        set_axis_val(left_end, lateral_axis_, ROI_HALF_WIDTH);
        set_axis_val(left_end, forward_axis_, max_forward);
        set_axis_val(left_end, height_axis_, 0.0f);
        
        // Right Boundary Line
        geometry_msgs::msg::Point right_start, right_end;
        set_axis_val(right_start, lateral_axis_, -ROI_HALF_WIDTH);
        set_axis_val(right_start, forward_axis_, min_forward);
        set_axis_val(right_start, height_axis_, 0.0f);
        
        set_axis_val(right_end, lateral_axis_, -ROI_HALF_WIDTH);
        set_axis_val(right_end, forward_axis_, max_forward);
        set_axis_val(right_end, height_axis_, 0.0f);

        roi_marker.points.push_back(left_start);
        roi_marker.points.push_back(left_end);
        roi_marker.points.push_back(right_start);
        roi_marker.points.push_back(right_end);

        roi_marker_publisher_->publish(roi_marker);

        // ==========================================
        // Step 2 & 3: Divide the region and check ROI inclusion
        // ==========================================
        const float SLICE_THICKNESS = 0.05f; 
        
        struct SliceData {
            float sum_lateral = 0.0f;
            int count = 0;
            float center_forward = 0.0f;
        };
        
        std::map<int, SliceData> slices;

        bool all_points_within_roi = true; 

        for (const auto& pt : cloud->points) {
            float forward_val = get_axis_val(pt, forward_axis_);
            float lateral_val = get_axis_val(pt, lateral_axis_);

            if (std::abs(lateral_val) > ROI_HALF_WIDTH ) {
                all_points_within_roi = false;
            }

            int slice_index = std::floor((forward_val - min_forward) / SLICE_THICKNESS);
            
            slices[slice_index].sum_lateral += lateral_val; 
            slices[slice_index].count += 1;
            slices[slice_index].center_forward = min_forward + (slice_index * SLICE_THICKNESS) + (SLICE_THICKNESS / 2.0f);
        }

        // ==========================================
        // The "Perfectly Straddled" Bypass Check
        // ==========================================
        if (all_points_within_roi) {
            std_msgs::msg::Float64 error_msg;
            error_msg.data = 0.0; 
            
            error_publisher_->publish(error_msg);
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                "Seedbed is perfectly within ROI. Driving straight (Angle Error: 0.0)");
            
            return; 
        }

        // Step 4: Prepare the RViz2 Marker for center line
        visualization_msgs::msg::Marker line_marker;
        line_marker.header = msg->header; 
        
        if (line_marker.header.frame_id.empty()) {
            line_marker.header.frame_id = "base_link"; 
        }

        line_marker.ns = "centerline";
        line_marker.id = 0;
        line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line_marker.action = visualization_msgs::msg::Marker::ADD;
        line_marker.pose.orientation.w = 1.0; 
        line_marker.scale.x = 0.03; 
        line_marker.color.r = 0.0f;
        line_marker.color.g = 0.0f;
        line_marker.color.b = 1.0f;
        line_marker.color.a = 1.0f; 

        std::vector<double> forward_vals;
        std::vector<double> lateral_vals;

        // Step 5: Calculate the final lateral centroid for each rectangle
        for (auto const& [index, data] : slices) {
            if (data.count < 1) continue; 

            float avg_lateral = data.sum_lateral / data.count;

            forward_vals.push_back(data.center_forward);
            lateral_vals.push_back(avg_lateral);

            geometry_msgs::msg::Point p;
            set_axis_val(p, height_axis_, 0.0f);         
            set_axis_val(p, lateral_axis_, avg_lateral); 
            set_axis_val(p, forward_axis_, data.center_forward); 

            line_marker.points.push_back(p);
        }

        if (!line_marker.points.empty()) {
            marker_publisher_->publish(line_marker);
        }

        // ==========================================
        // Step 6: Polynomial Fit and Fixed Error Calculation
        // ==========================================
        if (forward_vals.size() >= 3) { 
            Eigen::MatrixXd A(forward_vals.size(), 3);
            Eigen::VectorXd B(lateral_vals.size());

            for (size_t i = 0; i < forward_vals.size(); ++i) {
                A(i, 0) = 1.0; 
                A(i, 1) = forward_vals[i]; 
                A(i, 2) = std::pow(forward_vals[i], 2); 
                B(i) = lateral_vals[i]; 
            }

            Eigen::VectorXd coeffs = A.householderQr().solve(B);
            double c = coeffs(0);
            double b = coeffs(1);
            double a = coeffs(2);

            // --- FIXED LOOKAHEAD DISTANCE ---
            double target_forward = 2.0; 
            
            double horizontal_error = (a * std::pow(target_forward, 2)) + (b * target_forward) + c;
            double angle_error = std::atan2(horizontal_error, target_forward);

            std_msgs::msg::Float64 error_msg;
            error_msg.data = angle_error; 
            
            error_publisher_->publish(error_msg);

            // ==========================================
            // Step 7: Visualize the Lookahead as a Fixed Horizontal Line
            // ==========================================
            visualization_msgs::msg::Marker lookahead_marker;
            lookahead_marker.header = line_marker.header; 
            lookahead_marker.ns = "lookahead_horizon";
            lookahead_marker.id = 1;
            lookahead_marker.type = visualization_msgs::msg::Marker::LINE_LIST; 
            lookahead_marker.action = visualization_msgs::msg::Marker::ADD;
            lookahead_marker.pose.orientation.w = 1.0; 
            
            // Set thickness to match ROI lines (0.02)
            lookahead_marker.scale.x = 0.02; 
            
            // Bright Red to stand out
            lookahead_marker.color.r = 1.0f;
            lookahead_marker.color.g = 0.0f;
            lookahead_marker.color.b = 0.0f;
            lookahead_marker.color.a = 0.8f;

            // Start of the line: Left boundary at 0.7m forward
            geometry_msgs::msg::Point p1; 
            set_axis_val(p1, height_axis_, 0.0f);
            set_axis_val(p1, lateral_axis_, ROI_HALF_WIDTH); 
            set_axis_val(p1, forward_axis_, target_forward);

            // End of the line: Right boundary at 0.7m forward
            geometry_msgs::msg::Point p2;
            set_axis_val(p2, height_axis_, 0.0f);
            set_axis_val(p2, lateral_axis_, -ROI_HALF_WIDTH); 
            set_axis_val(p2, forward_axis_, target_forward); 

            lookahead_marker.points.push_back(p1);
            lookahead_marker.points.push_back(p2);

            lookahead_marker_publisher_->publish(lookahead_marker);
            
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                "Not enough valid slices to compute polynomial curve! Found: %zu, Required: 3", forward_vals.size());
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr lookahead_marker_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr roi_marker_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr error_publisher_; 
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathExtractionNode>());
    rclcpp::shutdown();
    return 0;
}