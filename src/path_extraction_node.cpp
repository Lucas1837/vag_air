#include <memory>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm> // Added for std::max
#include <Eigen/Dense> 
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/point.hpp"

// PCL Headers
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h> 

class PathExtractionNode : public rclcpp::Node
{
public:
    PathExtractionNode() : Node("path_extraction_node")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/processed_pcd", 10,
            std::bind(&PathExtractionNode::cloud_callback, this, std::placeholders::_1));

        marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/seed_bed_centerline", 10);
        
        lookahead_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/lookahead_error_visualization", 10);
        
        error_publisher_ = this->create_publisher<geometry_msgs::msg::Point>("/error", 10);
        
        RCLCPP_INFO(this->get_logger(), "Path Extraction Node Started. Listening to /processed_pcd...");
    }

private:
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->points.empty()) return;

        // Step 1: Find the nearest and farthest Z (depth) limits of the seed bed
        pcl::PointXYZ min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);

        // Step 2: Define the size of your tiny rectangles (5 cm depth slices)
        const float SLICE_THICKNESS = 0.05f; 
        
        struct SliceData {
            float sum_y = 0.0f;
            int count = 0;
            float center_z = 0.0f;
        };
        
        std::map<int, SliceData> slices;

        // Step 3: Divide the region into rectangles and sum the Y coordinates
        for (const auto& pt : cloud->points) {
            int slice_index = std::floor((pt.z - min_pt.z) / SLICE_THICKNESS);
            
            slices[slice_index].sum_y += pt.y; 
            slices[slice_index].count += 1;
            slices[slice_index].center_z = min_pt.z + (slice_index * SLICE_THICKNESS) + (SLICE_THICKNESS / 2.0f);
        }

        // Step 4: Prepare the RViz2 Marker for center line
        visualization_msgs::msg::Marker line_marker;
        line_marker.header = msg->header; 
        line_marker.ns = "centerline";
        line_marker.id = 0;
        line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line_marker.action = visualization_msgs::msg::Marker::ADD;
        
        line_marker.scale.x = 0.02; 
        
        line_marker.color.r = 0.0f;
        line_marker.color.g = 0.0f;
        line_marker.color.b = 1.0f;
        line_marker.color.a = 1.0f; 

        std::vector<double> z_vals;
        std::vector<double> y_vals;

        // Step 5: Calculate the final Y centroid for each rectangle and connect the dots
        for (auto const& [index, data] : slices) {
            if (data.count < 3) continue; 

            float avg_y = data.sum_y / data.count;

            z_vals.push_back(data.center_z);
            y_vals.push_back(avg_y);

            geometry_msgs::msg::Point p;
            p.x = 0.0f;          
            p.y = avg_y;         
            p.z = data.center_z; 

            line_marker.points.push_back(p);
        }

        marker_publisher_->publish(line_marker);

        // ==========================================
        // NEW Step 6: Polynomial Fit and Dynamic Error Calculation
        // ==========================================
        if (z_vals.size() >= 3) { 
            Eigen::MatrixXd A(z_vals.size(), 3);
            Eigen::VectorXd B(y_vals.size());

            for (size_t i = 0; i < z_vals.size(); ++i) {
                A(i, 0) = 1.0; 
                A(i, 1) = z_vals[i]; 
                A(i, 2) = std::pow(z_vals[i], 2); 
                B(i) = y_vals[i]; 
            }

            Eigen::VectorXd coeffs = A.householderQr().solve(B);
            double c = coeffs(0);
            double b = coeffs(1);
            double a = coeffs(2);

            // Set the base lookahead distance 
            double lookahead_dist = 0.05; 
            double target_z = 0.0;
            double horizontal_error = 0.0;

            // 1. Initial Logic Override Check
            if (min_pt.z < lookahead_dist) {
                target_z = z_vals[0]; 
                horizontal_error = y_vals[0]; 
                RCLCPP_DEBUG(this->get_logger(), "Override Triggered: Using first index centroid.");
            } else {
                target_z = lookahead_dist;
                horizontal_error = (a * std::pow(target_z, 2)) + (b * target_z) + c;
            }

            // 2. The Forward-Only Safety Gate
            // If the math calculated a target_z that is negative or zero, force it forward!
            if (target_z <= 0.0) {
                target_z = 0.01; // Clamp to 1cm strictly forward
                
                // Because we overrode the distance, we MUST use the polynomial to 
                // calculate the true lateral error at this new 1cm forward mark
                horizontal_error = (a * std::pow(target_z, 2)) + (b * target_z) + c;
                RCLCPP_WARN(this->get_logger(), "Safety Gate: Negative Z detected! Clamped lookahead to > 0.");
            }

            // 3. Calculate the angle error directly using the Z and Y coordinates
            // atan2 correctly handles all quadrants and computes the direct heading angle to the target
            double angle_error = std::atan2(horizontal_error, target_z);

            geometry_msgs::msg::Point error_msg;
            error_msg.x = horizontal_error;
            error_msg.y = angle_error;
            error_msg.z = 0.0;

            error_publisher_->publish(error_msg);

            RCLCPP_INFO(this->get_logger(), "Tracking Error at %.3f m ahead -> Horizontal: %.3f m, Angle: %.3f rad", target_z, horizontal_error, angle_error);

            // ==========================================
            // NEW Step 7: Visualize the Lookahead Horizontal Error
            // ==========================================
            visualization_msgs::msg::Marker lookahead_marker;
            lookahead_marker.header = msg->header;
            lookahead_marker.ns = "lookahead";
            lookahead_marker.id = 1;
            lookahead_marker.type = visualization_msgs::msg::Marker::LINE_LIST; 
            lookahead_marker.action = visualization_msgs::msg::Marker::ADD;
            
            lookahead_marker.scale.x = 0.01;
            
            lookahead_marker.color.r = 1.0f;
            lookahead_marker.color.g = 0.0f;
            lookahead_marker.color.b = 0.0f;
            lookahead_marker.color.a = 1.0f;

            geometry_msgs::msg::Point p1; 
            p1.x = 0.0f;
            p1.y = 0.0f; 
            p1.z = target_z; 

            geometry_msgs::msg::Point p2;
            p2.x = 0.0f;
            p2.y = horizontal_error; 
            p2.z = target_z; 

            lookahead_marker.points.push_back(p1);
            lookahead_marker.points.push_back(p2);

            lookahead_marker_publisher_->publish(lookahead_marker);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr lookahead_marker_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr error_publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathExtractionNode>());
    rclcpp::shutdown();
    return 0;
}