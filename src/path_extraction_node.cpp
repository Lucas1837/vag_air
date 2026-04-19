#include <memory>
#include <vector>
#include <cmath>
#include <numeric>
#include <limits>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <std_msgs/msg/float64.hpp>
#include "visualization_msgs/msg/marker.hpp"

// PCL Headers
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>

class TargetBandExtractionNode : public rclcpp::Node
{
public:
    TargetBandExtractionNode() : Node("target_band_extraction_node")
    {
        // --- System Parameters ---
        this->declare_parameter<double>("lookahead_distance", 1.50);        
        this->declare_parameter<double>("lookahead_window_thickness", 0.15); // Box depth
        this->declare_parameter<double>("search_width", 1.20);               // Visual width of the RViz box
        this->declare_parameter<int>("moving_avg_window", 10);              
        
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/processed_pcd", 10,
            std::bind(&TargetBandExtractionNode::cloud_callback, this, std::placeholders::_1));

        marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/steering_vector", 10);
        box_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/lookahead_box", 10);
        error_publisher_ = this->create_publisher<std_msgs::msg::Float64>("/lookahead_angle", 10);
        
        RCLCPP_INFO(this->get_logger(), "Dynamic Box Extraction Node Started (Curve-Safe).");
    }

private:
    std::vector<double> angle_error_buffer_;
    std::vector<double> horizontal_error_buffer_;

    void publish_straight_command(const std::string& reason) 
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
            "Gap lost (%s). Commanding robot to go STRAIGHT.", reason.c_str());
            
        std_msgs::msg::Float64 error_msg; 
        error_msg.data = 0.0;      
        error_publisher_->publish(error_msg);

        horizontal_error_buffer_.clear();
        angle_error_buffer_.clear();
    }

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) 
    {
        double target_dist = this->get_parameter("lookahead_distance").as_double();
        double thickness = this->get_parameter("lookahead_window_thickness").as_double();
        double search_width = this->get_parameter("search_width").as_double();
        size_t window_size = this->get_parameter("moving_avg_window").as_int();

        double min_x = target_dist - (thickness / 2.0);
        double max_x = target_dist + (thickness / 2.0);

        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *raw_cloud);

        if (raw_cloud->points.empty()) {
            publish_straight_command("Empty raw cloud");
            return;
        }

        // 1. Draw the Lookahead Search Box in RViz
        publish_search_box(msg->header, min_x, max_x, search_width);

        // 2. Crop Cloud EXACTLY to the X-axis target band (The Box)
        pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(raw_cloud);
        pass_x.setFilterFieldName("x"); 
        pass_x.setFilterLimits(min_x, max_x); 
        pass_x.filter(*cropped_cloud);

        // NOTICE: The Y-axis PassThrough filter has been completely removed! 
        // The robot can now track sharp curves outside the center.

        if (cropped_cloud->points.empty()) {
            publish_straight_command("No ground found inside the lookahead box");
            return;
        }

        // 3. Noise Filtering
        pcl::PointCloud<pcl::PointXYZ>::Ptr clean_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cropped_cloud);
        sor.setMeanK(10);             
        sor.setStddevMulThresh(1.0);  
        sor.filter(*clean_cloud);

        if (clean_cloud->points.empty()) {
            publish_straight_command("Cloud empty after noise filtering");
            return;
        }

        // ==========================================
        // 4. FIND THE TRUE MIDPOINT OF THE GROUND
        // ==========================================
        double min_y = std::numeric_limits<double>::max();
        double max_y = -std::numeric_limits<double>::max();

        // Scan all ground points in the box to find the physical boundaries
        for (const auto& pt : clean_cloud->points) {
            if (pt.y < min_y) min_y = pt.y;
            if (pt.y > max_y) max_y = pt.y;
        }

        // The center of the path is exactly between the furthest left and furthest right ground points
        double target_y = (min_y + max_y) / 2.0;

        // Calculate Angle (Reversed Direction: Left=Negative, Right=Positive)
        double angle_rad = std::atan2(target_y, target_dist);

        // ==========================================
        // 5. STABILIZE AND PUBLISH
        // ==========================================
        horizontal_error_buffer_.push_back(target_y);
        angle_error_buffer_.push_back(angle_rad);

        if (horizontal_error_buffer_.size() > window_size) {
            horizontal_error_buffer_.erase(horizontal_error_buffer_.begin());
            angle_error_buffer_.erase(angle_error_buffer_.begin());
        }

        double smooth_horizontal_error = std::accumulate(horizontal_error_buffer_.begin(), horizontal_error_buffer_.end(), 0.0) / horizontal_error_buffer_.size();
        double smooth_angle_error = std::accumulate(angle_error_buffer_.begin(), angle_error_buffer_.end(), 0.0) / angle_error_buffer_.size();

        std_msgs::msg::Float64 error_msg; 
        error_msg.data = smooth_angle_error;      
        error_publisher_->publish(error_msg);

        // Terminal Printouts
        double angle_deg = smooth_angle_error * (180.0 / M_PI);
        if (std::abs(smooth_horizontal_error) <= 0.03) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, 
                "[STRAIGHT] Centered in curve. Angle: %.2f deg", angle_deg);
        } else if (smooth_horizontal_error > 0.03) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, 
                "[STEER LEFT]  Path center is %.2fm left. Angle: %.2f deg", smooth_horizontal_error, angle_deg);
        } else {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, 
                "[STEER RIGHT] Path center is %.2fm right. Angle: %.2f deg", std::abs(smooth_horizontal_error), angle_deg);
        }

        // Draw the steering vector pointing to the dynamically found midpoint
        publish_steering_vector(msg->header, target_dist, target_y);
    }

    void publish_search_box(std_msgs::msg::Header header, double min_x, double max_x, double search_width) {
        visualization_msgs::msg::Marker box_marker;
        box_marker.header = header;
        if (box_marker.header.frame_id.empty()) box_marker.header.frame_id = "base_link";
        
        box_marker.ns = "search_box";
        box_marker.id = 1;
        box_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        box_marker.action = visualization_msgs::msg::Marker::ADD;
        box_marker.pose.orientation.w = 1.0;
        box_marker.scale.x = 0.02; 
        box_marker.color.r = 0.0f; box_marker.color.g = 1.0f; box_marker.color.b = 0.0f; box_marker.color.a = 0.8f; 

        geometry_msgs::msg::Point p1, p2, p3, p4;
        p1.z = p2.z = p3.z = p4.z = 0.0;

        // Front edge
        p1.x = max_x; p1.y = search_width;  p2.x = max_x; p2.y = -search_width;
        box_marker.points.push_back(p1); box_marker.points.push_back(p2);
        // Back edge
        p3.x = min_x; p3.y = search_width;  p4.x = min_x; p4.y = -search_width;
        box_marker.points.push_back(p3); box_marker.points.push_back(p4);
        // Left edge
        box_marker.points.push_back(p1); box_marker.points.push_back(p3);
        // Right edge
        box_marker.points.push_back(p2); box_marker.points.push_back(p4);

        box_marker_publisher_->publish(box_marker);
    }

    void publish_steering_vector(std_msgs::msg::Header header, double target_x, double target_y)
    {
        if (header.frame_id.empty()) header.frame_id = "base_link";

        visualization_msgs::msg::Marker vector_marker;
        vector_marker.header = header;
        vector_marker.ns = "steering_vector";
        vector_marker.id = 2;
        vector_marker.type = visualization_msgs::msg::Marker::ARROW;
        vector_marker.action = visualization_msgs::msg::Marker::ADD;
        
        geometry_msgs::msg::Point start, end;
        start.x = 0.0; start.y = 0.0; start.z = 0.0;
        end.x = target_x; end.y = target_y; end.z = 0.0;
        
        vector_marker.points.push_back(start);
        vector_marker.points.push_back(end);

        vector_marker.scale.x = 0.05; 
        vector_marker.scale.y = 0.10; 
        vector_marker.scale.z = 0.10; 
        vector_marker.color.r = 1.0f; vector_marker.color.g = 1.0f; vector_marker.color.b = 0.0f; vector_marker.color.a = 1.0f; 

        marker_publisher_->publish(vector_marker);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr box_marker_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr error_publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TargetBandExtractionNode>());
    rclcpp::shutdown();
    return 0;
}