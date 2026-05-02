#include <memory>
#include <vector>
#include <map>
#include <cmath>
#include <numeric>
#include <Eigen/Dense> 
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/float64.hpp" // Added Float64 header

// PCL Headers
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>

class PathExtractionNode : public rclcpp::Node
{
public:
    PathExtractionNode() : Node("path_extraction_node")
    {
        // --- System Parameters ---
        this->declare_parameter<double>("lookahead_min", 0.30);      // START BORDERLINE AT 0.3m
        this->declare_parameter<double>("lookahead_max", 2.5);      // END BORDERLINE AT 1.5m
        this->declare_parameter<double>("roi_half_width", 0.6);     // NEW: 0.45m half width (0.9m total)
        this->declare_parameter<double>("lookahead_dist", 1.0); // NEW: 0.70m target distance
        this->declare_parameter<double>("slice_thickness", 0.05);    // 5cm rectangles
        this->declare_parameter<int>("min_points_per_slice", 5); 
        this->declare_parameter<int>("moving_avg_window", 10);       // Buffer for static stabilization
        
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/processed_pcd", 10,
            std::bind(&PathExtractionNode::cloud_callback, this, std::placeholders::_1));

        marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/polynomial_path", 10);
        target_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/lookahead_target", 10);
        roi_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("/roi_boundaries", 10);
        
        // Changed publisher topic and message type
        angle_publisher_ = this->create_publisher<std_msgs::msg::Float64>("/lookahead_angle", 10);
        
        RCLCPP_INFO(this->get_logger(), "Stabilized Polynomial Path Extraction Node Started.");
    }

private:
    std::vector<double> horizontal_error_buffer_;
    std::vector<double> angle_error_buffer_;

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) 
    {
        double min_x = this->get_parameter("lookahead_min").as_double();
        double max_x = this->get_parameter("lookahead_max").as_double();
        double roi_y = this->get_parameter("roi_half_width").as_double();
        double target_dist = this->get_parameter("lookahead_dist").as_double();
        double slice_thickness = this->get_parameter("slice_thickness").as_double();
        int min_points = this->get_parameter("min_points_per_slice").as_int();
        size_t window_size = this->get_parameter("moving_avg_window").as_int();

        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *raw_cloud);

        if (raw_cloud->points.empty()) return;

        // ==========================================
        // 1. Draw the 2 Borderlines (ROI)
        // ==========================================
        publish_roi_marker(msg->header, min_x, max_x, roi_y);

        // ==========================================
        // 2. Crop Cloud (Ignore Under Robot & Adjacent Rows)
        // ==========================================
        pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(raw_cloud);
        pass_x.setFilterFieldName("x"); 
        pass_x.setFilterLimits(min_x, max_x); 
        pass_x.filter(*cropped_cloud);

        pcl::PassThrough<pcl::PointXYZ> pass_y;
        pass_y.setInputCloud(cropped_cloud);
        pass_y.setFilterFieldName("y"); 
        pass_y.setFilterLimits(-roi_y, roi_y); 
        pass_y.filter(*cropped_cloud);

        if (cropped_cloud->points.empty()) return;

        // ==========================================
        // 3. Noise Filtering
        // ==========================================
        pcl::PointCloud<pcl::PointXYZ>::Ptr clean_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cropped_cloud);
        sor.setMeanK(20);             
        sor.setStddevMulThresh(1.0);  
        sor.filter(*clean_cloud);

        if (clean_cloud->points.empty()) return;

        // ==========================================
        // 4. Divide Region into Rectangles & Centroids
        // ==========================================
        struct Slice { double sum_y = 0.0; int count = 0; };
        std::map<int, Slice> slices;

        for (const auto& pt : clean_cloud->points) {
            int slice_idx = std::floor(pt.x / slice_thickness);
            slices[slice_idx].sum_y += pt.y;
            slices[slice_idx].count += 1;
        }

        std::vector<double> x_centroids;
        std::vector<double> y_centroids;

        for (const auto& [idx, data] : slices) {
            if (data.count >= min_points) {
                double x_center = (idx * slice_thickness) + (slice_thickness / 2.0);
                double y_center = data.sum_y / data.count;
                
                x_centroids.push_back(x_center);
                y_centroids.push_back(y_center);
            }
        }

        if (x_centroids.size() < 3) return;

        // ==========================================
        // 5. Polynomial Fitting (Eigen)
        // ==========================================
        Eigen::MatrixXd A(x_centroids.size(), 3);
        Eigen::VectorXd B(y_centroids.size());

        for (size_t i = 0; i < x_centroids.size(); ++i) {
            A(i, 0) = 1.0; 
            A(i, 1) = x_centroids[i]; 
            A(i, 2) = std::pow(x_centroids[i], 2); 
            B(i) = y_centroids[i]; 
        }

        Eigen::VectorXd coeffs = A.householderQr().solve(B);
        double c = coeffs(0);
        double b = coeffs(1);
        double a = coeffs(2);

        // ==========================================
        // 6. Calculate Errors & Smooth (Stabilization)
        // ==========================================
        double target_y = (a * std::pow(target_dist, 2)) + (b * target_dist) + c;
        double angle_rad = std::atan2(target_y, target_dist);

        horizontal_error_buffer_.push_back(target_y);
        angle_error_buffer_.push_back(angle_rad);

        if (horizontal_error_buffer_.size() > window_size) {
            horizontal_error_buffer_.erase(horizontal_error_buffer_.begin());
            angle_error_buffer_.erase(angle_error_buffer_.begin());
        }

        double smooth_horizontal_error = std::accumulate(horizontal_error_buffer_.begin(), horizontal_error_buffer_.end(), 0.0) / horizontal_error_buffer_.size();
        double smooth_angle_error = std::accumulate(angle_error_buffer_.begin(), angle_error_buffer_.end(), 0.0) / angle_error_buffer_.size();

        // Publish Angle Error to Motor Topics (Runs at full speed)
        std_msgs::msg::Float64 angle_msg;

        angle_msg.data = smooth_angle_error;
        angle_publisher_->publish(angle_msg);

        // ==========================================
        // 7. Throttled Terminal Printouts
        // ==========================================
        double angle_deg = smooth_angle_error * (180.0 / M_PI);
        
        if (std::abs(smooth_horizontal_error) <= 0.03) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, 
                "[STRAIGHT] Perfectly in ROI. Error: %.3fm, Angle: %.2f deg", smooth_horizontal_error, angle_deg);
        } 
        else if (smooth_horizontal_error > 0.03) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, 
                "[STEER LEFT]  Path is %.3fm to the left. Angle: %.2f deg", smooth_horizontal_error, angle_deg);
        } 
        else {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 500, 
                "[STEER RIGHT] Path is %.3fm to the right. Angle: %.2f deg", std::abs(smooth_horizontal_error), angle_deg);
        }

        // ==========================================
        // 8. Visualizations for RViz
        // ==========================================
        publish_path_and_target(msg->header, a, b, c, min_x, max_x, target_dist, target_y, roi_y);
    }

    void publish_roi_marker(std_msgs::msg::Header header, double min_x, double max_x, double half_y) {
        visualization_msgs::msg::Marker roi_marker;
        roi_marker.header = header;
        if (roi_marker.header.frame_id.empty()) roi_marker.header.frame_id = "base_link";
        
        roi_marker.ns = "roi_bounds";
        roi_marker.id = 2;
        roi_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        roi_marker.action = visualization_msgs::msg::Marker::ADD;
        roi_marker.pose.orientation.w = 1.0;
        roi_marker.scale.x = 0.02; 
        roi_marker.color.r = 0.0f; roi_marker.color.g = 1.0f; roi_marker.color.b = 0.0f; roi_marker.color.a = 0.8f;

        geometry_msgs::msg::Point p;
        p.z = 0.0;
        
        // Left Bound
        p.x = min_x; p.y = half_y; roi_marker.points.push_back(p);
        p.x = max_x; p.y = half_y; roi_marker.points.push_back(p);
        // Right Bound
        p.x = min_x; p.y = -half_y; roi_marker.points.push_back(p);
        p.x = max_x; p.y = -half_y; roi_marker.points.push_back(p);

        roi_marker_publisher_->publish(roi_marker);
    }

    // Notice I added 'roi_y' to the parameters here so it knows how wide to draw the new line
    void publish_path_and_target(std_msgs::msg::Header header, double a, double b, double c, double min_x, double max_x, double target_x, double target_y, double roi_y)
    {
        if (header.frame_id.empty()) header.frame_id = "base_link";

        // Draw Polynomial Path
        visualization_msgs::msg::Marker path_marker;
        path_marker.header = header;
        path_marker.ns = "poly_path";
        path_marker.id = 0;
        path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        path_marker.action = visualization_msgs::msg::Marker::ADD;
        path_marker.pose.orientation.w = 1.0; // Required for lines
        path_marker.scale.x = 0.03; 
        path_marker.color.r = 0.0f; path_marker.color.g = 1.0f; path_marker.color.b = 1.0f; path_marker.color.a = 1.0f; 

        for (double x = min_x; x <= max_x; x += 0.05) {
            geometry_msgs::msg::Point p;
            p.x = x;
            p.y = (a * x * x) + (b * x) + c;
            p.z = 0.0;
            path_marker.points.push_back(p);
        }
        marker_publisher_->publish(path_marker);

        // --- NEW: Draw Lookahead Line and Midpoint ---
        visualization_msgs::msg::Marker target_marker;
        target_marker.header = header;
        target_marker.ns = "lookahead_target";
        target_marker.id = 1;
        target_marker.type = visualization_msgs::msg::Marker::LINE_LIST; // Allows multiple separate lines
        target_marker.action = visualization_msgs::msg::Marker::ADD;
        target_marker.pose.orientation.w = 1.0;
        target_marker.scale.x = 0.02; // Line thickness
        target_marker.color.r = 1.0f; target_marker.color.g = 0.0f; target_marker.color.b = 0.0f; target_marker.color.a = 1.0f; 

        // Line 1: Draw a horizontal line spanning the expected width of your robot
        geometry_msgs::msg::Point p1, p2;
        p1.x = target_x; p1.y = -roi_y; p1.z = 0.0;
        p2.x = target_x; p2.y =  roi_y; p2.z = 0.0;
        target_marker.points.push_back(p1);
        target_marker.points.push_back(p2);

        // Line 2: Draw a vertical tick marking the "midpoint" on the polynomial curve
        geometry_msgs::msg::Point p3, p4;
        p3.x = target_x - 0.10; p3.y = target_y; p3.z = 0.0;
        p4.x = target_x + 0.10; p4.y = target_y; p4.z = 0.0;
        target_marker.points.push_back(p3);
        target_marker.points.push_back(p4);

        target_publisher_->publish(target_marker);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr target_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr roi_marker_publisher_;
    
    // Updated publisher member
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr angle_publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathExtractionNode>());
    rclcpp::shutdown();
    return 0;
}