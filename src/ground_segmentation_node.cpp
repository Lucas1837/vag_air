#include <memory>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include <numeric>
#include <unordered_set> 

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/point.hpp"             

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h> 
#include <pcl_conversions/pcl_conversions.h>

class SeedbedSegmentationNode : public rclcpp::Node
{
public:
    enum class Axis { X, Y, Z };
    Axis height_axis_ = Axis::Z;  
    Axis lateral_axis_ = Axis::Y; 

    SeedbedSegmentationNode() : Node("seedbed_segmentation_node")
    {
        normal_k_search_ = this->declare_parameter<int>("normal_k_search", 30);
        y_normal_tolerance_ = this->declare_parameter<double>("y_normal_tolerance", 0.7); 
        
        min_lateral_y_ = this->declare_parameter<double>("min_lateral_y", -0.60);
        max_lateral_y_ = this->declare_parameter<double>("max_lateral_y", 0.60);

        max_wall_height_ = this->declare_parameter<double>("max_wall_height", 0.35);
        min_wall_height_ = this->declare_parameter<double>("min_wall_height", 0.05);
        veto_grid_res_ = this->declare_parameter<double>("veto_grid_res", 0.10); 

        sor_mean_k_ = this->declare_parameter<int>("sor_mean_k", 50);
        sor_stddev_mul_thresh_ = this->declare_parameter<double>("sor_stddev_mul_thresh", 1.0);
        y_z_score_threshold_ = this->declare_parameter<double>("y_z_score_threshold", 1.6);
        fill_resolution_ = this->declare_parameter<double>("fill_resolution", 0.08);

        param_cb_ = this->add_on_set_parameters_callback(
            std::bind(&SeedbedSegmentationNode::onParamChange, this, std::placeholders::_1));

        rclcpp::QoS best_effort_qos = rclcpp::QoS(10).best_effort();

        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/point_cloud", best_effort_qos, 
            std::bind(&SeedbedSegmentationNode::cloudCallback, this, std::placeholders::_1));

        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/processed_pcd", best_effort_qos);
        debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/debug_walls_3d", best_effort_qos);
        
        RCLCPP_INFO(this->get_logger(), "Ground Segmentation Node Started.");
    }

private:
    int normal_k_search_;
    double y_normal_tolerance_;
    double min_lateral_y_; 
    double max_lateral_y_; 
    double max_wall_height_;
    double min_wall_height_; 
    double veto_grid_res_;
    int sor_mean_k_;
    double sor_stddev_mul_thresh_;
    double y_z_score_threshold_; 
    double fill_resolution_; 

    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_pub_;

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

    rcl_interfaces::msg::SetParametersResult onParamChange(const std::vector<rclcpp::Parameter> &parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto &param : parameters) {
            if (param.get_name() == "normal_k_search") normal_k_search_ = param.as_int();
            else if (param.get_name() == "y_normal_tolerance") y_normal_tolerance_ = param.as_double();
            else if (param.get_name() == "min_lateral_y") min_lateral_y_ = param.as_double();
            else if (param.get_name() == "max_lateral_y") max_lateral_y_ = param.as_double();
            else if (param.get_name() == "max_wall_height") max_wall_height_ = param.as_double();
            else if (param.get_name() == "min_wall_height") min_wall_height_ = param.as_double();
            else if (param.get_name() == "veto_grid_res") veto_grid_res_ = param.as_double();
            else if (param.get_name() == "sor_mean_k") sor_mean_k_ = param.as_int();
            else if (param.get_name() == "sor_stddev_mul_thresh") sor_stddev_mul_thresh_ = param.as_double();
            else if (param.get_name() == "y_z_score_threshold") y_z_score_threshold_ = param.as_double();
            else if (param.get_name() == "fill_resolution") fill_resolution_ = param.as_double();
        }
        return result;
    }

    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (msg->width * msg->height == 0) return;

        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *raw_cloud);
        if (raw_cloud->empty()) return;

        // Seed_beed height (Z-Axis) PassThrough Filter (Lower Limit Only) 
        pcl::PointCloud<pcl::PointXYZ>::Ptr z_filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PassThrough<pcl::PointXYZ> pass_z;
        pass_z.setInputCloud(raw_cloud);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(min_wall_height_, std::numeric_limits<double>::max());
        pass_z.filter(*z_filtered_cloud);

        if (z_filtered_cloud->empty()) return;

        // Lateral (Y-Axis) Boundary PassThrough Filter 
        pcl::PointCloud<pcl::PointXYZ>::Ptr bounded_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PassThrough<pcl::PointXYZ> pass_y;
        pass_y.setInputCloud(z_filtered_cloud); 
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(min_lateral_y_, max_lateral_y_);
        pass_y.filter(*bounded_cloud);

        if (bounded_cloud->empty()) return;

        // Create the Veto grid to eliminate tall and vertical noise 
        std::unordered_set<int64_t> veto_grid;
        pcl::PointCloud<pcl::PointXYZ>::Ptr short_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        for (const auto& pt : bounded_cloud->points) {
            if (pt.z > max_wall_height_) {
                int32_t bx = std::round(pt.x / veto_grid_res_);
                int32_t by = std::round(pt.y / veto_grid_res_);
                int64_t key = (static_cast<int64_t>(bx) << 32) | (static_cast<uint32_t>(by));
                veto_grid.insert(key);
            } else {
                short_cloud->push_back(pt);
            }
        }

        if (short_cloud->empty()) return;

        // Surface Normal Estimation 
        pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setSearchMethod(tree);
        ne.setInputCloud(short_cloud); 
        ne.setKSearch(normal_k_search_);
        ne.compute(*normals);

        // Extract seedbed Walls & flattening 
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_3d(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZ>());
        
        for (size_t i = 0; i < short_cloud->points.size(); ++i) {
            if (std::abs(normals->points[i].normal_y) >= y_normal_tolerance_) {
                pcl::PointXYZ pt = short_cloud->points[i];
                
                int32_t bx = std::round(pt.x / veto_grid_res_);
                int32_t by = std::round(pt.y / veto_grid_res_);
                int64_t key = (static_cast<int64_t>(bx) << 32) | (static_cast<uint32_t>(by));

                if (veto_grid.find(key) == veto_grid.end()) {
                    cloud_3d->push_back(pt);
                    pt.z = 0.0f; 
                    cloud_2d->push_back(pt);
                }
            }
        }
        
        if (!cloud_3d->empty()) {
            sensor_msgs::msg::PointCloud2 debug_msg;
            pcl::toROSMsg(*cloud_3d, debug_msg);
            debug_msg.header = msg->header;
            debug_pub_->publish(debug_msg);
        }

        if (cloud_2d->empty()) return;

        //  SOR Filtering (noise cancle) 
        pcl::PointCloud<pcl::PointXYZ>::Ptr pruned_cloud_2d(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud_2d);
        sor.setMeanK(sor_mean_k_);
        sor.setStddevMulThresh(sor_stddev_mul_thresh_);
        sor.filter(*pruned_cloud_2d);

        if (pruned_cloud_2d->empty()) return;

        //  Seed_bed wall Lateral Center (mean Y) Calculation
        std::map<int, std::pair<float, float>> mean_map;
        const float slice_res = 0.05f; 

        for (const auto& pt : pruned_cloud_2d->points) {
            int fwd_bin = std::round(pt.x / slice_res);
            if (mean_map.find(fwd_bin) == mean_map.end()) {
                mean_map[fwd_bin] = {pt.y, pt.y};
            } else {
                if (pt.y < mean_map[fwd_bin].first)  mean_map[fwd_bin].first = pt.y;
                if (pt.y > mean_map[fwd_bin].second) mean_map[fwd_bin].second = pt.y;
            }
        }

        double sum_mid_x = 0.0;
        double sum_mid_y = 0.0;
        int valid_slices = 0;

        for (const auto& kv : mean_map) {
            float min_y = kv.second.first;
            float max_y = kv.second.second;
            
            if ((max_y - min_y) > 0.20f) { // the 20cm represent the minimum distance for a left to right point cloud to be classified as a seedbed_wall 
            
                sum_mid_x += kv.first * slice_res;
                sum_mid_y += (min_y + max_y) / 2.0;
                valid_slices++;
            }
        }

        if (valid_slices > 0) { 
            double mean_x = sum_mid_x / valid_slices;
            double mean_y = sum_mid_y / valid_slices;

            // Lateral (Y-Axis) Statistical Outlier Removal
            double sq_sum_y = 0.0;
            for (const auto& pt : pruned_cloud_2d->points) {
                sq_sum_y += (pt.y - mean_y) * (pt.y - mean_y);
            }
            double stddev_y = std::sqrt(sq_sum_y / pruned_cloud_2d->points.size());

            pcl::PointCloud<pcl::PointXYZ>::Ptr y_filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto& pt : pruned_cloud_2d->points) {
                if (std::abs(pt.y - mean_y) <= y_z_score_threshold_ * stddev_y) {
                    y_filtered_cloud->push_back(pt);
                }
            }

            if (y_filtered_cloud->empty()) return;

            // Gap Filling 
            pcl::PointCloud<pcl::PointXYZ>::Ptr filled_gaps_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            Axis axis_forward, axis_lateral;
            if (height_axis_ == Axis::X) { axis_forward = Axis::Y; axis_lateral = Axis::Z; }
            else if (height_axis_ == Axis::Y) { axis_forward = Axis::X; axis_lateral = Axis::Z; }
            else { axis_forward = Axis::X; axis_lateral = Axis::Y; }

            const float fill_res = static_cast<float>(fill_resolution_); 
            std::map<int, std::pair<float, float>> boundary_map;

            for (const auto& pt : y_filtered_cloud->points) {
                float fwd_val = get_axis_val(pt, axis_forward);
                float lat_val = get_axis_val(pt, axis_lateral);
                int fwd_bin = std::round(fwd_val / fill_res);

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

                float current_lat = min_lat + fill_res;
                while (current_lat < max_lat) {
                    pcl::PointXYZ fill_pt;
                    set_axis_val(fill_pt, axis_forward, fwd_bin * fill_res);
                    set_axis_val(fill_pt, axis_lateral, current_lat);
                    set_axis_val(fill_pt, height_axis_, 0.0f); 
                    filled_gaps_cloud->points.push_back(fill_pt);
                    current_lat += fill_res;
                }
            }
            filled_gaps_cloud->width = filled_gaps_cloud->points.size();
            filled_gaps_cloud->height = 1;
            filled_gaps_cloud->is_dense = true;

            // Publish 
            if (!filled_gaps_cloud->points.empty()) {
                sensor_msgs::msg::PointCloud2 output_msg;
                pcl::toROSMsg(*filled_gaps_cloud, output_msg);
                output_msg.header = msg->header;
                pub_->publish(output_msg);
            }
        }
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SeedbedSegmentationNode>());
    rclcpp::shutdown();
    return 0;
}