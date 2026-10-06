// iterative_wall_extraction_node.cpp

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <functional>
#include <unordered_map>
#include <limits> 

#include <Eigen/Core>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/common/common.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/filters/statistical_outlier_removal.h" 
#include "pcl/features/normal_3d.h"
#include "pcl/sample_consensus/method_types.h"
#include "pcl/sample_consensus/model_types.h"
#include "pcl/segmentation/sac_segmentation.h"
#include "pcl/segmentation/extract_clusters.h" 
#include "pcl/search/kdtree.h"
#include "pcl_conversions/pcl_conversions.h"

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

class IterativeWallExtractionNode : public rclcpp::Node
{
public:
  IterativeWallExtractionNode()
  : Node("iterative_wall_extraction_node")
  {
    // Voxel Grid Parameter
    this->declare_parameter<double>("voxel_leaf_size", 0.03);

    // RANSAC & Normal Parameters
    this->declare_parameter<double>("distance_threshold", 0.05);
    this->declare_parameter<double>("angle_tolerance_deg", 20.0);
    this->declare_parameter<int>("max_iterations", 10000);
    this->declare_parameter<int>("min_wall_points", 50); 
    this->declare_parameter<int>("normal_k_search", 50);
    this->declare_parameter<double>("normal_distance_weight", 0.1);

    // Axis parameters (Defaulted to Y-axis for lateral walls)
    this->declare_parameter<double>("axis_x", 0.0);
    this->declare_parameter<double>("axis_y", 1.0);
    this->declare_parameter<double>("axis_z", 0.0);

    // SOR Parameters
    this->declare_parameter<int>("sor_mean_k", 50);
    this->declare_parameter<double>("sor_stddev_mul_thresh", 0.3);

    // Clustering Parameters
    this->declare_parameter<double>("cluster_tolerance", 0.05);
    this->declare_parameter<int>("min_cluster_size", 30);
    this->declare_parameter<int>("max_cluster_size", 25000);
    
    // Merge Parameter
    this->declare_parameter<double>("height_merge_threshold", 0.05); 

    const auto sensor_qos = rclcpp::SensorDataQoS();

    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/point_cloud", sensor_qos,
      std::bind(&IterativeWallExtractionNode::pointCloudCallback, this, std::placeholders::_1));

    result_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/result", sensor_qos); 
      
    remaining_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/remaining_pcd", sensor_qos); 
      
    target_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/processed_pcd", sensor_qos); 
      
    marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/cluster_markers", 10);

    RCLCPP_INFO(this->get_logger(), "Iterative Wall Extraction Node Started (Targeting Highest Z Cluster).");
  }

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (msg->width * msg->height == 0) {
      return;
    }

    PointCloudT::Ptr input_cloud(new PointCloudT());
    pcl::fromROSMsg(*msg, *input_cloud);
    if (input_cloud->empty()) {
      return;
    }

    // Retrieve parameters
    double voxel_leaf_size = this->get_parameter("voxel_leaf_size").as_double();
    double distance_threshold = this->get_parameter("distance_threshold").as_double();
    double angle_tolerance_deg = this->get_parameter("angle_tolerance_deg").as_double();
    int max_iterations = this->get_parameter("max_iterations").as_int();
    int min_wall_points = this->get_parameter("min_wall_points").as_int();
    int k_search = this->get_parameter("normal_k_search").as_int();
    double normal_weight = this->get_parameter("normal_distance_weight").as_double();
    
    double ax = this->get_parameter("axis_x").as_double();
    double ay = this->get_parameter("axis_y").as_double();
    double az = this->get_parameter("axis_z").as_double();
    
    int sor_mean_k = this->get_parameter("sor_mean_k").as_int();
    double sor_stddev_mul_thresh = this->get_parameter("sor_stddev_mul_thresh").as_double();

    double cluster_tolerance = this->get_parameter("cluster_tolerance").as_double();
    int min_cluster_size = this->get_parameter("min_cluster_size").as_int();
    int max_cluster_size = this->get_parameter("max_cluster_size").as_int();
    double height_merge_threshold = this->get_parameter("height_merge_threshold").as_double();

    const double eps_angle_rad = angle_tolerance_deg * M_PI / 180.0;

    // ---------------------------------------------------------
    // 0. Voxel Grid Downsampling
    // ---------------------------------------------------------
    PointCloudT::Ptr filtered_cloud(new PointCloudT());
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(input_cloud);
    vg.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    vg.filter(*filtered_cloud);

    if (filtered_cloud->empty()) {
      return;
    }

    // ---------------------------------------------------------
    // 1. Calculate Normals for the filtered point cloud
    // ---------------------------------------------------------
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::search::KdTree<PointT>::Ptr norm_tree(new pcl::search::KdTree<PointT>());
    ne.setSearchMethod(norm_tree);
    ne.setInputCloud(filtered_cloud); 
    ne.setKSearch(k_search);

    pcl::PointCloud<pcl::Normal>::Ptr input_normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*input_normals);

    // ---------------------------------------------------------
    // 2. Iterative RANSAC Setup
    // ---------------------------------------------------------
    PointCloudT::Ptr working_cloud(new PointCloudT(*filtered_cloud)); 
    pcl::PointCloud<pcl::Normal>::Ptr working_normals(new pcl::PointCloud<pcl::Normal>(*input_normals));
    PointCloudT::Ptr all_walls_cloud(new PointCloudT());

    while (working_cloud->size() > static_cast<size_t>(min_wall_points))
    {
      pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
      seg.setOptimizeCoefficients(false);
      seg.setModelType(pcl::SACMODEL_NORMAL_PARALLEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setAxis(Eigen::Vector3f(ax, ay, az)); 
      seg.setEpsAngle(eps_angle_rad);
      seg.setDistanceThreshold(distance_threshold);
      seg.setMaxIterations(max_iterations);
      seg.setNormalDistanceWeight(normal_weight);

      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

      seg.setInputCloud(working_cloud);
      seg.setInputNormals(working_normals);
      seg.segment(*inliers, *coefficients);

      if (inliers->indices.empty() || static_cast<int>(inliers->indices.size()) < min_wall_points) {
        break;
      }

      pcl::ExtractIndices<PointT> extract_points;
      extract_points.setInputCloud(working_cloud);
      extract_points.setIndices(inliers);
      
      PointCloudT::Ptr wall_chunk(new PointCloudT());
      extract_points.setNegative(false); 
      extract_points.filter(*wall_chunk);
      *all_walls_cloud += *wall_chunk;

      PointCloudT::Ptr remainder_points(new PointCloudT());
      extract_points.setNegative(true); 
      extract_points.filter(*remainder_points);
      working_cloud.swap(remainder_points);

      pcl::ExtractIndices<pcl::Normal> extract_normals;
      extract_normals.setInputCloud(working_normals);
      extract_normals.setIndices(inliers);
      extract_normals.setNegative(true); 
      
      pcl::PointCloud<pcl::Normal>::Ptr remainder_normals(new pcl::PointCloud<pcl::Normal>());
      extract_normals.filter(*remainder_normals);
      working_normals.swap(remainder_normals);
    }

    // ---------------------------------------------------------
    // 3. Publish the aggregated wall points
    // ---------------------------------------------------------
    if (!all_walls_cloud->empty()) {
      sensor_msgs::msg::PointCloud2 out_msg;
      pcl::toROSMsg(*all_walls_cloud, out_msg);
      out_msg.header = msg->header;
      result_pub_->publish(out_msg);
    }

    // ---------------------------------------------------------
    // 3.5 Statistical Outlier Removal (SOR) on remaining cloud
    // ---------------------------------------------------------
    if (!working_cloud->empty()) {
      pcl::StatisticalOutlierRemoval<PointT> sor;
      sor.setInputCloud(working_cloud);
      sor.setMeanK(sor_mean_k);
      sor.setStddevMulThresh(sor_stddev_mul_thresh);
      
      PointCloudT::Ptr sor_filtered_cloud(new PointCloudT());
      sor.filter(*sor_filtered_cloud);
      working_cloud.swap(sor_filtered_cloud);
    }

    // ---------------------------------------------------------
    // 4. Publish the remaining point cloud
    // ---------------------------------------------------------
    if (working_cloud->empty()) {
      RCLCPP_DEBUG(this->get_logger(), "Remaining cloud is empty. Skipping clustering.");
      return;
    }

    sensor_msgs::msg::PointCloud2 remaining_msg;
    pcl::toROSMsg(*working_cloud, remaining_msg);
    remaining_msg.header = msg->header;
    remaining_pub_->publish(remaining_msg);

    // ---------------------------------------------------------
    // 5. Euclidean Clustering on the remaining point cloud
    // ---------------------------------------------------------
    pcl::search::KdTree<PointT>::Ptr cluster_tree(new pcl::search::KdTree<PointT>);
    cluster_tree->setInputCloud(working_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    
    ec.setClusterTolerance(cluster_tolerance); 
    ec.setMinClusterSize(min_cluster_size);     
    ec.setMaxClusterSize(max_cluster_size); 
    ec.setSearchMethod(cluster_tree);
    ec.setInputCloud(working_cloud);
    
    ec.extract(cluster_indices);

    if (cluster_indices.empty()) {
        RCLCPP_INFO(this->get_logger(), "No clusters found.");
        return;
    }

    // ---------------------------------------------------------
    // 5.5 Height-Based Cluster Merging (Union-Find)
    // ---------------------------------------------------------
    std::vector<double> raw_avg_heights;
    raw_avg_heights.reserve(cluster_indices.size());
    for (const auto& idx_set : cluster_indices) {
        double sum_z = 0.0;
        for (const auto& idx : idx_set.indices) {
            sum_z += working_cloud->points[idx].z;
        }
        raw_avg_heights.push_back(sum_z / idx_set.indices.size());
    }

    std::vector<int> parent(cluster_indices.size());
    std::iota(parent.begin(), parent.end(), 0);

    std::function<int(int)> find_root = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]]; 
            x = parent[x];
        }
        return x;
    };

    auto unite = [&](int a, int b) {
        int root_a = find_root(a);
        int root_b = find_root(b);
        if (root_a != root_b) {
            parent[root_a] = root_b;
        }
    };

    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        for (size_t j = i + 1; j < cluster_indices.size(); ++j) {
            if (std::abs(raw_avg_heights[i] - raw_avg_heights[j]) < height_merge_threshold) {
                unite(i, j);
            }
        }
    }

    std::unordered_map<int, std::vector<int>> merged_clusters_map;
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        int root = find_root(i);
        merged_clusters_map[root].insert(
            merged_clusters_map[root].end(),
            cluster_indices[i].indices.begin(),
            cluster_indices[i].indices.end()
        );
    }

    // ---------------------------------------------------------
    // 6. Targeting (Find cluster with the Highest Z value)
    // ---------------------------------------------------------
    int best_cluster_id = -1;
    double max_cluster_z = -std::numeric_limits<double>::max();

    for (const auto& kv : merged_clusters_map) {
        int cluster_id = kv.first;
        const std::vector<int>& merged_indices = kv.second;

        // Calculate the average Z value for the cluster
        double sum_z = 0.0;
        for (int idx : merged_indices) {
            sum_z += working_cloud->points[idx].z;
        }
        double avg_z = sum_z / merged_indices.size();

        // Check if this cluster is the highest one found so far
        if (avg_z > max_cluster_z) {
            max_cluster_z = avg_z;
            best_cluster_id = cluster_id;
        }
    }

    // ---------------------------------------------------------
    // 7. Flatten the target cluster & Publish
    // ---------------------------------------------------------
    PointCloudT::Ptr target_cloud(new PointCloudT());
    if (best_cluster_id != -1) {
        const std::vector<int>& target_indices = merged_clusters_map[best_cluster_id];
        target_cloud->reserve(target_indices.size());
        
        for (int idx : target_indices) {
            PointT pt = working_cloud->points[idx];
            pt.z = 0.0; // Force Z to 0 to flatten
            target_cloud->push_back(pt);
        }
        
        target_cloud->width = target_cloud->size();
        target_cloud->height = 1;
        target_cloud->is_dense = true;

        sensor_msgs::msg::PointCloud2 target_msg;
        pcl::toROSMsg(*target_cloud, target_msg);
        target_msg.header = msg->header;
        target_pub_->publish(target_msg);
    } else {
        RCLCPP_WARN(this->get_logger(), "No valid target cluster found for flattening.");
    }

    // ---------------------------------------------------------
    // 8. Calculate Centroids and Publish Markers
    // ---------------------------------------------------------
    visualization_msgs::msg::MarkerArray marker_array;
    visualization_msgs::msg::Marker delete_all_marker;
    delete_all_marker.header = msg->header; 
    delete_all_marker.ns = "cluster_centers";
    delete_all_marker.id = 0;
    delete_all_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_all_marker);

    int marker_id = 1;
    for (const auto& kv : merged_clusters_map)
    {
        int cluster_id = kv.first;
        const std::vector<int>& merged_indices = kv.second;
        
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;
        size_t num_points = merged_indices.size();

        for (const auto& idx : merged_indices)
        {
            sum_x += working_cloud->points[idx].x;
            sum_y += working_cloud->points[idx].y;
            sum_z += working_cloud->points[idx].z;
        }

        double center_x = sum_x / num_points;
        double center_y = sum_y / num_points;
        double center_z = sum_z / num_points;

        visualization_msgs::msg::Marker marker;
        marker.header = msg->header;
        marker.ns = "cluster_centers";
        marker.id = marker_id++; 
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = center_x;
        marker.pose.position.y = center_y;
        marker.pose.position.z = center_z;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
        
        // Highlight the best cluster (Red) vs all others (Green)
        if (cluster_id == best_cluster_id) {
            marker.color.r = 1.0; 
            marker.color.g = 0.0;
            marker.color.b = 0.0;
        } else {
            marker.color.r = 0.0; 
            marker.color.g = 1.0;
            marker.color.b = 0.0;
        }
        marker.color.a = 0.8; 
        marker.lifetime = rclcpp::Duration::from_seconds(0.2);
        
        marker_array.markers.push_back(marker);
    }

    marker_publisher_->publish(marker_array);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr result_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr remaining_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr target_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IterativeWallExtractionNode>());
  rclcpp::shutdown();
  return 0;
}