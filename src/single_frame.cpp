#include <memory>
#include <limits>
#include <vector>
#include <cmath>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

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

class GroundSegmentationNode : public rclcpp::Node
{
public:
    GroundSegmentationNode() : Node("ground_segmentation_node")
    {   
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zed/zed_node/point_cloud/cloud_registered", 10,
            std::bind(&GroundSegmentationNode::cloud_callback, this, std::placeholders::_1));

        // subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        //    "/cloud_pcd", 10,
        //    std::bind(&GroundSegmentationNode::cloud_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/processed_pcd", 10);
        
        RCLCPP_INFO(this->get_logger(), "Ground Segmentation Node Started. Listening....");
    }

private:
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *raw_cloud);

        // ==========================================
        // VoxelGrid Downsampling
        // ==========================================
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(raw_cloud);
        voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
        voxel_filter.filter(*cloud);

        // ==========================================
        // RANSAC Ground Segmentation
        // ==========================================
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        
        //Configure segmentation 
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE); 
        seg.setMethodType(pcl::SAC_RANSAC);    
        seg.setMaxIterations(200); 
        seg.setDistanceThreshold(0.03); 
        

        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "Could not estimate a planar model. Skipping frame.");
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_objects(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        
        //Extract the seedbed
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true); 
        extract.filter(*cloud_objects);

        // ==========================================
        // 2. Flatten to 2D
        // ==========================================
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZ>());
        *cloud_2d = *cloud_objects; 

        for (auto& pt : cloud_2d->points) {
            pt.x = 0.0f;
        }

        // ==========================================
        // NEW: Apply Statistical Outlier Removal BEFORE clustering
        // ==========================================
        pcl::PointCloud<pcl::PointXYZ>::Ptr pruned_cloud_2d(new pcl::PointCloud<pcl::PointXYZ>());
        if (!cloud_2d->points.empty()) {
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(cloud_2d);
            sor.setMeanK(10); // Analyze the 10 closest neighbors
            sor.setStddevMulThresh(1); // Prune points 1 standard deviation away from the local mean
            sor.filter(*pruned_cloud_2d);
        }

        // ==========================================
        // 3. Cluster and Isolate the CENTER-MOST Area
        // ==========================================
        pcl::PointCloud<pcl::PointXYZ>::Ptr final_target_object(new pcl::PointCloud<pcl::PointXYZ>());

        if (!pruned_cloud_2d->points.empty()) {
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
            tree->setInputCloud(pruned_cloud_2d); // Swapped to pruned cloud

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            
            ec.setClusterTolerance(0.05); 
            ec.setMinClusterSize(5); 
            ec.setMaxClusterSize(250000);
            ec.setSearchMethod(tree);
            ec.setInputCloud(pruned_cloud_2d); 
            ec.extract(cluster_indices);

            if (!cluster_indices.empty()) {
                double min_distance_to_center = std::numeric_limits<double>::max();
                int best_cluster_index = -1;
                
                // Smallest allowed area to be considered a valid seed bed (e.g., 0.001 sq meters)
                const double MIN_AREA_THRESHOLD = 0.001; 

                for (size_t i = 0; i < cluster_indices.size(); ++i) {
                    float min_y = std::numeric_limits<float>::max();
                    float max_y = -std::numeric_limits<float>::max();
                    float min_z = std::numeric_limits<float>::max();
                    float max_z = -std::numeric_limits<float>::max();

                    double sum_y = 0.0; 

                    for (const auto& idx : cluster_indices[i].indices) {
                        float y = pruned_cloud_2d->points[idx].y; 
                        float z = pruned_cloud_2d->points[idx].z; 
                        
                        if (y < min_y) min_y = y;
                        if (y > max_y) max_y = y;
                        if (z < min_z) min_z = z;
                        if (z > max_z) max_z = z;

                        sum_y += y; // Add up all the Y coordinates
                    }

                    double area = (max_y - min_y) * (max_z - min_z);

                    // Only consider clusters that are large enough to be the actual seed bed
                    if (area >= MIN_AREA_THRESHOLD) {
                        // Calculate the average Y position (the lateral center of the cluster)
                        double centroid_y = sum_y / cluster_indices[i].indices.size();
                        
                        // Absolute distance from the cluster's center to the camera's center axis (Y = 0)
                        double lateral_offset = std::abs(centroid_y);

                        // Track the cluster that is closest to the center line
                        if (lateral_offset < min_distance_to_center) {
                            min_distance_to_center = lateral_offset;
                            best_cluster_index = i;
                        }
                    }
                }

                // If we successfully found a cluster that met the minimum size requirement
                if (best_cluster_index != -1) {
                    for (const auto& idx : cluster_indices[best_cluster_index].indices) {
                        final_target_object->points.push_back(pruned_cloud_2d->points[idx]); // Swapped to pruned cloud
                    }

                    final_target_object->width = final_target_object->points.size();
                    final_target_object->height = 1;
                    final_target_object->is_dense = true;

                    RCLCPP_INFO(this->get_logger(), "Isolated seed bed. Lateral offset from center: %.3f meters", min_distance_to_center);
                } else {
                    RCLCPP_WARN(this->get_logger(), "No clusters met the minimum size threshold.");
                }
            }
        }

        // ==========================================
        // 4. Publish to RViz2
        // ==========================================
        sensor_msgs::msg::PointCloud2 output_msg;
        // Use the final isolated and pre-pruned target
        pcl::toROSMsg(*final_target_object, output_msg);
        
        output_msg.header = msg->header; 

        publisher_->publish(output_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GroundSegmentationNode>());
    rclcpp::shutdown();
    return 0;
}