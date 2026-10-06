#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/float32.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"

namespace path_extraction
{

struct Point2D
{
  double x;
  double y;
};

//helper function to solve 3*3 linear eq
bool solve3x3(const double A[3][3], const double b[3], double x[3])
{
  auto det3 = [](const double m[3][3]) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  };

  const double det = det3(A);
  if (std::fabs(det) < 1e-12) {
    return false;
  }

  for (int col = 0; col < 3; ++col) {
    double Ai[3][3];
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        Ai[r][c] = (c == col) ? b[r] : A[r][c];
      }
    }
    x[col] = det3(Ai) / det;
  }
  return true;
}

class PathExtractionNode : public rclcpp::Node
{
public:
  PathExtractionNode()
  : Node("path_extraction_node")
  {
    slice_depth_ = this->declare_parameter<double>("slice_depth", 0.05);
    forward_start_ = this->declare_parameter<double>("forward_start", 0.0);
    forward_limit_ = this->declare_parameter<double>("forward_limit", 1.5);
    lookahead_x_ = this->declare_parameter<double>("lookahead_x", 1.0);
    max_slice_width_ = this->declare_parameter<double>("max_slice_width", 1.2);
    min_valid_centroids_ = this->declare_parameter<int>("min_valid_centroids", 2);
    marker_half_span_ = this->declare_parameter<double>("marker_half_span", 1.0);

    rclcpp::QoS best_effort_qos(rclcpp::KeepLast(10));
    best_effort_qos.best_effort();

    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/processed_pcd", best_effort_qos,
      std::bind(&PathExtractionNode::cloudCallback, this, std::placeholders::_1));

    angle_pub_ = this->create_publisher<std_msgs::msg::Float32>(
      "/lookahead_angle", best_effort_qos);

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/path_extraction/markers", 10);

    RCLCPP_INFO(this->get_logger(), "path_extraction_node started.");
  }

  //Cloud callback
private:
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    const std::vector<Point2D> points = readForwardPoints(msg);

    const std::vector<Point2D> centroids = extractSliceCentroids(points);

    if (static_cast<int>(centroids.size()) <= min_valid_centroids_) {
      RCLCPP_WARN(
        this->get_logger(),
        "Only %zu valid centroid(s) found (need > %d). Skipping frame.",
        centroids.size(), min_valid_centroids_);
      publishMarkers(msg->header, false, 0.0, 0.0, 0.0, 0.0, centroids);
      return;
    }

    double a = 0.0, b = 0.0, c = 0.0;
    if (!fitQuadratic(centroids, a, b, c)) {
      RCLCPP_WARN(this->get_logger(), "Polynomial fit failed (singular system). Skipping frame.");
      publishMarkers(msg->header, false, 0.0, 0.0, 0.0, 0.0, centroids);
      return;
    }

    const double target_y = evalPoly(a, b, c, lookahead_x_);
    const double angle = std::atan2(target_y, lookahead_x_);

    std_msgs::msg::Float32 angle_msg;
    angle_msg.data = static_cast<float>(angle);
    angle_pub_->publish(angle_msg);

    publishMarkers(msg->header, true, a, b, c, target_y, centroids);
  }

  //Main logic
  std::vector<Point2D> readForwardPoints(const sensor_msgs::msg::PointCloud2::SharedPtr & msg)
  {
    std::vector<Point2D> points;
    points.reserve(msg->width * msg->height);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    
    //Filtering out points below the lower bound of the path (Forward direction in x axis)
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y) {
      const double x = static_cast<double>(*iter_x);
      const double y = static_cast<double>(*iter_y);
      if (!std::isfinite(x) || !std::isfinite(y)) {
        continue;
      }
      
      if (x < forward_start_) {
        continue;  
      }
      
      points.push_back({x, y});
    }
    return points;
  }

  //Slicing of the path region
  std::vector<Point2D> extractSliceCentroids(const std::vector<Point2D> & points)
  {
    std::vector<Point2D> centroids;

    if (points.empty()) {
      return centroids;
    }

    double max_x = -std::numeric_limits<double>::infinity();
    for (const auto & p : points) {
      max_x = std::max(max_x, p.x);
    }

    const double slicing_range = forward_limit_ - forward_start_;
    if (slicing_range <= 0.0) return centroids; 
    
    const int num_slices = static_cast<int>(std::round(slicing_range / slice_depth_));
    const double eps = 1e-9;

    //Define the slicing boundary for each slices
    for (int i = 0; i < num_slices; ++i) {
      const double x_start = forward_start_ + (i * slice_depth_);
      const double x_end = x_start + slice_depth_;

      if (x_start >= forward_limit_ - eps) {
        break;  
      }
      if (x_start > max_x + eps) {
        break;  
      }

      double min_y = std::numeric_limits<double>::infinity();
      double max_y = -std::numeric_limits<double>::infinity();
      double sum_x = 0.0, sum_y = 0.0;
      int count = 0;

      //find max and min y, and summing up x and y coordinate in a slice
      for (const auto & p : points) {
        if (p.x >= x_start && p.x < x_end) {
          min_y = std::min(min_y, p.y);
          max_y = std::max(max_y, p.y);
          sum_x += p.x;
          sum_y += p.y;
          ++count;
        }
      }

      if (count == 0) {
        continue; 
      }
      //Stop slice if the width is too large
      const double slice_width = max_y - min_y;
      if (slice_width > max_slice_width_) {
        continue; 
      }
      //slice centroids calculation 
      centroids.push_back({sum_x / count, sum_y / count});
    }

    return centroids;
  }

  //Polynomial fitting
  bool fitQuadratic(const std::vector<Point2D> & pts, double & a, double & b, double & c)
  {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;  
    double t0 = 0, t1 = 0, t2 = 0;                  

    for (const auto & p : pts) {
      const double x = p.x;
      const double x2 = x * x;
      const double x3 = x2 * x;
      const double x4 = x3 * x;
      s0 += 1.0;
      s1 += x;
      s2 += x2;
      s3 += x3;
      s4 += x4;
      t0 += p.y;
      t1 += x * p.y;
      t2 += x2 * p.y;
    }

    const double A[3][3] = {
      {s4, s3, s2},
      {s3, s2, s1},
      {s2, s1, s0}
    };
    const double rhs[3] = {t2, t1, t0};
    double sol[3] = {0, 0, 0};

    if (!solve3x3(A, rhs, sol)) {
      return false;
    }

    a = sol[0];
    b = sol[1];
    c = sol[2];
    return true;
  }

  static double evalPoly(double a, double b, double c, double x)
  {
    return a * x * x + b * x + c;
  }

  //Rviz Visualise
  void publishMarkers(
    const std_msgs::msg::Header & header,
    bool have_path,
    double a, double b, double c,
    double target_y,
    const std::vector<Point2D> & centroids)
  {
    visualization_msgs::msg::MarkerArray marker_array;
    int id = 0;

    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header = header;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(clear_marker);

    marker_array.markers.push_back(
      makeLineMarker(
        header, id++, "forward_start", forward_start_,
        0.0, 1.0, 0.0, 1.0, 0.02));

    marker_array.markers.push_back(
      makeLineMarker(
        header, id++, "forward_limit", forward_limit_,
        1.0, 0.0, 0.0, 1.0, 0.02));

    // 3. Lookahead distance indicator
    marker_array.markers.push_back(
      makeLineMarker(
        header, id++, "lookahead_line", lookahead_x_,
        1.0, 1.0, 0.0, 1.0, 0.02));

    marker_array.markers.push_back(makeCentroidMarker(header, id++, centroids));

    if (have_path) {
      marker_array.markers.push_back(makePathMarker(header, id++, a, b, c));
      marker_array.markers.push_back(makeSphereMarker(header, id++, lookahead_x_, target_y));
      marker_array.markers.push_back(makeArrowMarker(header, id++, lookahead_x_, target_y));
    }

    marker_pub_->publish(marker_array);
  }

  visualization_msgs::msg::Marker makeLineMarker(
    const std_msgs::msg::Header & header, int id, const std::string & ns,
    double x, double r, double g, double b_, double a_, double thickness)
  {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = thickness;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b_;
    marker.color.a = a_;
    marker.pose.orientation.w = 1.0;

    geometry_msgs::msg::Point p1, p2;
    p1.x = x; p1.y = -marker_half_span_; p1.z = 0.0;
    p2.x = x; p2.y = marker_half_span_; p2.z = 0.0;
    marker.points.push_back(p1);
    marker.points.push_back(p2);
    return marker;
  }

  visualization_msgs::msg::Marker makePathMarker(
    const std_msgs::msg::Header & header, int id, double a, double b, double c)
  {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "generated_path";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.03;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0;
    marker.pose.orientation.w = 1.0;

    const int steps = 30;
    for (int i = 0; i <= steps; ++i) {
      const double x = forward_start_ + (forward_limit_ - forward_start_) * static_cast<double>(i) / steps;
      geometry_msgs::msg::Point p;
      p.x = x;
      p.y = evalPoly(a, b, c, x);
      p.z = 0.0;
      marker.points.push_back(p);
    }
    return marker;
  }

  visualization_msgs::msg::Marker makeSphereMarker(
    const std_msgs::msg::Header & header, int id, double x, double y)
  {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "intersection_point";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.12;
    marker.scale.y = 0.12;
    marker.scale.z = 0.12;
    marker.color.r = 0.0;
    marker.color.g = 0.4;
    marker.color.b = 1.0;
    marker.color.a = 1.0;
    return marker;
  }

  visualization_msgs::msg::Marker makeArrowMarker(
    const std_msgs::msg::Header & header, int id, double x, double y)
  {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "steering_arrow";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;

    geometry_msgs::msg::Point start, end;
    start.x = 0.0; start.y = 0.0; start.z = 0.0;
    end.x = x; end.y = y; end.z = 0.0;
    marker.points.push_back(start);
    marker.points.push_back(end);

    marker.scale.x = 0.05;  
    marker.scale.y = 0.10;  
    marker.scale.z = 0.10;  
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
    marker.color.a = 1.0;
    return marker;
  }

  visualization_msgs::msg::Marker makeCentroidMarker(
    const std_msgs::msg::Header & header, int id, const std::vector<Point2D> & centroids)
  {
    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "slice_centroids";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;
    marker.color.r = 1.0;
    marker.color.g = 0.65;
    marker.color.b = 0.0;
    marker.color.a = 1.0;

    for (const auto & c : centroids) {
      geometry_msgs::msg::Point p;
      p.x = c.x;
      p.y = c.y;
      p.z = 0.0;
      marker.points.push_back(p);
    }
    return marker;
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr angle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  double slice_depth_;
  double forward_start_; 
  double forward_limit_;
  double lookahead_x_;
  double max_slice_width_;
  int min_valid_centroids_;
  double marker_half_span_;
};

}  

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<path_extraction::PathExtractionNode>());
  rclcpp::shutdown();
  return 0;
}