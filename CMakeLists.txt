cmake_minimum_required(VERSION 3.8)
project(vitrox_project)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==============================================================================
# DEPENDENCIES
# ==============================================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED) # Added for C++ Time Synchronizer
find_package(pcl_ros REQUIRED)         # Added for fast PCL to ROS transforms
find_package(Eigen3 REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io segmentation filters)
find_package(tf2_eigen REQUIRED)

# ==============================================================================
# C++ EXECUTABLES
# ==============================================================================
add_executable(ground_segmentation_node src/ground_segmentation_node.cpp)
add_executable(path_extraction_node src/path_extraction_node.cpp)
add_executable(cam_fusion_node src/fuse_cam.cpp)

# Include Directories
target_include_directories(path_extraction_node PUBLIC ${PCL_INCLUDE_DIRS})
target_include_directories(ground_segmentation_node PUBLIC ${PCL_INCLUDE_DIRS})
target_include_directories(cam_fusion_node PUBLIC ${PCL_INCLUDE_DIRS})

# Link Libraries
target_link_libraries(path_extraction_node ${PCL_LIBRARIES} Eigen3::Eigen)
target_link_libraries(ground_segmentation_node ${PCL_LIBRARIES})
target_link_libraries(cam_fusion_node ${PCL_LIBRARIES} Eigen3::Eigen)

# Target Dependencies (ROS 2 specific)
ament_target_dependencies(path_extraction_node
  rclcpp
  sensor_msgs
  visualization_msgs
  geometry_msgs
  pcl_conversions
)

ament_target_dependencies(ground_segmentation_node
  rclcpp
  sensor_msgs
  pcl_conversions
  tf2
  tf2_ros
  tf2_sensor_msgs
  visualization_msgs
  tf2_eigen
)

ament_target_dependencies(cam_fusion_node
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_ros            # Added
  message_filters    # Added
  tf2
  tf2_ros
  tf2_sensor_msgs
  visualization_msgs
  tf2_eigen
)

# Install C++ Executables
install(TARGETS
  ground_segmentation_node 
  path_extraction_node
  cam_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

# ==============================================================================
# PYTHON EXECUTABLES & ASSETS
# ==============================================================================
# Install Python scripts directly as PROGRAMS
install(PROGRAMS
  src/cam_fusion.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# ==============================================================================
# TESTING & LINTING
# ==============================================================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()