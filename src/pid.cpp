#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <cmath>     
#include <algorithm> 

class AngularPidNode : public rclcpp::Node
{
public:
    AngularPidNode() : Node("angular_pid_node")
    {
        this->declare_parameter("kp", 5.0);
        this->declare_parameter("ki", 0.0);
        this->declare_parameter("kd", 0.6);
        this->declare_parameter("max_integral", 10.0);
        this->declare_parameter("linear_x", 0.03);   //max speed  
        this->declare_parameter("deadband", 0.02);
        
        // Exponential Scaling & Smoothing Parameters
        this->declare_parameter("min_linear_x", 0.1); 
        this->declare_parameter("speed_reduction_factor", 2.0); 
        this->declare_parameter("speed_smoothing", 0.1);        
        
        // --- NEW: Angular Smoothing Parameter ---
        this->declare_parameter("angular_smoothing", 0.5); // Alpha for steering (0.0 to 1.0)
        
        this->declare_parameter("velocity_topic", "/cmd_vel");

        integral_error_ = 0.0;
        previous_error_ = 0.0;
        last_published_angular_z_ = 0.0;
        last_published_linear_x_ = 0.0; 
        
        current_smoothed_linear_x_ = 0.0; 
        current_smoothed_angular_z_ = 0.0; // --- NEW: Track filtered steering state ---
        
        has_previous_time_ = false;
        first_message_sent_ = false;

        std::string topic_name = this->get_parameter("velocity_topic").as_string();
        velocity_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(topic_name, 10);

        // --- UPDATED: Create Best Effort QoS ---
        rclcpp::QoS best_effort_qos(10);
        best_effort_qos.best_effort();

        // --- UPDATED: Changed Float64 to Float32 and applied Best Effort QoS ---
        error_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "lookahead_angle", best_effort_qos,
            std::bind(&AngularPidNode::errorCallback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "Angular PID Node initialized with Dual Low-Pass Filters.");
    }

private:
    // --- UPDATED: Changed Float64 to Float32 ---
    void errorCallback(const std_msgs::msg::Float32::SharedPtr msg)
    {
        rclcpp::Time current_time = this->now();
        double current_error = msg->data;

        double kp = this->get_parameter("kp").as_double();
        double ki = this->get_parameter("ki").as_double();
        double kd = this->get_parameter("kd").as_double();
        double max_integral = this->get_parameter("max_integral").as_double();
        double max_linear_x = this->get_parameter("linear_x").as_double();
        double deadband = this->get_parameter("deadband").as_double();
        double min_linear_x = this->get_parameter("min_linear_x").as_double();
        double reduction_factor = this->get_parameter("speed_reduction_factor").as_double();
        
        double alpha_linear = this->get_parameter("speed_smoothing").as_double(); 
        double alpha_angular = this->get_parameter("angular_smoothing").as_double(); // --- NEW ---

        double dt = 0.0;
        if (has_previous_time_) {
            dt = (current_time - previous_time_).seconds();
        }

        if (dt <= 0.0) {
            previous_time_ = current_time;
            has_previous_time_ = true;
            return;
        }

        // --- PID MATH ---
        double p_term = kp * current_error;

        integral_error_ += current_error * dt;
        if (integral_error_ > max_integral) integral_error_ = max_integral;
        if (integral_error_ < -max_integral) integral_error_ = -max_integral;
        double i_term = ki * integral_error_;

        double derivative = (current_error - previous_error_) / dt;
        double d_term = kd * derivative;

        // Raw, potentially noisy target calculated by the PID math
        double raw_angular_velocity = p_term + i_term + d_term;

        // --- 1. EXPONENTIAL VELOCITY SCALING ---
        double target_linear_x = max_linear_x * std::exp(-reduction_factor * std::abs(raw_angular_velocity));
        target_linear_x = std::max(min_linear_x, target_linear_x);

        // --- 2. INITIALIZE FILTERS ---
        if (!first_message_sent_) {
            current_smoothed_linear_x_ = max_linear_x;
            current_smoothed_angular_z_ = raw_angular_velocity; // Initialize steering instantly
        }
        
        // --- 3. DISCRETE LOW-PASS FILTERS (EMA) ---
        // Linear Filter
        current_smoothed_linear_x_ = current_smoothed_linear_x_ + alpha_linear * (target_linear_x - current_smoothed_linear_x_);
        
        // Angular Filter (Smoothes out the noisy steering)
        current_smoothed_angular_z_ = current_smoothed_angular_z_ + alpha_angular * (raw_angular_velocity - current_smoothed_angular_z_);

        // --- DEADBAND FILTER & KICKSTART ---
        if (!first_message_sent_ || 
            std::abs(current_smoothed_angular_z_ - last_published_angular_z_) > deadband ||
            std::abs(current_smoothed_linear_x_ - last_published_linear_x_) > deadband) {
            
            auto twist_msg = geometry_msgs::msg::Twist();
            twist_msg.linear.x = current_smoothed_linear_x_;
            
            // Pass the smoothed output to the wheels instead of the raw PID output
            twist_msg.angular.z = current_smoothed_angular_z_; 
            
            velocity_pub_->publish(twist_msg);
            
            last_published_angular_z_ = current_smoothed_angular_z_;
            last_published_linear_x_ = current_smoothed_linear_x_;
            first_message_sent_ = true; 
        }

        previous_error_ = current_error;
        previous_time_ = current_time;
    }

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_pub_;
    // --- UPDATED: Changed Float64 to Float32 ---
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr error_sub_;

    double integral_error_;
    double previous_error_;
    double last_published_angular_z_;
    double last_published_linear_x_;
    
    double current_smoothed_linear_x_; 
    double current_smoothed_angular_z_; // NEW: Steering filter state memory
    
    rclcpp::Time previous_time_;
    bool has_previous_time_;
    bool first_message_sent_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AngularPidNode>());
    rclcpp::shutdown();
    return 0;
}