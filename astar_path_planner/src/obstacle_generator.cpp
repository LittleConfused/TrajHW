#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <random>
#include <Eigen/Dense>


visualization_msgs::Marker createCylinderMarker(int id, double x, double y, double radius) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "obstacles";
    marker.id = id;
    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::ADD;

    // 设置圆柱体的位置
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = 0.5; // 假设圆柱体高度为1m
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    // 设置圆柱体的尺寸
    marker.scale.x = radius * 2; // 圆柱体直径
    marker.scale.y = radius * 2;
    marker.scale.z = 1.0; // 高度固定为1m

    // 设置颜色
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 0.8; // 半透明

    return marker;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "obstacle_generator");
    ros::NodeHandle nh;
    double map_min_, map_max_, num_obstacles_, min_radius_, max_radius_;
    nh.param("obstacle_generator/map_min", map_min_, -5.0);
    nh.param("obstacle_generator/map_max", map_max_, 5.0);
    nh.param("obstacle_generator/num_obstacles", num_obstacles_, 10.0);
    nh.param("obstacle_generator/min_radius", min_radius_, 0.2);
    nh.param("obstacle_generator/max_radius", max_radius_, 0.5);

    ros::Publisher obstacle_pub = nh.advertise<visualization_msgs::MarkerArray>("obstacles", 1);


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> pos_dist(map_min_, map_max_);
    std::uniform_real_distribution<> radius_dist(min_radius_, max_radius_);
    visualization_msgs::MarkerArray marker_array;
    for (int i = 0; i < num_obstacles_; ++i) {
        double x = pos_dist(gen);
        double y = pos_dist(gen);
        double radius = radius_dist(gen);
        marker_array.markers.push_back(createCylinderMarker(i, x, y, radius));
    }
    ros::Rate rate(1);
    ros::Duration(3).sleep();
    while (ros::ok()) {
        obstacle_pub.publish(marker_array);
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}