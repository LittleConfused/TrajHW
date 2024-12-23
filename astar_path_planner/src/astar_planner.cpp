#include <ros/ros.h>
// #include "../include/astar_path_planner/astar_planner.h"
#include "visualization_msgs/MarkerArray.h"
#include <geometry_msgs/Point.h>
#include "nav_msgs/Path.h"

#include "../../minco_package/include/back_end_optimizer.h"



int main(int argc, char** argv) {
    ros::init(argc, argv, "astar_planner");
    ros::NodeHandle nh;
    double map_min_, map_max_, grid_resolution_;
    double start_x_, start_y_, goal_x_, goal_y_;
    nh.param("astar_planner/map_min", map_min_, -5.0);
    nh.param("astar_planner/map_max", map_max_, 5.0);
    nh.param("astar_planner/grid_resolution", grid_resolution_, 0.1);
    nh.param("astar_planner/start_x", start_x_, -4.5);
    nh.param("astar_planner/start_y", start_y_, -4.5);
    nh.param("astar_planner/goal_x", goal_x_, 4.5);
    nh.param("astar_planner/goal_y", goal_y_, 4.5);

    // 地图参数
    int grid_width = std::round((map_max_ - map_min_) / grid_resolution_);
    int grid_height = grid_width;

    AStarPlanner planner(grid_width, grid_height, map_min_, map_max_, grid_resolution_);
    // 障碍物订阅
    ros::Subscriber obstacle_sub = nh.subscribe<visualization_msgs::MarkerArray>("obstacles", 1,
                                                                                 [&planner, &grid_resolution_, &map_min_](const visualization_msgs::MarkerArray::ConstPtr& msg) {
                                                                                     for (const auto& marker : msg->markers) {
                                                                                         planner.setObstacle(marker.pose.position.x, marker.pose.position.y, marker.scale.x / 2.0);
                                                                                     }
                                                                                 });



    // 发布路径
    ros::Rate rate(10);
    ros::Publisher path_pub = nh.advertise<nav_msgs::Path>("path", 1);
    // 起点和终点参数
    Eigen::Vector2d start(start_x_, start_y_);
    Eigen::Vector2d goal(goal_x_, goal_y_);
    while (ros::ok()) {
        planner.reset();
//        // 等待障碍物加载
//        ros::Duration(1.0).sleep();
        ros::spinOnce();
        // 执行路径搜索
        std::vector<Eigen::Vector2d> path = planner.findPath(start, goal);

        // 路径可视化
        if (path.empty()){
            continue;
        }
        nav_msgs::Path path_msg;
        path_msg.header.frame_id = "map";
        path_msg.header.stamp = ros::Time::now();
        for (const auto& point : path) {
            geometry_msgs::PoseStamped pose;
            pose.pose.position.x = point.x();
            pose.pose.position.y = point.y();
            pose.pose.position.z = 0.0; // 平面路径，z 设置为 0
            path_msg.poses.push_back(pose);
        }
        path_pub.publish(path_msg);

        TrajOpt::Ptr minco_traj_optimizer;
        Trajectory final_traj;
        minco_traj_optimizer.reset(new TrajOpt);
        minco_traj_optimizer -> setParam(nh);

        bool ret_opt;
        Eigen::MatrixXd initState  = Eigen::MatrixXd::Zero(3,3);
        Eigen::MatrixXd finalState = Eigen::MatrixXd::Zero(3,3);
        initState.col(0)  = Eigen::Vector3d(start_x_, start_y_, 0.0);                             //伪代码，初末位置
        finalState.col(0) = Eigen::Vector3d(path[path.size()-1].x(), path[path.size()-1].y(), 0.0);

        std::vector<Eigen::Vector3d> Q;
        int N;  

        for (int ind = 1; ind < path.size() - 1; ind += 1)         // Q不应包含初末位置
        {
            Q.push_back( Eigen::Vector3d(path[ind].x(), path[ind].y(), 0.0) );
        }
        N = Q.size() + 1;

        ret_opt = minco_traj_optimizer -> generate_traj(initState, finalState, Q, N, final_traj, false);
        if(ret_opt == true)
        {
            minco_traj_optimizer -> drawTraj(final_traj);
        }

        rate.sleep();
    }

    return 0;
}