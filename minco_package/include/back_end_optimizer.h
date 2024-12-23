#pragma once
// #include <map_manager/PCSmap_manager.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Float64.h>
// #include <plan_env/edt_environment.h>

#include <chrono>
#include <thread>

#include "minco.hpp"
#include "../../astar_path_planner/include/astar_path_planner/astar_planner.h"

extern double time_traj_generation;
using namespace std;
class TrajOpt : AStarPlanner {

  public:
      //ros
      ros::NodeHandle nh;
      ros::Publisher debug_pub, debug_wp_pub, draw_cur_pos, draw_cur_ori;  // jxlin: add an extra pub
      ros::Publisher speedPub, accPub, omegaPub, domegaPub;

      // PCSmapManager::Ptr environment;
      // MappingClass::Ptr environment_ptr_;
      // EDTEnvironment::Ptr environment_ptr_new;

      bool pause_debug = true;
      // # pieces and # key points
      int N, K, dim_t, dim_p;
      double ts_;

      // weight for time regularization term
      double rhoT;

      // collision avoiding and dynamics paramters
      double pok, vmax, amax, omegamax, domegamax, vmax_h , amax_h, vmin; 
      double rhoP, rhoV, rhoA, rhoOmega, rhodOmega;
      double rhoTracking_, rhosVisibility_;
      double clearance_d_, tolerance_d_, theta_clearance_;
      // SE3 dynamic limitation parameters
      double thrust_max_, thrust_min_;
      double omega_max_, omega_yaw_max_;
      // corridor
      std::vector<Eigen::MatrixXd> cfgVs_;
      std::vector<Eigen::MatrixXd> cfgHs_;

      // Minimum Jerk Optimizer
      minco::MinJerkOpt jerkOpt;

      // col(0) is P of (x,y,z), col(1) is V .. ， col(2) is A
      Eigen::MatrixXd initS;  
      Eigen::MatrixXd finalS;

      // weight for each vertex
      Eigen::VectorXd p;

      // duration of each piece of the trajectory
      Eigen::VectorXd t;
      // double* x;
      // Eigen::VectorXd x;
      double sum_T;

      std::vector<Eigen::Vector3d> tracking_ps_;
      std::vector<Eigen::Vector3d> tracking_visible_ps_;
      std::vector<double> tracking_thetas_;
      double tracking_dur_;
      double tracking_dist_;
      double tracking_dt_;
      vector<int> motion_state_list_;
      vector<int> motion_direction_list_;
      bool keep_flying_flag_;
      // polyH utils
      bool extractVs(const std::vector<Eigen::MatrixXd>& hPs,
                     std::vector<Eigen::MatrixXd>& vPs) const;

      void drawTraj(Trajectory end_path);
      void drawDebugWp(std::vector<Eigen::Vector3d> end_path);
      void drawCurrentPos(const Eigen::Vector3d &center, const double &radius); // jxlin: 显示实时位置
      void drawCurrentOri(const Eigen::Vector3d &position, const Eigen::Vector3d &velocity); // jxlin: 显示实时朝向
      // void deleteX(){delete[] x;}

      // record cost
      double cost_smooth_, cost_time_, cost_p_, cost_v_, cost_v_low_, cost_a_, cost_w_pos_, cost_w_neg_, cost_dw_pos_, cost_dw_neg_;
      inline void record_clear();
      inline void print_record();

  public:
      TrajOpt() {}
      ~TrajOpt() {}

      void setParam(ros::NodeHandle& nh)
      {
        this->nh = nh;
        nh.param("optimization/K", K, 8);
        nh.param("optimization/pok", pok, 0.3);
        nh.param("optimization/vmax", vmax, 3.0);
        nh.param("optimization/amax", amax, 10.0);
        nh.param("optimization/omegamax", omegamax, 1.0);
        nh.param("optimization/domegamax", domegamax, 1.0);
        nh.param("optimization/vmaxz", vmax_h, 0.2);
        nh.param("optimization/amaxz", amax_h, 0.4);
        nh.param("optimization/vmin", vmin, 0.01);
        nh.param("optimization/rhoT", rhoT, 1000.0);
        nh.param("optimization/rhoP", rhoP, 10000.0);
        nh.param("optimization/rhoV", rhoV, 1000.0);
        nh.param("optimization/rhoA", rhoA, 1000.0);
        nh.param("optimization/rhoOmega", rhoOmega, 1000.0);
        nh.param("optimization/rhodOmega", rhodOmega, 1000.0);
        nh.param("optimization/pause_debug", pause_debug, false);
        debug_pub = nh.advertise<visualization_msgs::Marker>("/traj_opt/debug_path", 10);
        debug_wp_pub = nh.advertise<visualization_msgs::Marker>("/traj_opt/debug_path_wp", 10);
        draw_cur_pos = nh.advertise<visualization_msgs::Marker>("/traj_opt/draw_cur_pos", 10);
        draw_cur_ori = nh.advertise<visualization_msgs::Marker>("/traj_opt/draw_cur_ori", 10); // 当前朝向
        speedPub = nh.advertise<std_msgs::Float64>("/traj_opt/speed", 1000);
        accPub = nh.advertise<std_msgs::Float64>("/traj_opt/accelarate", 1000);
        omegaPub = nh.advertise<std_msgs::Float64>("/traj_opt/omega", 1000);
        domegaPub = nh.advertise<std_msgs::Float64>("/traj_opt/domega", 1000);

      }
      // void setEnvironment(const PCSmapManager::Ptr& mapPtr)
      // {
      //     environment = mapPtr;
      // }

      // void setEnvironment(const MappingClass::Ptr& mappingPtr)
      // {
      //     environment_ptr_ = mappingPtr;
      // }
      // void setEnvironmentNew(const EDTEnvironment::Ptr& mappingPtr)
      // {
      //     environment_ptr_new = mappingPtr;
      // }
      bool generate_traj(const Eigen::MatrixXd& initState,
                         const Eigen::MatrixXd& finalState,
                         const std::vector<Eigen::Vector3d>& Q,
                         const int N,
                         Trajectory& traj,
                         bool keep_result);

      void addTimeIntPenalty(double& cost);
      bool grad_cost_p(const Eigen::Vector3d& p,
                       Eigen::Vector3d& gradp,
                       double& costp);
      bool grad_cost_v(const Eigen::Vector3d& v,
                       Eigen::Vector3d& gradv,
                       double& costv);
      bool grad_cost_a(const Eigen::Vector3d& a,
                       Eigen::Vector3d& grada,
                       double& costa);
      double getMaxOmega(const Trajectory& traj);
      void positiveSmoothedL3(const double &x, double &f, double &df);

  public:
    typedef std::shared_ptr<TrajOpt> Ptr;

};

inline void TrajOpt::record_clear(){
  cost_smooth_ = 0.0;
  cost_time_ = 0.0;
  cost_p_ = 0.0;
  cost_v_ = 0.0;
  cost_v_low_ = 0.0;
  cost_a_ = 0.0;
  cost_w_pos_ = 0.0;
  cost_w_neg_ = 0.0;
  cost_dw_pos_ = 0.0;
  cost_dw_neg_ = 0.0;
}
inline void TrajOpt::print_record(){
  std::cout << "------------Opt Cost---------------" << std::endl;
  std::cout << "cost_smooth: " << cost_smooth_ << std::endl;
  std::cout << "cost_time: " << cost_time_ << std::endl;
  std::cout << "cost_p: " << cost_p_ << std::endl;
  std::cout << "cost_v: " << cost_v_ << std::endl;
  std::cout << "cost_v_low: " << cost_v_low_ << std::endl;
  std::cout << "cost_a: " << cost_a_ << std::endl;
  std::cout << "cost_w_pos: " << cost_w_pos_ << std::endl;
  std::cout << "cost_w_neg: " << cost_w_neg_ << std::endl;
  std::cout << "cost_dw_pos: " << cost_dw_pos_ << std::endl;
  std::cout << "cost_dw_neg: " << cost_dw_neg_ << std::endl;
}