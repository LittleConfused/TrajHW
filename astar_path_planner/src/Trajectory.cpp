#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

// 定义路径格式
typedef std::vector<Eigen::Vector2d> Path;

// Step 4: 自行实现轨迹生成类
class TrajectoryGenerator {
    // 轨迹规划的目标是根据A*算法给出的无碰撞路径，计算轨迹航点（控制点），从而生成一条以时间参数化的平滑轨迹，可以用于控制移动机器人跟踪
    // 本次作业中，我们要求生成一条分段多项式轨迹，即每段轨迹均为一个多项式函数
    // 你可以选择使用多项式、B样条、贝塞尔曲线、MINCO等多种轨迹基函数实现
    // 每段轨迹的连接处需要满足一定的连续性条件，如位置连续、速度连续、加速度连续等，这将构成轨迹优化的主要约束条件
    // 轨迹的初始和终止状态为到达指定位置，速度、加速度等状态均为0
    // 优化目标可以是最小化轨迹的加加速度（jerk）或加加加速度（snap），请自行选择合适的优化目标及多项式阶数
    // 本次作业对轨迹段的时间选择不做进一步要求，可以自行选择固定时间或自定义策略生成时间分配
    // 可以任意选用求解器，如qpOASES、OSQP-Eigen等，也可以自行实现闭式解法
public:
    TrajectoryGenerator() = default;

        // your code


};