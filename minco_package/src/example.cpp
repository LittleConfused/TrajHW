#include "poly_traj_utils.hpp"
#include "back_end_optimizer.h"



void main()
{
    // 伪代码
    TrajOpt::Ptr minco_traj_optimizer;   
    Trajectory final_traj;             //poly_traj_utils.hpp 中定义 Trajectory类

    minco_traj_optimizer.reset(new TrajOpt);
    minco_traj_optimizer -> setParam(nh);
    //minco_traj_optimizer -> setEnvironment(sdf_map);        //若使用ESDF约束，需要给优化器传递地图类对象

    bool ret_opt;
    MatrixXd initState  = MatrixXd::Zero(3,3);
    MatrixXd finalState = MatrixXd::Zero(3,3);
    initState.col(0)  = START_POS;                             //伪代码，初末位置
    finalState.col(0) = END_POS;


    vector<Vector3d> Q;
    int N;
    //伪代码，假设你有一个path
    for( int ind = 1 ; ind < path_size - 1 ; ind += 1 )         // Q不应包含初末位置
    {
        Q.push_back( path[ind] );
    }
    N = Q.size() + 1;

    ret_opt = minco_traj_optimizer -> generate_traj(initState, finalState, Q, N, final_traj, false);
    if(ret_opt == true)
    {
        publishTraj(final_traj);
    }

}
