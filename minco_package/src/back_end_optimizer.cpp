#include <back_end_optimizer.h>
#include <lbfgs.hpp>
//////////////////////////

static double rhoP_tmp_;
double time_traj_generation = 0.0;

// SECTION  variables transformation and gradient transmission
static double expC2(double t) {
  return t > 0.0 ? ((0.5 * t + 1.0) * t + 1.0)
                 : 1.0 / ((0.5 * t - 1.0) * t + 1.0);
}

static double logC2(double T) {
  return T > 1.0 ? (sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T - 1.0));
}

static inline double gdT2t(double t) {
  if (t > 0) {
    return t + 1.0;
  } else {
    double denSqrt = (0.5 * t - 1.0) * t + 1.0;
    return (1.0 - t) / (denSqrt * denSqrt);
  }
}

static void forwardT(const double& t, Eigen::Ref<Eigen::VectorXd> vecT) {
  vecT.setConstant(expC2(t));
}

static void addLayerTGrad(const double& t,
                          const Eigen::Ref<const Eigen::VectorXd>& gradT,
                          double& gradt) {
  gradt = gradT.sum() * gdT2t(t);
}

// !SECTION variables transformation and gradient transmission

// SECTION object function
static inline double objectiveFunc(void* ptrObj,
                                   const Eigen::VectorXd &x,
                                   Eigen::VectorXd &grad
                                   ) {
  TrajOpt& obj = *(TrajOpt*)ptrObj;
  obj.record_clear();
  Eigen::VectorXd VT(obj.N);
  Eigen::Map<const Eigen::MatrixXd> T( x.data() , 1, (obj.dim_t) );
  Eigen::Map<const Eigen::MatrixXd> P(x.data() + (obj.dim_t), 3, (obj.dim_p) );
  Eigen::Map<Eigen::MatrixXd> gradT(grad.data() , 1, (obj.dim_t) );
  Eigen::Map<Eigen::MatrixXd> gradP(grad.data() + (obj.dim_t), 3, (obj.dim_p) );

  //VT = T.row(0);
  double t = T(0,0);
  //VT = VectorXd::Ones( (obj.N) ) * T(0,0);
  forwardT(t, VT);

  (obj.jerkOpt).generate(P, (obj.finalS), VT);  // jxlin: 根据当前状态生成轨迹

  double cost = (obj.jerkOpt).getTrajJerkCost(); // jxlin: 计算当前minmum jerk cost
  obj.cost_smooth_ = cost;
  // std::cout << "cost: " << cost << std::endl; 
  (obj.jerkOpt).calGrads_CT();   // jxlin: 计算关于C和T的梯度
  // obj.addTimeIntPenalty(cost);   // jxlin: 计算额外约束的cost和梯度

  (obj.jerkOpt).calGrads_PT();   // jxlin: 将C转到P和T上  将Gdc转成T和P的导数gradT 和 gradP
  (obj.jerkOpt).gdT.array() += (obj.rhoT);

  cost += (obj.rhoT) * VT.sum();
  obj.cost_time_ = (obj.rhoT) * VT.sum();
  // jxlin: debug t
  // std::cout << "t is " << std::endl;
  // std::cout << VT.sum() << std::endl;
  // std::cout << "------" << std::endl;

  addLayerTGrad(t , (obj.jerkOpt).gdT, gradT(0,0));
  // gradT = (obj.jerkOpt).gdT.transpose();
  gradP = (obj.jerkOpt).gdP;
  // std::cout<<cost<<std::endl;
  return cost;
}


// !SECTION object function
static inline int earlyExit(void* ptrObj,
                            const Eigen::VectorXd &x,
                            const Eigen::VectorXd &grad,
                            const double fx,
                            const double step,
                            int k,
                            int ls) {
  TrajOpt& obj = *(TrajOpt*)ptrObj;

  // show false改为true将动态展示优化的中间过程
  if (false) {
    
    Eigen::VectorXd VT(obj.N);
    Eigen::Map<const Eigen::MatrixXd> T( x.data() , 1, (obj.dim_t) );
    Eigen::Map<const Eigen::MatrixXd> P( x.data() + (obj.dim_t) , 3, (obj.dim_p) );

    // VT = Eigen::VectorXd::Ones( (obj.N) ) * T(0,0);
    
    forwardT(T(0,0), VT);
    obj.jerkOpt.generate(P, obj.finalS, VT);
    auto traj = obj.jerkOpt.getTraj();
    obj.drawTraj(traj);

    // NOTE pause
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return 0;
}

///////////////////////////////
/*
initState 和 finalState: 初末状态， 3*3矩阵
第一列: px,py,pz
第二列: vx,vy,vz
第三列: ax,ay,az

Q: minco 中的q向量
keep_result：若设置为true,优化完成后将不删除(this->x)
*/
/////////////////////////////////


bool TrajOpt::generate_traj(const Eigen::MatrixXd& initState,
                            const Eigen::MatrixXd& finalState,
                            const std::vector<Eigen::Vector3d>& Q,
                            const int N,
                            // const double ts,
                            Trajectory& traj,
                            bool keep_result 
                            // vector<int> motion_state_list,
                            // bool keep_flying_flag, 
                            /*vector<int> motion_direction_list*/) {

  ros::Time time_1 = ros::Time::now();

  // motion_state_list_ = motion_state_list;
  // keep_flying_flag_ = keep_flying_flag;
  // motion_direction_list_ = motion_direction_list;
  //   if(keep_flying_flag_){
  //   std::cout << "keep flying in frontend!: " << std::endl;
  // }

  this->N = N;
  // this->ts_ = ts;
  
  // this->dim_t = N;
  this->dim_t = 1;
  this->dim_p = N - 1;

  // this->x = new double[ (this->dim_t) + 3 * (this->dim_p)];
  Eigen::VectorXd x((this->dim_t) + 3 * (this->dim_p));

  Eigen::VectorXd VT(N);
  Eigen::Map<Eigen::MatrixXd> T( x.data() , 1, (this->dim_t) );
  Eigen::Map<Eigen::MatrixXd> P( x.data() + (this->dim_t) , 3, (this->dim_p) );

  // NOTE set boundary conditions  
  (this->initS)  = initState;
  (this->finalS) = finalState;
  double tempNorm = (this->initS).col(1).norm(); // v0
  (this->initS).col(1) *= tempNorm > (this->vmax) ? ((this->vmax) / tempNorm) : 1.0;
  tempNorm = (this->initS).col(2).norm(); //a0
  (this->initS).col(2) *= tempNorm > (this->amax) ? ((this->amax) / tempNorm) : 1.0;



  // set initial guess
  
  // double len0 = (initState.col(0)  - Q[0]).norm();
  // double lenf = (finalState.col(0) - Q[N-1]).norm();
  // T(0,0)   = len0 / (this->vmax);
  // T(0,N-1) = lenf / (this->vmax);
  // for (int i = 0; i < N; i++)
  // {
  //   // T(0,i) =  (Q[i]  - Q[i-1]).norm() / (this->vmax);
  //   T(0, i) = ts;
  // }
  
  double len = 0.0;
  len += (initState.col(0)  - Q[0]).norm();
  len += (finalState.col(0)  - Q[N-1]).norm();
  for (int i = 1; i < N - 1; i++)
  {
    len +=  (Q[i]  - Q[i-1]).norm() ;
  }
  double T0 = len / N / (this->vmax);
  // double T0 = ts;
  T(0,0) = logC2(T0);
 
  for (int i = 0; i < N - 1; ++i) {
    P.col(i) = Q[i];
  }
  (this->jerkOpt).reset(initState, N);
  // NOTE optimization
  lbfgs::lbfgs_parameter_t lbfgs_params;
  // lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
  // lbfgs_params.mem_size = 1024;
  // lbfgs_params.past = 3;
  // lbfgs_params.g_epsilon = 1e-3;
  // lbfgs_params.min_step = 1e-32;
  // lbfgs_params.delta = 1e-3;
  // lbfgs_params.line_search_type = 0;
  lbfgs_params.mem_size = 32;
  lbfgs_params.past = 3;
  lbfgs_params.g_epsilon = 0.0;
  lbfgs_params.min_step = 1e-32;
  lbfgs_params.delta = 5e-3;
  lbfgs_params.max_linesearch = 256;
  double minObjectiveXY , minObjectiveZ;



  rhoP_tmp_ = (this->rhoP);

  auto opt_ret1 = lbfgs::lbfgs_optimize(x, 
                                       minObjectiveXY,
                                       &objectiveFunc,
                                       nullptr,
                                       &earlyExit,
                                       this,
                                       lbfgs_params);

  // auto opt_ret1 = lbfgs::lbfgs_optimize((this->dim_t) + 3 * (this->dim_p), 
  //                                      this->x, 
  //                                      &minObjectiveXY,
  //                                      &objectiveFunc, nullptr,
  //                                      nullptr, this, &lbfgs_params);

  std::cout << "\033[32m"
            << "ret: " << opt_ret1 << "\033[0m" << std::endl;
  print_record();
  if (this->pause_debug) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }
  if (opt_ret1 < 0) {
    if( !keep_result )
    {
      // delete[] this->x;
      x.setZero();
    }
    return false;
  }

  forwardT(T(0,0), VT);
  // VT = Eigen::VectorXd::Ones(N) * T(0,0);
  (this->jerkOpt).generate(P, finalState, VT);
  traj = (this->jerkOpt).getTraj();
  double max_planning_omega = getMaxOmega(traj);
  std::cout << "max omega: " << max_planning_omega << std::endl;

  ros::Time time_2 = ros::Time::now();
  time_traj_generation = (time_2 - time_1).toSec();
  ROS_WARN("Time consume in Trajectory generation is %f ms", time_traj_generation * 1000.0);
  
  if( !keep_result )
  {
    // delete[] this->x;
    x.setZero();
  }
  
  return true;
}


void TrajOpt::addTimeIntPenalty(double& cost) {
  Eigen::Vector3d pos, vel, acc, jer, sna;
  Eigen::Vector3d grad_tmp, grad_tmp_p, grad_tmp_v ;
  double cost_tmp, cost_tmp_p, cost_tmp_v;
  Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3, beta4;
  double s1, s2, s3, s4, s5;
  double step, alpha;
  Eigen::Matrix<double, 6, 3> gradViolaPc, gradViolaVc, gradViolaAc, gradViolaOmegac, gradViolaOmegaPositivec, gradViolaOmegaNegativec,
                              gradVioladOmegaPositivec, gradVioladOmegaNegativec;
  Eigen::Vector3d gradViolaA_v, gradViolaA_a;
  Eigen::Vector3d gradViolaOmega_v, gradViolaOmega_a;
  Eigen::Vector3d gradViolaOmegaPositive_v, gradViolaOmegaPositive_a;
  Eigen::Vector3d gradViolaOmegaNegative_v, gradViolaOmegaNegative_a;
  Eigen::Vector3d gradVioladOmegaPositive_v, gradVioladOmegaPositive_a, gradVioladOmegaPositive_j;
  Eigen::Vector3d gradVioladOmegaNegative_v, gradVioladOmegaNegative_a, gradVioladOmegaNegative_j;
  double gradViolaPt, gradViolaVt, gradViolaAt, gradViolaOmegat, gradViolaOmegaPositivet, gradViolaOmegaNegativet,
         gradVioladOmegaPositivet, gradVioladOmegaNegativet;
  double omg;
  int innerLoop;


  for (int i = 0; i < N; ++i) {
    // innerLoop = this->ts_ / 0.05 + 1;
    // double little_k = innerLoop - 1;
    const auto& c = jerkOpt.b.block<6, 3>(i * 6, 0);
    step = jerkOpt.T1(i) / K;
    // std::cout << "K: " << K << std::endl;
    // step = jerkOpt.T1(i) / little_k;
    s1 = 0.0;
    innerLoop = K + 1;

    bool backward_flag = false;
    if(i < motion_direction_list_.size()){
      if(motion_direction_list_[i] == 1) backward_flag = true;
    }

    bool ground_flag = false;
    if(motion_state_list_[i+1] == 0 && motion_state_list_[i] == 0){
      //ground traj seg
      // std::cout << "backend gnd seg: " << i << std::endl;
      ground_flag = true;
    }
    else{
      // std::cout << "backend air seg: " << i << std::endl;
      ground_flag = false;     
    }

    for (int j = 0; j < innerLoop; ++j) {
      s2 = s1 * s1;
      s3 = s2 * s1;
      s4 = s2 * s2;
      s5 = s4 * s1;
      beta0 << 1.0, s1, s2, s3, s4, s5;
      beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4;
      beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
      beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;
      beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120 * s1;
      alpha = 1.0 / K * j;
      pos = c.transpose() * beta0;   // jxlin: calculate current position
      vel = c.transpose() * beta1;
      acc = c.transpose() * beta2;
      jer = c.transpose() * beta3;
      sna = c.transpose() * beta4;

      omg = (j == 0 || j == innerLoop - 1) ? 0.5 : 1.0;

      double violaVelPenaD, violaVelZPenaD, violaAccZPenaD, violaAccPenaD, violaOmegaPenaD, violaVelLowPenaD, violaOmegaPositivePenaD, violaOmegaNegativePenaD;
      double violadOmegaPositivePenaD, violadOmegaNegativePenaD;
      double violaVelPena, violaVelZPena, violaAccZPena,  violaAccPena, violaOmegaPena, violaVelLowPena, violaOmegaPositivePena, violaOmegaNegativePena;
      double violadOmegaPositivePena, violadOmegaNegativePena;
      double vel1, acc1_vel1, jer1_vel1, acc1_b_vel1, jer1_b_vel1, acc1_vel1_reci_vel2;
      double z_h5, z_h5_e;
      // Eigen::Matrix<double, 2, 2> B_h;
      // B_h << 0, -1,
      //         1, 0;

      Eigen::Matrix<double, 3, 3> B;
      B << 0, -1, 0,
           1,  0, 0,
           0,  0, 0;
      
      
      if(ground_flag){
        // pos << pos[0], pos[1], 0.0;
        vel[2] = 0.0;
        acc[2] = 0.0;
        jer[2] = 0.0;
        sna[2] = 0.0;        
      }
      // Eigen::Vector3d pos_only_2d, vel_only_2d, acc_only_2d, jerk_only_2d, snap_only_2d;
      // pos_only_2d << pos[0], pos[1], 0.0;
      // vel_only_2d << vel[0], vel[1], 0.0;
      // acc_only_2d << acc[0], acc[1], 0.0;
      // jerk_only_2d << jer[0], jer[1], 0.0;
      // snap_only_2d << sna[0], sna[1], 0.0;

      vel1 = vel.norm();
      acc1_vel1 = acc.transpose() * vel;
      jer1_vel1 = jer.transpose() * vel;
      acc1_b_vel1 = acc.transpose() * B * vel;
      jer1_b_vel1 = jer.transpose() * B * vel;
      // acc1_b_vel1 = fabs(acc_xy.transpose() * B_h * vel_xy);
      // acc1_b_vel1 = fabs(acc.transpose() * B * vel);
      // std::cout << "2x2: " << fabs(acc_xy.transpose() * B_h * vel_xy) << std::endl;
      // std::cout << "3x3: " << acc1_b_vel1 << std::endl;

      double vel2_reci,vel2_reci_e,vel3_2_reci, vel3_2_reci_e,acc2, cur2, cur, omega, omega2, domega;
      double epis = 0.05;
      vel2_reci = 1.0 / (vel1 * vel1);//速度平方分之一
      vel2_reci_e = 1.0 / (vel1 * vel1 + epis);
      vel3_2_reci = vel2_reci * sqrt(vel2_reci);
      vel3_2_reci_e = vel2_reci_e * sqrt(vel2_reci_e);
      
      acc1_vel1_reci_vel2 = acc1_vel1 * vel2_reci;
      // z_h5 = z_h3 * (vel2_reci * vel2_reci * vel2_reci);
      // z_h5_e = z_h3 * (vel2_reci_e * vel2_reci_e * vel2_reci_e);
      // violaVel = 1.0 / vel2_reci - max_vel_ * max_vel_;//速度损失函数

      acc2 = acc1_vel1 * acc1_vel1 * vel2_reci;//加速度平方
      // cur2 = z_h3 * z_h3 * (vel2_reci_e * vel2_reci_e * vel2_reci_e);//曲率平方
      // cur = acc1_b_vel1 * vel3_2_reci;//曲率

      omega = acc1_b_vel1 * vel2_reci;//角速度
      omega2 = omega * omega;
      domega = jer1_b_vel1 * vel2_reci - 2.0 * acc1_b_vel1 * acc1_vel1 * vel2_reci * vel2_reci;

      if(backward_flag){
        omega = -omega;
        domega = -domega;
      }
      // std::cout << "w: " << omega << std::endl;
      Eigen::Vector3d vel_z(0, 0, vel[2]);
      Eigen::Vector3d acc_z(0, 0, acc[2]);

      // double acc1_vel1z = acc_z.transpose() * B * vel_z;
      // double vel2_reciz = 1.0 / (vel1 * vel1);

      double cost_v = vel.squaredNorm() - (this->vmax) * (this->vmax);
      double cost_v_lowerbound = (this->vmin) * (this->vmin) - vel.squaredNorm();
      double cost_a = acc2 - (this->amax) * (this->amax);
      double cost_v_z = vel_z.squaredNorm() - (this->vmax_h) * (this->vmax_h);
      double cost_a_z = acc_z.squaredNorm() - (this->amax_h) * (this->amax_h);
      double cost_omega = omega2 - (this->omegamax) * (this->omegamax);
      double cost_omega_positive = omega - this->omegamax;
      double cost_omega_negative = -omega - this->omegamax;
      double cost_domega_positive = domega - this->domegamax;
      double cost_domega_negative = -domega - this->domegamax;
      
      

    
      if (grad_cost_p(pos, grad_tmp, cost_tmp)) {     
        // if(ground_flag) grad_tmp[2] = 0;  
        // std::cout << "grad_colli: " << grad_tmp << std::endl;
        gradViolaPc = beta0 * grad_tmp.transpose();
        gradViolaPt = alpha * grad_tmp.dot(vel);
        (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += omg * step * gradViolaPc;
        (this->jerkOpt).gdT(i) += omg * (cost_tmp / K/*little_k*/ + step * gradViolaPt);
        cost += omg * step * cost_tmp;
        cost_p_ += omg * step * cost_tmp;
        // std::cout << "p: " << cost_tmp << std::endl;
      }

      if (cost_v > 0.0)
      {
        positiveSmoothedL3(cost_v, violaVelPena, violaVelPenaD);
        // std::cout << "grad_vel: " << gradViolaVc << std::endl;
        gradViolaVc = 2.0 * beta1 * vel.transpose();
        gradViolaVt = 2.0 * alpha * acc.transpose() * vel;
        (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += this->rhoV * omg * step * violaVelPenaD * gradViolaVc;
        (this->jerkOpt).gdT(i) += this->rhoV * omg * (violaVelPena / K + violaVelPenaD * step * gradViolaVt);
        cost += this->rhoV * omg * step * violaVelPena;
        cost_v_ += this->rhoV * omg * step * violaVelPena;
        // std::cout << "v: " << violaVelPena << std::endl;
      }

      if (cost_v_z > 0.0)
      {
        positiveSmoothedL3(cost_v_z, violaVelZPena, violaVelZPenaD);
        // std::cout << "grad_vel: " << gradViolaVc << std::endl;
        gradViolaVc = 2.0 * beta1 * vel_z.transpose();
        gradViolaVt = 2.0 * alpha * acc_z.transpose() * vel;
        (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += this->rhoV * omg * step * violaVelZPenaD * gradViolaVc;
        (this->jerkOpt).gdT(i) += this->rhoV * omg * (violaVelZPena / K + violaVelZPenaD * step * gradViolaVt);
        cost += this->rhoV * omg * step * violaVelZPena;
        cost_v_z += this->rhoV * omg * step * violaVelZPena;
        // std::cout << "v: " << violaVelPena << std::endl;
      }

      if (cost_v_lowerbound > 0.0)  // jxlin: avoid the velocity close to 0
      {
        positiveSmoothedL3(cost_v_lowerbound, violaVelLowPena, violaVelLowPenaD);
        // std::cout << "grad_vellow: " << gradViolaVc << std::endl;
        gradViolaVc = 2.0 * beta1 * vel.transpose();
        gradViolaVt = 2.0 * alpha * acc.transpose() * vel;
        (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += this->rhoV * 1000 * omg * step * violaVelLowPenaD * gradViolaVc;
        (this->jerkOpt).gdT(i) += this->rhoV * 1000 * omg * (violaVelLowPena / K + violaVelLowPenaD * step * gradViolaVt);
        cost += this->rhoV * 1000 * omg * step * violaVelLowPena;
        cost_v_low_ += this->rhoV * 1000 * omg * step * violaVelLowPena;
        // std::cout << "v: " << violaVelPena << std::endl;
      }

      
      if (cost_a > 0.0)
      {
        positiveSmoothedL3(cost_a, violaAccPena, violaAccPenaD);
        // std::cout << "grad_acc: " << gradViolaA_v << std::endl;
        // std::cout << "grad_acc1: " << gradViolaA_a << std::endl;
        // std::cout << "grad_acc2: " << gradViolaAc << std::endl;
        gradViolaA_v = 2.0 * acc1_vel1_reci_vel2 * acc - 2.0 * acc1_vel1_reci_vel2 * acc1_vel1_reci_vel2 * vel;
        gradViolaA_a = 2.0 * acc1_vel1_reci_vel2 * vel;
        gradViolaAc = beta1 * gradViolaA_v.transpose() + beta2 * gradViolaA_a.transpose();
        gradViolaAt = alpha * (acc.dot(gradViolaA_v) + jer.dot(gradViolaA_a));

        // gradViolaAc = 2.0 * beta1 * (acc1_vel1_reci_vel2 * acc.transpose() - acc1_vel1_reci_vel2 * acc1_vel1_reci_vel2 * vel.transpose()) +
        //               2.0 * beta2 * acc1_vel1_reci_vel2 * vel.transpose();
        // gradViolaAt = 2.0 * alpha * (acc1_vel1_reci_vel2 * (acc.squaredNorm() + jer1_vel1) - acc1_vel1_reci_vel2 * acc1_vel1_reci_vel2 * acc1_vel1);

        (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += this->rhoA * omg * step * violaAccPenaD * gradViolaAc;
        (this->jerkOpt).gdT(i) += this->rhoA * omg * (violaAccPena / K + violaAccPenaD * step * gradViolaAt);
        cost += this->rhoA * omg * step * violaAccPena;
        cost_a_ += this->rhoA * omg * step * violaAccPena;
        // std::cout << "a: " << violaAccPena << std::endl;
      }

      // if (cost_a_z > 0.0)
      // {
      //   positiveSmoothedL3(cost_a, violaAccZPena, violaAccZPenaD);
      //   // std::cout << "grad_acc: " << gradViolaA_v << std::endl;
      //   // std::cout << "grad_acc1: " << gradViolaA_a << std::endl;
      //   // std::cout << "grad_acc2: " << gradViolaAc << std::endl;
      //   gradViolaA_v = 2.0 * acc1_vel1_reci_vel2 * acc - 2.0 * acc1_vel1_reci_vel2 * acc1_vel1_reci_vel2 * vel;
      //   gradViolaA_a = 2.0 * acc1_vel1_reci_vel2 * vel;
      //   gradViolaAc = beta1 * gradViolaA_v.transpose() + beta2 * gradViolaA_a.transpose();
      //   gradViolaAt = alpha * (acc.dot(gradViolaA_v) + jer.dot(gradViolaA_a));

      //   // gradViolaAc = 2.0 * beta1 * (acc1_vel1_reci_vel2 * acc.transpose() - acc1_vel1_reci_vel2 * acc1_vel1_reci_vel2 * vel.transpose()) +
      //   //               2.0 * beta2 * acc1_vel1_reci_vel2 * vel.transpose();
      //   // gradViolaAt = 2.0 * alpha * (acc1_vel1_reci_vel2 * (acc.squaredNorm() + jer1_vel1) - acc1_vel1_reci_vel2 * acc1_vel1_reci_vel2 * acc1_vel1);

      //   (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += this->rhoA * omg * step * violaAccPenaD * gradViolaAc;
      //   (this->jerkOpt).gdT(i) += this->rhoA * omg * (violaAccPena / K + violaAccPenaD * step * gradViolaAt);
      //   cost += this->rhoA * omg * step * violaAccPena;
      //   cost_a_ += this->rhoA * omg * step * violaAccPena;
      //   // std::cout << "a: " << violaAccPena << std::endl;
      // }

      if (cost_omega_positive > 0.0 && !keep_flying_flag_)
      {
        positiveSmoothedL3(cost_omega_positive, violaOmegaPositivePena, violaOmegaPositivePenaD);
        gradViolaOmegaPositive_v = vel2_reci * B.transpose() * acc - 2.0 * acc1_b_vel1 * vel2_reci * vel2_reci * vel;
        gradViolaOmegaPositive_a = vel2_reci * B * vel;

        gradViolaOmegaPositivec = beta1 * gradViolaOmegaPositive_v.transpose() + beta2 * gradViolaOmegaPositive_a.transpose();
        gradViolaOmegaPositivet = alpha * (acc.dot(gradViolaOmegaPositive_v) + jer.dot(gradViolaOmegaPositive_a));

        // std::cout << "grad_omg1: " << gradViolaOmegaPositive_v << std::endl;
        // std::cout << "grad_omg2: " << gradViolaOmegaPositive_a << std::endl;
        // std::cout << "grad_omg3: " << gradViolaOmegaPositivec << std::endl;

        (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += this->rhoOmega * omg * step * violaOmegaPositivePenaD * gradViolaOmegaPositivec;
        (this->jerkOpt).gdT(i) += this->rhoOmega * omg * (violaOmegaPositivePena / K + violaOmegaPositivePenaD * step * gradViolaOmegaPositivet);
        cost += this->rhoOmega * omg * step * violaOmegaPositivePena;

        cost_w_pos_ += this->rhoOmega * omg * step * violaOmegaPositivePena;
        // std::cout << "w: " << violaOmegaPena << std::endl;
      }

      if (cost_omega_negative > 0.0 && !keep_flying_flag_)
      {
        positiveSmoothedL3(cost_omega_negative, violaOmegaNegativePena, violaOmegaNegativePenaD);
        gradViolaOmegaNegative_v = -vel2_reci * B.transpose() * acc + 2.0 * acc1_b_vel1 * vel2_reci * vel2_reci * vel;
        gradViolaOmegaNegative_a = -vel2_reci * B * vel;

        gradViolaOmegaNegativec = beta1 * gradViolaOmegaNegative_v.transpose() + beta2 * gradViolaOmegaNegative_a.transpose();
        gradViolaOmegaNegativet = alpha * (acc.dot(gradViolaOmegaNegative_v) + jer.dot(gradViolaOmegaNegative_a));

        // std::cout << "grad_omg11: " << gradViolaOmegaNegative_v << std::endl;
        // std::cout << "grad_omg22: " << gradViolaOmegaNegative_a << std::endl;
        // std::cout << "grad_omg33: " << gradViolaOmegaPositivec << std::endl;

        (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += this->rhoOmega * omg * step * violaOmegaNegativePenaD * gradViolaOmegaNegativec;
        (this->jerkOpt).gdT(i) += this->rhoOmega * omg * (violaOmegaNegativePena / K + violaOmegaNegativePenaD * step * gradViolaOmegaNegativet);
        cost += this->rhoOmega * omg * step * violaOmegaNegativePena;

        cost_w_neg_ += this->rhoOmega * omg * step * violaOmegaNegativePena;
        // std::cout << "w: " << violaOmegaPena << std::endl;
      }

      if (cost_domega_positive > 0.0 && !keep_flying_flag_)
      {
        positiveSmoothedL3(cost_domega_positive, violadOmegaPositivePena, violadOmegaPositivePenaD);
        gradVioladOmegaPositive_v = B.transpose() * jer * vel2_reci - 2.0 * jer1_b_vel1 * vel2_reci * vel2_reci * vel -
                                    2.0 * (acc1_vel1 * B.transpose() * acc + acc1_b_vel1 * acc) * vel2_reci * vel2_reci +
                                    8.0 * acc1_b_vel1 * acc1_vel1 * vel * vel2_reci * vel2_reci * vel2_reci;
        gradVioladOmegaPositive_a = -2.0 * (acc1_vel1 * B * vel + acc1_b_vel1 * vel) * vel2_reci * vel2_reci;
        gradVioladOmegaPositive_j = B * vel * vel2_reci;

        gradVioladOmegaPositivec = beta1 * gradVioladOmegaPositive_v.transpose() + beta2 * gradVioladOmegaPositive_a.transpose() + beta3 * gradVioladOmegaPositive_j.transpose();
        gradVioladOmegaPositivet = alpha * (acc.dot(gradVioladOmegaPositive_v) + jer.dot(gradVioladOmegaPositive_a) + sna.dot(gradVioladOmegaPositive_j));

        // std::cout << "grad_domg11: " << gradVioladOmegaPositive_v << std::endl;
        // std::cout << "grad_domg22: " << gradViolaOmegaNegative_a << std::endl;
        // std::cout << "grad_domg33: " << gradVioladOmegaPositive_j << std::endl;
        // std::cout << "grad_domg44: " << gradVioladOmegaPositivec << std::endl;

        (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += this->rhodOmega * omg * step * violadOmegaPositivePenaD * gradVioladOmegaPositivec;
        (this->jerkOpt).gdT(i) += this->rhodOmega * omg * (violadOmegaPositivePena / K + violadOmegaPositivePenaD * step * gradVioladOmegaPositivet);
        cost += this->rhodOmega * omg * step * violadOmegaPositivePena;

        cost_dw_pos_ += this->rhodOmega * omg * step * violadOmegaPositivePena;
        // std::cout << "dwp: " << violadOmegaPositivePena << std::endl;
      }

      if (cost_domega_negative > 0.0 && !keep_flying_flag_)
      {
        positiveSmoothedL3(cost_domega_negative, violadOmegaNegativePena, violadOmegaNegativePenaD);
        gradVioladOmegaNegative_v = -B.transpose() * jer * vel2_reci + 2.0 * jer1_b_vel1 * vel2_reci * vel2_reci * vel +
                                    2.0 * (acc1_vel1 * B.transpose() * acc + acc1_b_vel1 * acc) * vel2_reci * vel2_reci -
                                    8.0 * acc1_b_vel1 * acc1_vel1 * vel * vel2_reci * vel2_reci * vel2_reci;
        gradVioladOmegaNegative_a = 2.0 * (acc1_vel1 * B * vel + acc1_b_vel1 * vel) * vel2_reci * vel2_reci;
        gradVioladOmegaNegative_j = -B * vel * vel2_reci;

        gradVioladOmegaNegativec = beta1 * gradVioladOmegaNegative_v.transpose() + beta2 * gradVioladOmegaNegative_a.transpose() + beta3 * gradVioladOmegaNegative_j.transpose();
        gradVioladOmegaNegativet = alpha * (acc.dot(gradVioladOmegaNegative_v) + jer.dot(gradVioladOmegaNegative_a) + sna.dot(gradVioladOmegaNegative_j));

        // std::cout << "grad_domg11: " << gradVioladOmegaPositive_v << std::endl;
        // std::cout << "grad_domg22: " << gradViolaOmegaNegative_a << std::endl;
        // std::cout << "grad_domg33: " << gradVioladOmegaPositive_j << std::endl;
        // std::cout << "grad_domg44: " << gradVioladOmegaPositivec << std::endl;

        (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += this->rhodOmega * omg * step * violadOmegaNegativePenaD * gradVioladOmegaNegativec;
        (this->jerkOpt).gdT(i) += this->rhodOmega * omg * (violadOmegaNegativePena / K + violadOmegaNegativePenaD * step * gradVioladOmegaNegativet);
        cost += this->rhodOmega * omg * step * violadOmegaNegativePena;

        cost_dw_neg_ += this->rhodOmega * omg * step * violadOmegaNegativePena;
        // std::cout << "dwn: " << violadOmegaNegativePena << std::endl;
      }

      // if (cost_omega > 0.0)
      // {
      //   positiveSmoothedL3(cost_omega, violaOmegaPena, violaOmegaPenaD);
      //   gradViolaOmega_v = 2.0 * acc1_b_vel1 * vel2_reci * vel2_reci * B.transpose() * acc - 
      //                      4.0 * acc1_b_vel1 * acc1_b_vel1 * vel2_reci * vel2_reci * vel2_reci * vel;
      //   gradViolaOmega_a = 2.0 * acc1_b_vel1 * vel2_reci * vel2_reci * B * vel;

      //   gradViolaOmegac = beta1 * gradViolaOmega_v.transpose() + beta2 * gradViolaOmega_a.transpose();
      //   gradViolaOmegat = alpha * (acc.dot(gradViolaOmega_v) + jer.dot(gradViolaOmega_a));

      //   (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += this->rhoOmega * omg * step * violaOmegaPenaD * gradViolaOmegac;
      //   (this->jerkOpt).gdT(i) += this->rhoOmega * omg * (violaOmegaPena / K + violaOmegaPenaD * step * gradViolaOmegat);
      //   cost += this->rhoOmega * omg * step * violaOmegaPena;

      //   // std::cout << "w: " << violaOmegaPena << std::endl;
      // }


      // if (grad_cost_v(vel, grad_tmp, cost_tmp)) {
      //   gradViolaVc = beta1 * grad_tmp.transpose();
      //   gradViolaVt = alpha * grad_tmp.dot(acc);
      //   // std::cout << "Vc_yes: " << gradViolaVc.transpose() << std::endl;
      //   // std::cout << "Vt_yes: " << gradViolaVt << std::endl;
      //   // std::cout << "---" << std::endl;

      //   (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += omg * step * gradViolaVc;
      //   (this->jerkOpt).gdT(i) += omg * (cost_tmp / K/*little_k*/ + step * gradViolaVt);
      //   cost += omg * step * cost_tmp;
      // }


      // if (grad_cost_a(acc, grad_tmp, cost_tmp)) {
      //   gradViolaAc = beta2 * grad_tmp.transpose();
      //   gradViolaAt = alpha * grad_tmp.dot(jer);
      //   // std::cout << "Ac_yes: " << gradViolaAc.transpose() << std::endl;
      //   // std::cout << "At_yes: " << gradViolaAt << std::endl;
      //   // std::cout << "---" << std::endl;
      //   (this->jerkOpt).gdC.block<6, 3>(i * 6, 0) += omg * step * gradViolaAc;
      //   (this->jerkOpt).gdT(i) += omg * (cost_tmp / K/*little_k*/ + step * gradViolaAt);
      //   cost += omg * step * cost_tmp;
      // }

      s1 += step;
    }
  }
}

//即max(x,0),输入x值，输出函数值和导数值df 保证惩罚函数及其导数梯度连续
void TrajOpt::positiveSmoothedL3(const double &x, double &f, double &df){
  /*df = x * x;
  f = df *x;
  df *= 3.0;*/
  const double pe = 1.0e-4;//为了保证导数连续，在0-pe部分使用光滑段连接
  const double half = 0.5 * pe;
  const double f3c = 1.0 / (pe * pe);
  const double f4c = -0.5 * f3c / pe;
  const double d2c = 3.0 * f3c;
  const double d3c = 4.0 * f4c;

  if (x < pe)//光滑段函数值和求导
  {
      f = (f4c * x + f3c) * x * x * x;
      df = (d3c * x + d2c) * x * x;
  }
  else//线性段函数值和求导
  {
      f = x - half;
      df = 1.0;
  }

  // return ;
}

// true: 当前点跟障碍物的距离小于设定值
bool TrajOpt::grad_cost_p(const Eigen::Vector3d& p,
                          Eigen::Vector3d& gradp,
                          double& costp) {
  
  costp = 0;
  gradp = Eigen::Vector3d::Zero();
  double distance = 0;
  double distance_threshold_short = 0.2;
  double distance_threshold_long = 0.3;

  // TODO
  Eigen::Vector3d pos_m = p - 0.05 * Eigen::Vector3d::Ones();
  Eigen::Vector3i idx;
  auto diff = (p - pos_m) * 10;

    double values[2][2][2];
  for (int x = 0; x < 2; x++)
    for (int y = 0; y < 2; y++)
      for (int z = 0; z < 2; z++) {
        Eigen::Vector3i current_idx = idx + Eigen::Vector3i(x, y, z);
        //values[x][y][z] = getDistance(current_idx);
        values[x][y][z] = calDist(indexToPosition(current_idx.x(), current_idx.y()));

      }

  double v00 = (1 - diff[0]) * values[0][0][0] + diff[0] * values[1][0][0];
  double v01 = (1 - diff[0]) * values[0][0][1] + diff[0] * values[1][0][1];
  double v10 = (1 - diff[0]) * values[0][1][0] + diff[0] * values[1][1][0];
  double v11 = (1 - diff[0]) * values[0][1][1] + diff[0] * values[1][1][1];
  double v0 = (1 - diff[1]) * v00 + diff[1] * v10;
  double v1 = (1 - diff[1]) * v01 + diff[1] * v11;
  double dist = (1 - diff[2]) * v0 + diff[2] * v1;

  gradp[2] = (v1 - v0) * 10;
  gradp[1] = ((1 - diff[2]) * (v10 - v00) + diff[2] * (v11 - v01)) * 10;
  gradp[0] = (1 - diff[2]) * (1 - diff[1]) * (values[1][0][0] - values[0][0][0]);
  gradp[0] += (1 - diff[2]) * diff[1] * (values[1][1][0] - values[0][1][0]);
  gradp[0] += diff[2] * (1 - diff[1]) * (values[1][0][1] - values[0][0][1]);
  gradp[0] += diff[2] * diff[1] * (values[1][1][1] - values[0][1][1]);
  gradp[0] *= 10;

  return dist;



  // environment_ptr_new -> evaluateEDTWithGrad(p, -1.0, distance, gradp);  // jxlin: get the value of distance and gradient of position
  
  // double distance = 0;
  // Eigen::Vector3d gradp = Eigen::Vector3d::Zero();

  if(distance <= distance_threshold_short)
  {
    // std::cout << " occ p: " << p.transpose() << ", dis: " << distance << ", grad: " << gradp.transpose() << std::endl;
    // std::cout << gradp.transpose() << std::endl;
    costp = this->rhoP * pow((distance_threshold_short - distance), 2);
    gradp = -2 * this->rhoP * (distance_threshold_short - distance) * gradp;
    // std::cout << "short cost: " << costp << std::endl; 
    // if (distance <= 0.05){
    //   costp = costp * this->rhoP;
    //   gradp = gradp * this->rhoP;
    //   // std::cout << "cost: " << costp << std::endl; 
    // }
    
  }
  else if (distance <= distance_threshold_long)
  {
    costp = this->rhoP / 10000 * pow((distance_threshold_long - distance), 2);
    gradp = -2 * this->rhoP / 10000 * (distance_threshold_long - distance) * gradp;
    // std::cout << "long cost: " << costp << std::endl; 
  }
  else 
  {
    return false;
  }
  return true;
}

bool TrajOpt::grad_cost_v(const Eigen::Vector3d& v,
                          Eigen::Vector3d& gradv,
                          double& costv) {
  double vpen  = v.squaredNorm() - (this->vmax) * (this->vmax);
  if (vpen > 0) {
    gradv = this->rhoV * 4 * vpen * v;
    costv = this->rhoV * vpen * vpen;
    // std::cout << "vpen is: " << vpen << std::endl;
    return true;
  }
  return false;
}

bool TrajOpt::grad_cost_a(const Eigen::Vector3d& a,
                          Eigen::Vector3d& grada,
                          double& costa) {

  grada = Eigen::Vector3d::Zero();
  costa = 0;
  double apen  = a.squaredNorm() - (this->amax) * (this->amax);

  if (apen > 0) {
    grada += (this->rhoA) * 4 * apen * a;
    costa += (this->rhoA) * apen * apen;
    return true;
  }
 
  return false;
}

/////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////
/// 以下均为可视化相关

void TrajOpt::drawTraj(Trajectory end_path)
{
  int id = 0;
  visualization_msgs::Marker sphere, line_strip;
  sphere.header.frame_id = line_strip.header.frame_id = "map";
  sphere.header.stamp = line_strip.header.stamp = ros::Time::now();
  sphere.type = visualization_msgs::Marker::SPHERE_LIST;
  line_strip.type = visualization_msgs::Marker::LINE_STRIP;
  sphere.action = line_strip.action = visualization_msgs::Marker::ADD;
  sphere.id = id;
  line_strip.id = id + 1000;
  id++;

  sphere.pose.orientation.w = line_strip.pose.orientation.w = 1.0;
  // sphere.color.r = line_strip.color.r = 1;
  // sphere.color.g = line_strip.color.g = 0;
  // sphere.color.b = line_strip.color.b = 1;
  // sphere.color.a = line_strip.color.a = 1;
  sphere.color.r = 0.5;
  sphere.color.g = 0.5;
  sphere.color.b = 1;
  sphere.color.a = 1;

  line_strip.color.r = 1;
  line_strip.color.g = 0;
  line_strip.color.b = 1;
  line_strip.color.a = 1;

  sphere.scale.x = 0.05;
  sphere.scale.y = 0.05;
  sphere.scale.z = 0.05;
  line_strip.scale.x = 0.05 / 2;
  line_strip.scale.y = 0.05 / 2;
  line_strip.scale.z = 0.05 / 2;
  geometry_msgs::Point pt;

  Eigen::VectorXd ts = end_path.getDurations();
  // std::cout << "Ts" << std::endl;
  // std::cout << ts[0] << std::endl;
  

  double dur = end_path.getDurations().sum();
  // jxlin: draw waypoints on the trajectory
  for (double i = 0; i <= dur; i+=ts[0])
  {
    Eigen::Vector3d dur_p = end_path.getPos(i);
    pt.x = dur_p(0);
    pt.y = dur_p(1);
    pt.z = dur_p(2);
    // line_strip.points.push_back(pt);
    sphere.points.push_back(pt);
  }

  for (double i = 0; i < dur - 1e-4; i+=0.01)
  {
    Eigen::Vector3d dur_p = end_path.getPos(i);
    pt.x = dur_p(0);
    pt.y = dur_p(1);
    pt.z = dur_p(2);
    line_strip.points.push_back(pt);
  }

  ros::Rate rate(10); // 10 Hz
  for (int i = 0; i < 10; ++i) {
    debug_pub.publish(line_strip);
    debug_pub.publish(sphere);
    rate.sleep();
  }
}

void TrajOpt::drawDebugWp(std::vector<Eigen::Vector3d> front_path)
{
  int id = 0;
  visualization_msgs::Marker mk;
  mk.header.frame_id = "map";
  mk.header.stamp    = ros::Time::now();
  mk.type            = visualization_msgs::Marker::SPHERE_LIST;
  mk.action          = visualization_msgs::Marker::DELETE;
  mk.id              = id++;
  // kino_pub_.publish(mk);

  mk.action             = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;

  mk.color.r = 0;
  mk.color.g = 1;
  mk.color.b = 0;
  mk.color.a = 1;

  mk.scale.x = 0.075;
  mk.scale.y = 0.075;
  mk.scale.z = 0.075;

  geometry_msgs::Point pt;
  for (int i = 0; i < int(front_path.size()); i++) {
    pt.x = front_path[i](0);
    pt.y = front_path[i](1);
    pt.z = 0;
    mk.points.push_back(pt);
  }
  debug_wp_pub.publish(mk);
  ros::Duration(0.001).sleep();
}

// jxlin: draw current position
void TrajOpt::drawCurrentPos(const Eigen::Vector3d &center, const double &radius)
{
  visualization_msgs::Marker sphereMarkers, sphereDeleter;

  sphereMarkers.id = 0;
  sphereMarkers.type = visualization_msgs::Marker::SPHERE_LIST;
  sphereMarkers.header.stamp = ros::Time::now();
  sphereMarkers.header.frame_id = "map";
  sphereMarkers.pose.orientation.w = 1.00;
  sphereMarkers.action = visualization_msgs::Marker::ADD;
  sphereMarkers.ns = "spheres";
  sphereMarkers.color.r = 0.00;
  sphereMarkers.color.g = 0.00;
  sphereMarkers.color.b = 1.00;
  sphereMarkers.color.a = 1.00;
  sphereMarkers.scale.x = radius;
  sphereMarkers.scale.y = radius;
  sphereMarkers.scale.z = radius;

  sphereDeleter = sphereMarkers;
  sphereDeleter.action = visualization_msgs::Marker::DELETE;

  geometry_msgs::Point point;
  point.x = center(0);
  point.y = center(1);
  point.z = center(2);
  sphereMarkers.points.push_back(point);

  draw_cur_pos.publish(sphereDeleter);
  draw_cur_pos.publish(sphereMarkers);

}

void TrajOpt::drawCurrentOri(const Eigen::Vector3d &position, const Eigen::Vector3d &velocity)
{
  visualization_msgs::Marker arrowMarkers, arrowDeleter;

  arrowMarkers.id = 0;
  arrowMarkers.type = visualization_msgs::Marker::ARROW;
  arrowMarkers.header.stamp = ros::Time::now();
  arrowMarkers.header.frame_id = "map";
  arrowMarkers.pose.position.x = position(0);
  arrowMarkers.pose.position.y = position(1);
  arrowMarkers.pose.position.z = position(2);
  arrowMarkers.pose.orientation.x = 0;
  arrowMarkers.pose.orientation.y = 0;
  arrowMarkers.pose.orientation.z = std::sin(std::atan2(velocity(1), velocity(0)) / 2);
  arrowMarkers.pose.orientation.w = std::cos(std::atan2(velocity(1), velocity(0)) / 2);
  arrowMarkers.action = visualization_msgs::Marker::ADD;
  arrowMarkers.ns = "arrow";
  arrowMarkers.color.r = 0.0;
  arrowMarkers.color.g = 0.0;
  arrowMarkers.color.b = 1.0;
  arrowMarkers.color.a = 1.00;
  arrowMarkers.scale.x = 0.3;
  arrowMarkers.scale.y = 0.05;
  arrowMarkers.scale.z = 0.05;

  arrowDeleter = arrowMarkers;
  arrowDeleter.action = visualization_msgs::Marker::DELETE;

  draw_cur_pos.publish(arrowDeleter);
  draw_cur_pos.publish(arrowMarkers);

}

double TrajOpt::getMaxOmega(const Trajectory& traj){
  double max_omega = 0.0;
  Eigen::Matrix<double, 3, 3> B;
  B << 0, -1, 0,
        1,  0, 0,
        0,  0, 0;
  double max_omega_t;
  Eigen::Vector3d max_omega_pos;
  for (double t = 0.0; t < traj.getTotalDuration(); t += 0.02){
    Eigen::Vector3d pos = traj.getPos(t);
    Eigen::Vector3d vel = traj.getVel(t);
    Eigen::Vector3d acc = traj.getAcc(t);
    double vel1 = vel.head(2).norm();
    double acc1_b_vel1 = acc.transpose() * B * vel;
    double vel2_reci = 1.0 / (vel1 * vel1);//速度平方分之一
    double omega = acc1_b_vel1 * vel2_reci;//角速度

    if (abs(omega) > max_omega) {
      max_omega = abs(omega);
      max_omega_pos = pos;
      max_omega_t = t;
    }
  }
  std::cout << "max omega happens in pos: " << max_omega_pos << 
  std::endl << " t: " << max_omega_t << std::endl;

  return max_omega;
}