cmake_minimum_required(VERSION 2.8.3)
project(minco_package)

find_package(OpenCV REQUIRED)

set(CMAKE_BUILD_TYPE "Release")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  # plan_env
  cv_bridge
#   visualization_msgs
#   cv_bridge
  message_filters
  # astar_path_planner
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES minco_package
 CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS} -O3 -Wall")

add_library( minco_package 
    src/back_end_optimizer.cpp 

    )
target_link_libraries( minco_package
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    )  

# add_executable(obj_generator
#     src/obj_generator.cpp 
#     )
# target_link_libraries(obj_generator 
#     ${catkin_LIBRARIES})