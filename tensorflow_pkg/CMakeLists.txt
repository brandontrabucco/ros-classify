cmake_minimum_required(VERSION 2.8.3)
project(tensorflow_pkg)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

add_message_files(
  FILES
  Alert.msg
)

add_service_files(
  FILES
  Model.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(classify_client src/classify_client.cpp)
target_link_libraries(classify_client ${catkin_LIBRARIES})
add_dependencies(classify_client tensorflow_pkg_gencpp)
