#include "ros/ros.h"
#include "tensorflow_pkg/Model.h"
#include <cstdlib>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "classify_client");
  if (argc != 1)
  {
    ROS_INFO("usage: classify_client");
    return 1;
  }

  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<tensorflow_pkg::Model>("classify");
  tensorflow_pkg::Model srv;
  srv.request.input_array = {
      -1.0, 0.1, 0.2, 0.0, 
      0.0, 0.7, 3.0, 0.0, 
      0.0, -0.1, -0.1, 0.0, 
      5.0, 2.0, 0.9, -0.1};
  if (client.call(srv))
  {
    ROS_INFO("Label: %d", (int)srv.response.class_label);
  }
  else
  {
    ROS_ERROR("Failed to call service add_two_ints");
    return 1;
  }

  return 0;
}
