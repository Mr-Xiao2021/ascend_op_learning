#include "basedriver.h"
#include <signal.h>

sensor_msgs::msg::Imu Mpu6050;
bool need_exit = false;

void ExitHandler(int sig)
{
    (void)sig;
    need_exit = true;
}

BaseDriver::BaseDriver(std::string name) : Node(name),Sampling_Time(0),Power_voltage(0)
{
  memset(&Robot_Pos, 0, sizeof(Robot_Pos));
  memset(&Robot_Vel, 0, sizeof(Robot_Vel));
  memset(&Receive_Data, 0, sizeof(Receive_Data)); 
  memset(&Send_Data, 0, sizeof(Send_Data));
  memset(&Mpu6050_Data, 0, sizeof(Mpu6050_Data));

  this->declare_parameter<std::string>("usart_port_name","serial");
  this->declare_parameter<int>("serial_baud_rate",115200);
  this->declare_parameter<std::string>("odom_frame_id","odom_combined");
  this->declare_parameter<std::string>("robot_frame_id","base_footprint");
  this->declare_parameter<std::string>("gyro_frame_id","gyro_link");
  this->declare_parameter<float>("odom_x_scale",1.0);
  this->declare_parameter<float>("odom_y_scale",1.0);
  this->declare_parameter<float>("odom_z_scale_positive",1.0);
  this->declare_parameter<float>("odom_z_scale_negative",1.0);

  this->get_parameter_or<std::string>("usart_port_name",  usart_port_name,  "/dev/serial"); //Fixed serial port number //固定串口号
  this->get_parameter_or<int>        ("serial_baud_rate", serial_baud_rate, 115200); //Communicate baud rate 115200 to the lower machine //和下位机通信波特率115200
  this->get_parameter_or<std::string>("odom_frame_id",    odom_frame_id,    "odom_combined");      //The odometer topic corresponds to the parent TF coordinate //里程计话题对应父TF坐标
  this->get_parameter_or<std::string>("robot_frame_id",   robot_frame_id,   "base_footprint"); //The odometer topic corresponds to sub-TF coordinates //里程计话题对应子TF坐标
  this->get_parameter_or<std::string>("gyro_frame_id",    gyro_frame_id,    "gyro_link"); //IMU topics correspond to TF coordinates //IMU话题对应TF坐标

  //Odometer correction parameters
  //里程计误差修正参数
  this->get_parameter_or<float>("odom_x_scale",    odom_x_scale,    1.0); 
  this->get_parameter_or<float>("odom_y_scale",    odom_y_scale,    1.0); 
  this->get_parameter_or<float>("odom_z_scale_positive",    odom_z_scale_positive,    1.0); 
  this->get_parameter_or<float>("odom_z_scale_negative",    odom_z_scale_negative,    1.0); 

  voltage_publisher = this->create_publisher<std_msgs::msg::Float32>("PowerVoltage", 10);
  odom_publisher = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
  imu_publisher = this->create_publisher<sensor_msgs::msg::Imu>("imu", 10);
  Cmd_Vel_Sub = this->create_subscription<geometry_msgs::msg::Twist>("cmd_vel", 10, std::bind(&BaseDriver::Cmd_Vel_Callback, this , std::placeholders::_1));

  try
  { 
    //Attempts to initialize and open the serial port //尝试初始化与开启串口
    Stm32_Serial.setPort(usart_port_name); //Select the serial port number to enable //选择要开启的串口号
    Stm32_Serial.setBaudrate(serial_baud_rate); //Set the baud rate //设置波特率
    serial::Timeout _time = serial::Timeout::simpleTimeout(2000); //Timeout //超时等待
    Stm32_Serial.setTimeout(_time);
    Stm32_Serial.open(); //Open the serial port //开启串口
  }
  catch (serial::IOException& e)
  {
    RCLCPP_ERROR(this->get_logger(),"robot can not open serial port,Please check the serial port cable! "); //If opening the serial port fails, an error message is printed //如果开启串口失败，打印错误信息
  }
  if(Stm32_Serial.isOpen())
  {
    RCLCPP_INFO(this->get_logger(),"robot serial port opened"); //Serial port opened successfully //串口开启成功提示
  }

}

BaseDriver::~BaseDriver()
{
  //对象结束前向下位机发送停止运动命令
  Send_Data.tx[0]=FRAME_HEADER;
  Send_Data.tx[1] = 0;  
  Send_Data.tx[2] = 0; 

  //The target velocity of the X-axis of the robot //机器人X轴的目标线速度 
  Send_Data.tx[4] = 0;     
  Send_Data.tx[3] = 0;  

  //The target velocity of the Y-axis of the robot //机器人Y轴的目标线速度 
  Send_Data.tx[6] = 0;
  Send_Data.tx[5] = 0;  

  //The target velocity of the Z-axis of the robot //机器人Z轴的目标角速度 
  Send_Data.tx[8] = 0;  
  Send_Data.tx[7] = 0;    
  Send_Data.tx[9]=Check_Sum(9,SEND_DATA_CHECK); //Check the bits for the Check_Sum function //校验位，规则参见Check_Sum函数
  Send_Data.tx[10]=FRAME_TAIL; 
  try
  {
    Stm32_Serial.write(Send_Data.tx,sizeof (Send_Data.tx)); //Send data to the serial port //向串口发数据  
  }
  catch (serial::IOException& e)   
  {
    RCLCPP_ERROR(this->get_logger(),"Unable to send data through serial port"); //If sending data fails, an error message is printed //如果发送数据失败,打印错误信息
  }
  Stm32_Serial.close(); //Close the serial port //关闭串口  
  RCLCPP_INFO(this->get_logger(),"Shutting down"); //Prompt message //提示信息
}

void BaseDriver::Cmd_Vel_Callback(const geometry_msgs::msg::Twist &twist_aux)
{
  short  transition;  //intermediate variable //中间变量

  //RCLCPP_INFO(this->get_logger(),"cmd_vel: x-%f y-%f",twist_aux.linear.x,twist_aux.angular.z);
  Send_Data.tx[0]=FRAME_HEADER; //frame head 0x7B //帧头0X7B
  Send_Data.tx[1] = 0; //set aside //预留位
  Send_Data.tx[2] = 0; //set aside //预留位

  //The target velocity of the X-axis of the robot
  //机器人x轴的目标线速度
  transition=0;
  transition = twist_aux.linear.x*1000; //将浮点数放大一千倍，简化传输
  Send_Data.tx[4] = transition;     //取数据的低8位
  Send_Data.tx[3] = transition>>8;  //取数据的高8位

  //The target velocity of the Y-axis of the robot
  //机器人y轴的目标线速度
  transition=0;
  transition = twist_aux.linear.y*1000;
  Send_Data.tx[6] = transition;
  Send_Data.tx[5] = transition>>8;

  //The target angular velocity of the robot's Z axis
  //机器人z轴的目标角速度
  transition=0;
  transition = twist_aux.angular.z*1000;
  Send_Data.tx[8] = transition;
  Send_Data.tx[7] = transition>>8;

  Send_Data.tx[9]=Check_Sum(9,SEND_DATA_CHECK); //For the BCC check bits, see the Check_Sum function //BCC校验位，规则参见Check_Sum函数
  Send_Data.tx[10]=FRAME_TAIL; //frame tail 0x7D //帧尾0X7D
  try
  {
    Stm32_Serial.write(Send_Data.tx,sizeof (Send_Data.tx)); //Sends data to the downloader via serial port //通过串口向下位机发送数据 
  }
  catch (serial::IOException& e)   
  {
    RCLCPP_ERROR(this->get_logger(),"Unable to send data through serial port"); //If sending data fails, an error message is printed //如果发送数据失败，打印错误信息
  }
}             

void BaseDriver::Publish_Odom()      //Pub the speedometer topic //发布里程计话题
{ 
	tf2::Quaternion quaternion;
	quaternion.setRPY(0, 0, Robot_Pos.Z);
	geometry_msgs::msg::Quaternion odom_quat;// = tf2::toMsg(quaternion);
	//tf2::toMsg(quaternion, odom_quat);
  odom_quat.x = quaternion.x();
  odom_quat.y = quaternion.y();
  odom_quat.z = quaternion.z();
  odom_quat.w = quaternion.w();
	
	
    nav_msgs::msg::Odometry odom; //Instance the odometer topic data //实例化里程计话题数据
    odom.header.stamp = this->now();//rclcpp::Clock().now(); 
    odom.header.frame_id = odom_frame_id; // Odometer TF parent coordinates //里程计TF父坐标
    odom.pose.pose.position.x = Robot_Pos.X; //Position //位置
    odom.pose.pose.position.y = Robot_Pos.Y;
    odom.pose.pose.position.z = Robot_Pos.Z;
    odom.pose.pose.orientation = odom_quat; //Posture, Quaternion converted by Z-axis rotation //姿态，通过Z轴转角转换的四元数

    odom.child_frame_id = robot_frame_id; // Odometer TF subcoordinates //里程计TF子坐标
    odom.twist.twist.linear.x =  Robot_Vel.X; //Speed in the X direction //X方向速度
    odom.twist.twist.linear.y =  Robot_Vel.Y; //Speed in the Y direction //Y方向速度
    odom.twist.twist.angular.z = Robot_Vel.Z; //Angular velocity around the Z axis //绕Z轴角速度 

    //There are two types of this matrix, which are used when the robot is at rest and when it is moving.Extended Kalman Filtering officially provides 2 matrices for the robot_pose_ekf feature pack
    //这个矩阵有两种，分别在机器人静止和运动的时候使用。扩展卡尔曼滤波官方提供的2个矩阵，用于robot_pose_ekf功能包
    if(Robot_Vel.X== 0&&Robot_Vel.Y== 0&&Robot_Vel.Z== 0)
      //If the velocity is zero, it means that the error of the encoder will be relatively small, and the data of the encoder will be considered more reliable
      //如果velocity是零，说明编码器的误差会比较小，认为编码器数据更可靠
      memcpy(&odom.pose.covariance, odom_pose_covariance2, sizeof(odom_pose_covariance2)),
      memcpy(&odom.twist.covariance, odom_twist_covariance2, sizeof(odom_twist_covariance2));
    else
      //If the velocity of the trolley is non-zero, considering the sliding error that may be brought by the encoder in motion, the data of IMU is considered to be more reliable
      //如果小车velocity非零，考虑到运动中编码器可能带来的滑动误差，认为imu的数据更可靠
      memcpy(&odom.pose.covariance, odom_pose_covariance, sizeof(odom_pose_covariance)),
      memcpy(&odom.twist.covariance, odom_twist_covariance, sizeof(odom_twist_covariance));       
    odom_publisher->publish(odom); //Pub odometer topic //发布里程计话题
}

void BaseDriver::Publish_ImuSensor() //Pub the IMU sensor topic //发布IMU传感器话题
{
  sensor_msgs::msg::Imu Imu_Data_Pub; //Instantiate IMU topic data //实例化IMU话题数据
  Imu_Data_Pub.header.stamp = rclcpp::Clock().now();
  Imu_Data_Pub.header.frame_id = gyro_frame_id; //IMU corresponds to TF coordinates, which is required to use the robot_pose_ekf feature pack 
                                                //IMU对应TF坐标，使用robot_pose_ekf功能包需要设置此项
  Imu_Data_Pub.orientation.x = Mpu6050.orientation.x; //A quaternion represents a three-axis attitude //四元数表达三轴姿态
  Imu_Data_Pub.orientation.y = Mpu6050.orientation.y; 
  Imu_Data_Pub.orientation.z = Mpu6050.orientation.z;
  Imu_Data_Pub.orientation.w = Mpu6050.orientation.w;
  Imu_Data_Pub.orientation_covariance[0] = 1e6; //Three-axis attitude covariance matrix //三轴姿态协方差矩阵
  Imu_Data_Pub.orientation_covariance[4] = 1e6;
  Imu_Data_Pub.orientation_covariance[8] = 1e-6;
  Imu_Data_Pub.angular_velocity.x = Mpu6050.angular_velocity.x; //Triaxial angular velocity //三轴角速度
  Imu_Data_Pub.angular_velocity.y = Mpu6050.angular_velocity.y;
  Imu_Data_Pub.angular_velocity.z = Mpu6050.angular_velocity.z;
  Imu_Data_Pub.angular_velocity_covariance[0] = 1e6; //Triaxial angular velocity covariance matrix //三轴角速度协方差矩阵
  Imu_Data_Pub.angular_velocity_covariance[4] = 1e6;
  Imu_Data_Pub.angular_velocity_covariance[8] = 1e-6;
  Imu_Data_Pub.linear_acceleration.x = Mpu6050.linear_acceleration.x; //Triaxial acceleration //三轴线性加速度
  Imu_Data_Pub.linear_acceleration.y = Mpu6050.linear_acceleration.y; 
  Imu_Data_Pub.linear_acceleration.z = Mpu6050.linear_acceleration.z;  
  imu_publisher->publish(Imu_Data_Pub); //Pub IMU topic //发布IMU话题
}

void BaseDriver::Publish_Voltage()   //Pub the power supply voltage topic //发布电源电压话题
{
	std_msgs::msg::Float32 voltage_msgs; //Define the data type of the power supply voltage publishing topic //定义电源电压发布话题的数据类型
    static float Count_Voltage_Pub=0;
    if(Count_Voltage_Pub++>10)
	{
		Count_Voltage_Pub=0;  
		voltage_msgs.data = Power_voltage; //The power supply voltage is obtained //电源供电的电压获取
		voltage_publisher->publish(voltage_msgs); //Post the power supply voltage topic unit: V, volt //发布电源电压话题单位：V、伏特
	}
}

//从串口(ttyUSB)读取运动底盘速度、IMU、电源电压数据
//Read motion chassis speed, IMU, power supply voltage data from serial port (ttyUSB)
bool BaseDriver::Get_Sensor_Data()   
{
  short transition_16=0, j=0, Header_Pos=0, Tail_Pos=0; //Intermediate variable //中间变量
  static int flag_error=0,temp=1; //Static variable that records the error flag and location //静态变量，用于记录出错标志位和出错位置
  uint8_t Receive_Data_Pr[RECEIVE_DATA_SIZE]={0},Receive_Data_Tr[temp]={0}; //Temporary variable to save the data of the lower machine //临时变量，保存下位机数据
  if(flag_error==0) //Normal condition detected //检测到正常情况
    Stm32_Serial.read(Receive_Data_Pr,sizeof (Receive_Data_Pr)); //Read the data sent by the lower computer through the serial port //通过串口读取下位机发送过来的数据
  else if (flag_error==1) //Error condition detected 检测到错误情况
  {
    //Read wrong bit data through serial port, read only and do not process, so that the correct data is read next time
    //通过串口读取错位数据，只读取不处理，以便于下次读取到的是正确的数据
    Stm32_Serial.read(Receive_Data_Tr,sizeof (Receive_Data_Tr)); 
    flag_error=0; //Error flag position 0 //错误标志位置0
  }

  /*//View the received raw data directly and debug it for use//直接查看接收到的原始数据，调试使用
  RCLCPP_INFO(this->get_logger(),"%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x",
  Receive_Data_Pr[0],Receive_Data_Pr[1],Receive_Data_Pr[2],Receive_Data_Pr[3],Receive_Data_Pr[4],Receive_Data_Pr[5],Receive_Data_Pr[6],Receive_Data_Pr[7],
  Receive_Data_Pr[8],Receive_Data_Pr[9],Receive_Data_Pr[10],Receive_Data_Pr[11],Receive_Data_Pr[12],Receive_Data_Pr[13],Receive_Data_Pr[14],Receive_Data_Pr[15],
  Receive_Data_Pr[16],Receive_Data_Pr[17],Receive_Data_Pr[18],Receive_Data_Pr[19],Receive_Data_Pr[20],Receive_Data_Pr[21],Receive_Data_Pr[22],Receive_Data_Pr[23]);
  */  

  //Record the position of the head and tail of the frame //记录帧头帧尾位置
  for(j=0;j<24;j++)
  {
    if(Receive_Data_Pr[j]==FRAME_HEADER)
    Header_Pos=j;
    else if(Receive_Data_Pr[j]==FRAME_TAIL)
    Tail_Pos=j;    
  }

  if(Tail_Pos==(Header_Pos+23))
  {
    //If the end of the frame is the last bit of the packet, copy the packet directly to receive_data.rx
    //如果帧尾在数据包最后一位，直接复制数据包到Receive_Data.rx
    memcpy(Receive_Data.rx, Receive_Data_Pr, sizeof(Receive_Data_Pr));
    flag_error=0; //Error flag position 0 for next reading //错误标志位置0，便于下次读取
  }
  else if(Header_Pos==(1+Tail_Pos))
  {
    //If the header is behind the end of the frame, record the position of the header so that the next reading of the error bit data can correct the data position
    //如果帧头在帧尾后面，记录帧头出现的位置，便于下次读取出错位数据以纠正数据位置
    //|********7D (7B************|**********7D) 7B************|
    temp=Header_Pos; //Record the length of the next read, calculated to be exactly the position of the frame head //记录下一次读取的长度，经计算正好为帧头的位置
    flag_error=1; //Error flag position 1, error bit array for next read //错误标志位置1，让下一次读取出错位数组
    return false;
  }
  else 
  {
    ////其它情况则认为数据包有错误，这种情况一般是正常的数据，但是除帧头帧尾在数据中间出现了7B或7D的数据
    // In other cases, the packet is considered to be faulty
    // This is generally normal data, but there is 7B or 7D data in the middle of the data except for the frame header and end.
    return false;
  }    
  
  /* //Check receive_data.rx for debugging use //查看Receive_Data.rx，调试使用
  RCLCPP_INFO(this->get_logger(),"%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x",
  Receive_Data.rx[0],Receive_Data.rx[1],Receive_Data.rx[2],Receive_Data.rx[3],Receive_Data.rx[4],Receive_Data.rx[5],Receive_Data.rx[6],Receive_Data.rx[7],
  Receive_Data.rx[8],Receive_Data.rx[9],Receive_Data.rx[10],Receive_Data.rx[11],Receive_Data.rx[12],Receive_Data.rx[13],Receive_Data.rx[14],Receive_Data.rx[15],
  Receive_Data.rx[16],Receive_Data.rx[17],Receive_Data.rx[18],Receive_Data.rx[19],Receive_Data.rx[20],Receive_Data.rx[21],Receive_Data.rx[22],Receive_Data.rx[23]); 
  */

  Receive_Data.Frame_Header= Receive_Data.rx[0]; //The first part of the data is the frame header 0X7B //数据的第一位是帧头0X7B
  Receive_Data.Frame_Tail= Receive_Data.rx[23];  //The last bit of data is frame tail 0X7D //数据的最后一位是帧尾0X7D

  if (Receive_Data.Frame_Header == FRAME_HEADER ) //Judge the frame header //判断帧头
  {
    if (Receive_Data.Frame_Tail == FRAME_TAIL) //Judge the end of the frame //判断帧尾
    { 
      if (Receive_Data.rx[22] == Check_Sum(22,READ_DATA_CHECK)) //BCC check passes or two packets are interlaced //BCC校验通过或者两组数据包交错
      {
        Receive_Data.Flag_Stop=Receive_Data.rx[1]; //set aside //预留位
        Robot_Vel.X = Odom_Trans(Receive_Data.rx[2],Receive_Data.rx[3]); //Get the speed of the moving chassis in the X direction //获取运动底盘X方向速度
        Robot_Vel.Y = Odom_Trans(Receive_Data.rx[4],Receive_Data.rx[5]); //Get the speed of the moving chassis in the Y direction, The Y speed is only valid in the omnidirectional mobile robot chassis
                                                                         //获取运动底盘Y方向速度，Y速度仅在全向移动机器人底盘有效
        Robot_Vel.Z = Odom_Trans(Receive_Data.rx[6],Receive_Data.rx[7]); //Get the speed of the moving chassis in the Z direction //获取运动底盘Z方向速度   
        
        //MPU6050 stands for IMU only and does not refer to a specific model. It can be either MPU6050 or MPU9250
        //Mpu6050仅代表IMU，不指代特定型号，既可以是MPU6050也可以是MPU9250
        Mpu6050_Data.accele_x_data = IMU_Trans(Receive_Data.rx[8],Receive_Data.rx[9]);   //Get the X-axis acceleration of the IMU     //获取IMU的X轴加速度  
        Mpu6050_Data.accele_y_data = IMU_Trans(Receive_Data.rx[10],Receive_Data.rx[11]); //Get the Y-axis acceleration of the IMU     //获取IMU的Y轴加速度
        Mpu6050_Data.accele_z_data = IMU_Trans(Receive_Data.rx[12],Receive_Data.rx[13]); //Get the Z-axis acceleration of the IMU     //获取IMU的Z轴加速度
        Mpu6050_Data.gyros_x_data = IMU_Trans(Receive_Data.rx[14],Receive_Data.rx[15]);  //Get the X-axis angular velocity of the IMU //获取IMU的X轴角速度  
        Mpu6050_Data.gyros_y_data = IMU_Trans(Receive_Data.rx[16],Receive_Data.rx[17]);  //Get the Y-axis angular velocity of the IMU //获取IMU的Y轴角速度  
        Mpu6050_Data.gyros_z_data = IMU_Trans(Receive_Data.rx[18],Receive_Data.rx[19]);  //Get the Z-axis angular velocity of the IMU //获取IMU的Z轴角速度  
        //Linear acceleration unit conversion is related to the range of IMU initialization of STM32, where the range is ±2g=19.6m/s^2
        //线性加速度单位转化，和STM32的IMU初始化的时候的量程有关,这里量程±2g=19.6m/s^2
        Mpu6050.linear_acceleration.x = Mpu6050_Data.accele_x_data / ACCEl_RATIO;
        Mpu6050.linear_acceleration.y = Mpu6050_Data.accele_y_data / ACCEl_RATIO;
        Mpu6050.linear_acceleration.z = Mpu6050_Data.accele_z_data / ACCEl_RATIO;
        //The gyroscope unit conversion is related to the range of STM32's IMU when initialized. Here, the range of IMU's gyroscope is ±500°/s
        //Because the robot generally has a slow Z-axis speed, reducing the range can improve the accuracy
        //陀螺仪单位转化，和STM32的IMU初始化的时候的量程有关，这里IMU的陀螺仪的量程是±500°/s
        //因为机器人一般Z轴速度不快，降低量程可以提高精度
        Mpu6050.angular_velocity.x =  Mpu6050_Data.gyros_x_data * GYROSCOPE_RATIO;
        Mpu6050.angular_velocity.y =  Mpu6050_Data.gyros_y_data * GYROSCOPE_RATIO;
        Mpu6050.angular_velocity.z =  Mpu6050_Data.gyros_z_data * GYROSCOPE_RATIO;

        //Get the battery voltage
        //获取电池电压
        transition_16 = 0;
        transition_16 |=  Receive_Data.rx[20]<<8;
        transition_16 |=  Receive_Data.rx[21];  
        Power_voltage = transition_16/1000+(transition_16 % 1000)*0.001; //Unit conversion millivolt(mv)->volt(v) //单位转换毫伏(mv)->伏(v)

        return true;
     }
    }
  } 
  return false;
}

bool BaseDriver::Get_Sensor_Data_New()
{
  short transition_16=0; //Intermediate variable //中间变量
  uint8_t i=0,check=0, error=1,Receive_Data_Pr[1]; //Temporary variable to save the data of the lower machine //临时变量，保存下位机数据
  static int count; //Static variable for counting //静态变量，用于计数
  Stm32_Serial.read(Receive_Data_Pr,sizeof(Receive_Data_Pr)); //Read the data sent by the lower computer through the serial port //通过串口读取下位机发送过来的数据

  /*//View the received raw data directly and debug it for use//直接查看接收到的原始数据，调试使用
  RCLCPP_INFO(this->get_logger(),"%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x",
  Receive_Data_Pr[0],Receive_Data_Pr[1],Receive_Data_Pr[2],Receive_Data_Pr[3],Receive_Data_Pr[4],Receive_Data_Pr[5],Receive_Data_Pr[6],Receive_Data_Pr[7],
  Receive_Data_Pr[8],Receive_Data_Pr[9],Receive_Data_Pr[10],Receive_Data_Pr[11],Receive_Data_Pr[12],Receive_Data_Pr[13],Receive_Data_Pr[14],Receive_Data_Pr[15],
  Receive_Data_Pr[16],Receive_Data_Pr[17],Receive_Data_Pr[18],Receive_Data_Pr[19],Receive_Data_Pr[20],Receive_Data_Pr[21],Receive_Data_Pr[22],Receive_Data_Pr[23]);
  */  

  Receive_Data.rx[count] = Receive_Data_Pr[0]; //Fill the array with serial data //串口数据填入数组

  Receive_Data.Frame_Header = Receive_Data.rx[0]; //The first part of the data is the frame header 0X7B //数据的第一位是帧头0X7B
  Receive_Data.Frame_Tail = Receive_Data.rx[23];  //The last bit of data is frame tail 0X7D //数据的最后一位是帧尾0X7D

  if(Receive_Data_Pr[0] == FRAME_HEADER || count>0) //Ensure that the first data in the array is FRAME_HEADER //确保数组第一个数据为FRAME_HEADER
    count++;
  else 
  	count=0;
  if(count == 24) //Verify the length of the packet //验证数据包的长度
  {
    count=0;  //Prepare for the serial port data to be refill into the array //为串口数据重新填入数组做准备
    if(Receive_Data.Frame_Tail == FRAME_TAIL) //Verify the frame tail of the packet //验证数据包的帧尾
    {
      check=Check_Sum(22,READ_DATA_CHECK);  //BCC check passes or two packets are interlaced //BCC校验通过或者两组数据包交错

      if(check == Receive_Data.rx[22])  
      {
        error=0;  //XOR bit check successful //异或位校验成功
      }
      if(error == 0)
      {
        /*//Check receive_data.rx for debugging use //查看Receive_Data.rx，调试使用 
        RCLCPP_INFO(this->get_logger(),"%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x-%x",
        Receive_Data.rx[0],Receive_Data.rx[1],Receive_Data.rx[2],Receive_Data.rx[3],Receive_Data.rx[4],Receive_Data.rx[5],Receive_Data.rx[6],Receive_Data.rx[7],
        Receive_Data.rx[8],Receive_Data.rx[9],Receive_Data.rx[10],Receive_Data.rx[11],Receive_Data.rx[12],Receive_Data.rx[13],Receive_Data.rx[14],Receive_Data.rx[15],
        Receive_Data.rx[16],Receive_Data.rx[17],Receive_Data.rx[18],Receive_Data.rx[19],Receive_Data.rx[20],Receive_Data.rx[21],Receive_Data.rx[22],Receive_Data.rx[23]); 
        */

        Receive_Data.Flag_Stop=Receive_Data.rx[1]; //set aside //预留位
        Robot_Vel.X = Odom_Trans(Receive_Data.rx[2],Receive_Data.rx[3]); //Get the speed of the moving chassis in the X direction //获取运动底盘X方向速度
          
        Robot_Vel.Y = Odom_Trans(Receive_Data.rx[4],Receive_Data.rx[5]); //Get the speed of the moving chassis in the Y direction, The Y speed is only valid in the omnidirectional mobile robot chassis
                                                                          //获取运动底盘Y方向速度，Y速度仅在全向移动机器人底盘有效
        Robot_Vel.Z = Odom_Trans(Receive_Data.rx[6],Receive_Data.rx[7]); //Get the speed of the moving chassis in the Z direction //获取运动底盘Z方向速度   
          
        //MPU6050 stands for IMU only and does not refer to a specific model. It can be either MPU6050 or MPU9250
        //Mpu6050仅代表IMU，不指代特定型号，既可以是MPU6050也可以是MPU9250
        Mpu6050_Data.accele_x_data = IMU_Trans(Receive_Data.rx[8],Receive_Data.rx[9]);   //Get the X-axis acceleration of the IMU     //获取IMU的X轴加速度  
        Mpu6050_Data.accele_y_data = IMU_Trans(Receive_Data.rx[10],Receive_Data.rx[11]); //Get the Y-axis acceleration of the IMU     //获取IMU的Y轴加速度
        Mpu6050_Data.accele_z_data = IMU_Trans(Receive_Data.rx[12],Receive_Data.rx[13]); //Get the Z-axis acceleration of the IMU     //获取IMU的Z轴加速度
        Mpu6050_Data.gyros_x_data = IMU_Trans(Receive_Data.rx[14],Receive_Data.rx[15]);  //Get the X-axis angular velocity of the IMU //获取IMU的X轴角速度  
        Mpu6050_Data.gyros_y_data = IMU_Trans(Receive_Data.rx[16],Receive_Data.rx[17]);  //Get the Y-axis angular velocity of the IMU //获取IMU的Y轴角速度  
        Mpu6050_Data.gyros_z_data = IMU_Trans(Receive_Data.rx[18],Receive_Data.rx[19]);  //Get the Z-axis angular velocity of the IMU //获取IMU的Z轴角速度  
        //Linear acceleration unit conversion is related to the range of IMU initialization of STM32, where the range is ±2g=19.6m/s^2
        //线性加速度单位转化，和STM32的IMU初始化的时候的量程有关,这里量程±2g=19.6m/s^2
        Mpu6050.linear_acceleration.x = Mpu6050_Data.accele_x_data / ACCEl_RATIO;
        Mpu6050.linear_acceleration.y = Mpu6050_Data.accele_y_data / ACCEl_RATIO;
        Mpu6050.linear_acceleration.z = Mpu6050_Data.accele_z_data / ACCEl_RATIO;
        //The gyroscope unit conversion is related to the range of STM32's IMU when initialized. Here, the range of IMU's gyroscope is ±500°/s
        //Because the robot generally has a slow Z-axis speed, reducing the range can improve the accuracy
        //陀螺仪单位转化，和STM32的IMU初始化的时候的量程有关，这里IMU的陀螺仪的量程是±500°/s
        //因为机器人一般Z轴速度不快，降低量程可以提高精度
        Mpu6050.angular_velocity.x =  Mpu6050_Data.gyros_x_data * GYROSCOPE_RATIO;
        Mpu6050.angular_velocity.y =  Mpu6050_Data.gyros_y_data * GYROSCOPE_RATIO;
        Mpu6050.angular_velocity.z =  Mpu6050_Data.gyros_z_data * GYROSCOPE_RATIO;

        //Get the battery voltage
        //获取电池电压
        transition_16 = 0;
        transition_16 |=  Receive_Data.rx[20]<<8;
        transition_16 |=  Receive_Data.rx[21];  
        Power_voltage = transition_16/1000+(transition_16 % 1000)*0.001; //Unit conversion millivolt(mv)->volt(v) //单位转换毫伏(mv)->伏(v)
          
        return true;

      }
      
    }

  }
  return false;
}

unsigned char BaseDriver::Check_Sum(unsigned char Count_Number,unsigned char mode) //BBC check function //BBC校验函数
{
  unsigned char check_sum=0,k;
  
  if(mode==0) //Receive data mode //接收数据模式
  {
   for(k=0;k<Count_Number;k++)
    {
     check_sum=check_sum^Receive_Data.rx[k]; //By bit or by bit //按位异或
     }
  }
  if(mode==1) //Send data mode //发送数据模式
  {
   for(k=0;k<Count_Number;k++)
    {
     check_sum=check_sum^Send_Data.tx[k]; //By bit or by bit //按位异或
     }
  }
  return check_sum; //Returns the bitwise XOR result //返回按位异或结果
}

short BaseDriver::IMU_Trans(uint8_t Data_High,uint8_t Data_Low)  //IMU data conversion read //IMU数据转化读取
{
  short transition_16;
  transition_16 = 0;
  transition_16 |=  Data_High<<8;   
  transition_16 |=  Data_Low;
  return transition_16;  

}

float BaseDriver::Odom_Trans(uint8_t Data_High,uint8_t Data_Low) //Odometer data is converted to read //里程计数据转化读取
{
  float data_return;
  short transition_16;
  transition_16 = 0;
  transition_16 |=  Data_High<<8;  //Get the high 8 bits of data   //获取数据的高8位
  transition_16 |=  Data_Low;      //Get the lowest 8 bits of data //获取数据的低8位
  data_return   =  (transition_16 / 1000)+(transition_16 % 1000)*0.001; // The speed unit is changed from mm/s to m/s //速度单位从mm/s转换为m/s
  return data_return;
}

void BaseDriver::Control()
{
  while(rclcpp::ok())
  {
    if (true == Get_Sensor_Data_New()) //通过串口读取并校验下位机发送过来的数据，然后数据转换为国际单位                                       
    {
      _Now = this->now();//rclcpp::Clock().now();
      if(_Last_Time.seconds()==0) _Last_Time=_Now; //首次进入时进行此操作，避免积分时间过大
                                                 
      Sampling_Time = (_Now - _Last_Time).seconds(); //获取时间间隔，用于积分速度获得位移(里程)
                                                   
      
      //Odometer correction parameters
      //里程计误差修正
      Robot_Vel.X=Robot_Vel.X*odom_x_scale;
      Robot_Vel.Y=Robot_Vel.Y*odom_y_scale;
      if(Robot_Vel.Z>=0)
        Robot_Vel.Z=Robot_Vel.Z*odom_z_scale_positive;
      else
        Robot_Vel.Z=Robot_Vel.Z*odom_z_scale_negative;

      //Speed * Time = displacement (odometer)
      //速度*时间=位移（里程计）
      Robot_Pos.X+=(Robot_Vel.X * cos(Robot_Pos.Z) - Robot_Vel.Y * sin(Robot_Pos.Z)) * Sampling_Time; //Calculate the displacement in the X direction, unit: m //计算X方向的位移，单位：m
      Robot_Pos.Y+=(Robot_Vel.X * sin(Robot_Pos.Z) + Robot_Vel.Y * cos(Robot_Pos.Z)) * Sampling_Time; //Calculate the displacement in the Y direction, unit: m //计算Y方向的位移，单位：m
      Robot_Pos.Z+=Robot_Vel.Z * Sampling_Time; //The angular displacement about the Z axis, in rad //绕Z轴的角位移，单位：rad 

      //Calculate the three-axis attitude from the IMU with the angular velocity around the three-axis and the three-axis acceleration
      //通过IMU绕三轴角速度与三轴加速度计算三轴姿态
      Quaternion_Solution(Mpu6050.angular_velocity.x, Mpu6050.angular_velocity.y, Mpu6050.angular_velocity.z,\
                Mpu6050.linear_acceleration.x, Mpu6050.linear_acceleration.y, Mpu6050.linear_acceleration.z);

      Publish_Odom();      //Pub the speedometer topic //发布里程计话题
      Publish_ImuSensor(); //Pub the IMU topic //发布IMU话题    
      Publish_Voltage();   //Pub the topic of power supply voltage //发布电源电压话题

      _Last_Time = _Now; //Record the time and use it to calculate the time interval //记录时间，用于计算时间间隔
    }
    rclcpp::spin_some(shared_from_this());
    //The loop waits for the callback function //循环等待回调函数
  }
}




int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    /*产生一个的节点*/
    auto node = std::make_shared<BaseDriver>("basedriver");
    node->Control();
    rclcpp::shutdown();
    return 0;
}
