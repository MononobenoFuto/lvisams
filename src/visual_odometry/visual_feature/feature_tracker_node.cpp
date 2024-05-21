#include "feature_tracker.h"
#include "lvi_sam/cloud_info.h"

#define SHOW_UNDISTORTION 0


// mtx lock for two threads
std::mutex mtx_lidar;

// global variable for saving the depthCloud shared between two threads
pcl::PointCloud<PointType>::Ptr depthCloud(new pcl::PointCloud<PointType>());

// global variables saving the lidar point cloud
deque<pcl::PointCloud<PointType>> cloudQueue;
deque<double> timeQueue;

// global depth register for obtaining depth of a feature
DepthRegister *depthRegister;

// feature publisher for VINS estimator
ros::Publisher pub_feature;
ros::Publisher pub_match;
ros::Publisher pub_restart;
ros::Publisher pub_linematch;

// feature tracker variables
FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

//time
// 05:1317354879.232045
// 06:1317355707.749201
// 07:1317357625.557814
// 08:1317359625.557814
ros::Time T1;
float _T1 = 1317357625.557814;
float match_quality;
ros::Subscriber sub_time;
#define PRINT_MATCHES 0
#define PRINT_MATCH_DOTS 1

std::string Float2Str(double x) {
    int a = static_cast<int>(x);
    int b = static_cast<int>((x - a) * 10000.0);
    a %= 10000;
    std::string sa = std::to_string(a);
    std::string sb = std::to_string(b);
    for(int i = sb.length(); i < 4; ++i) sb = "0" + sb;
    return sa + "." + sb;
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    double cur_img_time = img_msg->header.stamp.toSec();

    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = cur_img_time;
        last_image_time = cur_img_time;
        return;
    }
    // detect unstable camera stream
    if (cur_img_time - last_image_time > 1.0 || cur_img_time < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = cur_img_time;
    // frequency control
    if (round(1.0 * pub_count / (cur_img_time - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (cur_img_time - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = cur_img_time;
            pub_count = 0;
        }
    }
    else
    {
        PUB_THIS_FRAME = false;
    }

    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);


    cv::Mat show_img = ptr->image;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), cur_img_time);
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

        #if SHOW_UNDISTORTION
            trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
        #endif
    }

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header.stamp = img_msg->header.stamp;
        feature_points->header.frame_id = "vins_body";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }

        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);

        // get feature depth from lidar point cloud
        pcl::PointCloud<PointType>::Ptr depth_cloud_temp(new pcl::PointCloud<PointType>());
        mtx_lidar.lock();
        *depth_cloud_temp = *depthCloud;
        mtx_lidar.unlock();

        sensor_msgs::ChannelFloat32 depth_of_points = depthRegister->get_depth(img_msg->header.stamp, show_img, depth_cloud_temp, trackerData[0].m_camera, feature_points->points);
        feature_points->channels.push_back(depth_of_points);

        // ROS_WARN("feature image time  = %f", feature_points->header.stamp.toSec());
        // float delta_time = (feature_points->header.stamp-T1).toSec();
        float delta_time = feature_points->header.stamp.toSec() - _T1;
        bool abnormal = 0;
        match_quality = 0;
        // std::ifstream file("/home/nyamori/catkin_ws/info/07_3/keys.txt");

        // if (file.is_open()) // 检查文件是否成功打开
        // {
        //     std::string line;
        //     while (std::getline(file, line)) // 逐行读取文件内容
        //     {
        //         std::istringstream iss(line); // 创建字符串流
        //         float num1, num2;
        //         iss >> num1 >> num2;
        //         if((delta_time > (num1-0.1)) && (delta_time < (num2+0.1))) {
        //             abnormal = 1; break;
        //         }
        //     }
        //     file.close();
        // } else {
        //     ROS_WARN("Failed to open the file.");
        // }
        // ROS_WARN("ima no jikan=%f hen = %d", delta_time, abnormal);

        
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_feature.publish(feature_points);

        // publish features in image
        if (pub_match.getNumSubscribers() != 0)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::RGB8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;
            // cv::Mat stereo_img = trackerData[0].cur_img;
            cv::Mat cur_img = trackerData[0].cur_img;
            cv::Mat prev_img = trackerData[0].prev_img;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    if (SHOW_TRACK)
                    {
                        // track match_quality
                        double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                        match_quality = match_quality + len;
                        cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(255 * (1 - len), 255 * len, 0), 4);
                    } else {
                        // depth 
                        if(j < depth_of_points.values.size())
                        {
                            if (depth_of_points.values[j] > 0)
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(0, 255, 0), 4);
                            else
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 4, cv::Scalar(0, 0, 255), 4);
                        }
                    }
                }
            }


            cv::cvtColor(cur_img, cur_img, CV_GRAY2RGB);
            cv::cvtColor(prev_img, prev_img, CV_GRAY2RGB);
            cv::Mat matched_img;
            std::vector<cv::DMatch>  matches;
            for(size_t i = 0; i < trackerData[0]._cur_pts.size(); ++i) matches.push_back(cv::DMatch(i, i, 0));
            cv::drawMatches(cur_img, trackerData[0]._cur_pts, prev_img, trackerData[0]._prev_pts, matches, matched_img);
            sensor_msgs::ImagePtr lineMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", matched_img).toImageMsg();
            pub_match.publish(ptr->toImageMsg());
            pub_linematch.publish(lineMsg);


            /*if(abnormal) {
                ROS_WARN("%d", matches.size());
                std::string matched_img_name = "/home/nyamori/catkin_ws/info/07_3/output_" + std::to_string(delta_time) + ".jpg";
                cv::imwrite(matched_img_name, matched_img);
            }*/

            if(PRINT_MATCH_DOTS) {
                std::string filePrefix = "/media/nyamori/8856D74A56D73820/vslam/dataset/07/";

                //cur_img, prev_img
                std::string filename = Float2Str(img_msg->header.stamp.toSec());
                // ROS_WARN("%f %s",img_msg->header.stamp.toSec(),  filename);
                cv::imwrite(filePrefix + filename + "_cur.jpg", trackerData[0].cur_img);
                cv::imwrite(filePrefix + filename + "_prev.jpg", trackerData[0].prev_img);
                std::ofstream mFile(filePrefix + filename + ".txt", std::ios::out|std::ios::app);
                mFile << trackerData[0]._cur_pts.size() << std::endl;
                for(size_t i = 0; i < trackerData[0]._cur_pts.size(); ++i) {
                    mFile << int(trackerData[0]._cur_pts[i].pt.x) << ' ' << int(trackerData[0]._cur_pts[i].pt.y) << ' ';
                    mFile << int(trackerData[0]._prev_pts[i].pt.x) << ' ' << int(trackerData[0]._prev_pts[i].pt.y) << std::endl;
                }
                mFile.close();

                std::ofstream filenameList(filePrefix + "list.txt", std::ios::out|std::ios::app);
                filenameList << filename << std::endl;
                filenameList.close();
            }

            if(PRINT_MATCHES) {
                std::ofstream file("/home/nyamori/catkin_ws/info/07_3/matches.csv", std::ios::out|std::ios::app);
                file << delta_time << ',' << matches.size() << ',' << match_quality << '\n';
                file.close();
            }
        }
    }
}


void lidar_callback(const sensor_msgs::PointCloud2ConstPtr& laser_msg)
{
    static int lidar_count = -1;
    if (++lidar_count % (LIDAR_SKIP+1) != 0)
        return;

    // 0. listen to transform
    static tf::TransformListener listener;
#if IF_OFFICIAL
    static tf::StampedTransform transform;   //; T_vinsworld_camera_FLU
#else
    static tf::StampedTransform transform_world_cFLU;   //; T_vinsworld_camera_FLU
    static tf::StampedTransform transform_cFLU_imu;    //; T_cameraFLU_imu
#endif
    try{
    #if IF_OFFICIAL
        listener.waitForTransform("vins_world", "vins_body_ros", laser_msg->header.stamp, ros::Duration(0.01));
        listener.lookupTransform("vins_world", "vins_body_ros", laser_msg->header.stamp, transform);
    #else   
        //? mod: 监听T_vinsworld_cameraFLU 和 T_cameraFLU_imu
        listener.waitForTransform("vins_world", "vins_cameraFLU", laser_msg->header.stamp, ros::Duration(0.01));
        listener.lookupTransform("vins_world", "vins_cameraFLU", laser_msg->header.stamp, transform_world_cFLU);
        listener.waitForTransform("vins_cameraFLU", "vins_body_imuhz", laser_msg->header.stamp, ros::Duration(0.01));
        listener.lookupTransform("vins_cameraFLU", "vins_body_imuhz", laser_msg->header.stamp, transform_cFLU_imu);
    #endif
    } 
    catch (tf::TransformException ex){
        // ROS_ERROR("lidar no tf");
        return;
    }

    double xCur, yCur, zCur, rollCur, pitchCur, yawCur;
#if IF_OFFICIAL
    xCur = transform.getOrigin().x();
    yCur = transform.getOrigin().y();
    zCur = transform.getOrigin().z();
    tf::Matrix3x3 m(transform.getRotation());
#else
    xCur = transform_world_cFLU.getOrigin().x();
    yCur = transform_world_cFLU.getOrigin().y();
    zCur = transform_world_cFLU.getOrigin().z();
    tf::Matrix3x3 m(transform_world_cFLU.getRotation());
#endif
    m.getRPY(rollCur, pitchCur, yawCur);
    //; T_vinswolrd_cameraFLU
    Eigen::Affine3f transNow = pcl::getTransformation(xCur, yCur, zCur, rollCur, pitchCur, yawCur);

    // 1. convert laser cloud message to pcl
    pcl::PointCloud<PointType>::Ptr laser_cloud_in(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*laser_msg, *laser_cloud_in);

    // 2. downsample new cloud (save memory)
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_ds(new pcl::PointCloud<PointType>());
    static pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(laser_cloud_in);
    downSizeFilter.filter(*laser_cloud_in_ds);
    *laser_cloud_in = *laser_cloud_in_ds;

    // 3. 把lidar坐标系下的点云转到相机的FLU坐标系下表示，因为下一步需要使用相机FLU坐标系下的点云进行初步过滤
#if IF_OFFICIAL
    pcl::PointCloud<PointType>::Ptr laser_cloud_offset(new pcl::PointCloud<PointType>());
    Eigen::Affine3f transOffset = pcl::getTransformation(L_C_TX, L_C_TY, L_C_TZ, L_C_RX, L_C_RY, L_C_RZ);
    pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_offset, transOffset);
    *laser_cloud_in = *laser_cloud_offset;
#else
    pcl::PointCloud<PointType>::Ptr laser_cloud_offset(new pcl::PointCloud<PointType>());
    //; T_cFLU_lidar
    tf::Transform transform_cFLU_lidar = transform_cFLU_imu * Transform_imu_lidar;
    double roll, pitch, yaw, x, y, z;
    x = transform_cFLU_lidar.getOrigin().getX();
    y = transform_cFLU_lidar.getOrigin().getY();
    z = transform_cFLU_lidar.getOrigin().getZ();
    tf::Matrix3x3(transform_cFLU_lidar.getRotation()).getRPY(roll, pitch, yaw);
    Eigen::Affine3f transOffset = pcl::getTransformation(x, y, z, roll, pitch, yaw);
    //; lidar本体坐标系下的点云，转到相机FLU坐标系下表示
    pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_offset, transOffset);
    *laser_cloud_in = *laser_cloud_offset;
#endif

    // 4. filter lidar points (only keep points in camera view)
    //; 根据已经转到相机FLU坐标系下的点云，先排除不在相机FoV内的点云
    pcl::PointCloud<PointType>::Ptr laser_cloud_in_filter(new pcl::PointCloud<PointType>());
    for (int i = 0; i < (int)laser_cloud_in->size(); ++i)
    {
        PointType p = laser_cloud_in->points[i];
        if (p.x >= 0 && abs(p.y / p.x) <= 10 && abs(p.z / p.x) <= 10)
            laser_cloud_in_filter->push_back(p);
    }
    *laser_cloud_in = *laser_cloud_in_filter;

    // 5. transform new cloud into global odom frame
    pcl::PointCloud<PointType>::Ptr laser_cloud_global(new pcl::PointCloud<PointType>());
    //; cameraFLU坐标系下的点云，转到vinsworld系下表示
    pcl::transformPointCloud(*laser_cloud_in, *laser_cloud_global, transNow);

    // 6. save new cloud
    double timeScanCur = laser_msg->header.stamp.toSec();
    cloudQueue.push_back(*laser_cloud_global);
    timeQueue.push_back(timeScanCur);

    // 7. pop old cloud
    while (!timeQueue.empty())
    {
        if (timeScanCur - timeQueue.front() > 5.0)
        {
            cloudQueue.pop_front();
            timeQueue.pop_front();
        } else {
            break;
        }
    }

    std::lock_guard<std::mutex> lock(mtx_lidar);
    // 8. fuse global cloud
    depthCloud->clear();
    for (int i = 0; i < (int)cloudQueue.size(); ++i)
        *depthCloud += cloudQueue[i];

    // 9. downsample global cloud
    pcl::PointCloud<PointType>::Ptr depthCloudDS(new pcl::PointCloud<PointType>());
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(depthCloud);
    downSizeFilter.filter(*depthCloudDS);
    *depthCloud = *depthCloudDS;
}

// extract time stamp
void _GetTime(const lvi_sam::cloud_infoConstPtr &msgIn)
{
    //_T1 = 1317359625.557814;
    if(T1.isZero()) T1 = msgIn->header.stamp;
    sub_time.shutdown();
}

int main(int argc, char **argv)
{
    // initialize ROS node
    ros::init(argc, argv, "vins");
    ros::NodeHandle n;
    ROS_INFO("\033[1;32m----> Visual Feature Tracker Started.\033[0m");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);
    readParameters(n);

    // read camera params
    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    // load fisheye mask to remove features on the boundry
    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_ERROR("load fisheye mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    // initialize depthRegister (after readParameters())
    depthRegister = new DepthRegister(n);
    
    // subscriber to image and lidar
    ros::Subscriber sub_img   = n.subscribe(IMAGE_TOPIC,       5,    img_callback);
    ros::Subscriber sub_lidar = n.subscribe(POINT_CLOUD_TOPIC, 5,    lidar_callback);
    if (!USE_LIDAR)
        sub_lidar.shutdown();
    sub_time = n.subscribe<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/feature/cloud_info", 5, &_GetTime, NULL, ros::TransportHints().tcpNoDelay());


    // messages to vins estimator
    pub_feature = n.advertise<sensor_msgs::PointCloud>(PROJECT_NAME + "/vins/feature/feature",     5);
    pub_match   = n.advertise<sensor_msgs::Image>     (PROJECT_NAME + "/vins/feature/feature_img", 5);
    pub_restart = n.advertise<std_msgs::Bool>         (PROJECT_NAME + "/vins/feature/restart",     5);
    pub_linematch = n.advertise<sensor_msgs::Image>   (PROJECT_NAME + "/vins/feature/feature_line", 5);

    // two ROS spinners for parallel processing (image and lidar)
    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();

    return 0;
}