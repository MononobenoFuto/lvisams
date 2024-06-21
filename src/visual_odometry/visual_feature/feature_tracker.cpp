#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

std::string intToStringWithLeadingZeros(uint number) {
    std::ostringstream oss;
    oss << std::setw(10) << std::setfill('0') << number;
    return oss.str();
}

int getLabelId(cv::Point2f &p, int shape) {
    int img_x = cvRound(p.x);
    int img_y = cvRound(p.y);
    return img_y * shape + img_x;
}

torch::Tensor matToTensor(const cv::Mat& mat) {
    // 将 cv::Mat 转换为 torch::Tensor
    cv::Mat float_mat;
    mat.convertTo(float_mat, CV_32F); // 将数据类型转换为 float
    auto tensor = torch::from_blob(float_mat.data, {mat.rows, mat.cols, mat.channels()}, torch::kFloat);
    tensor = tensor.permute({2, 0, 1}); // 调整维度顺序为 CxHxW
    return tensor.clone(); // 复制数据以确保与 cv::Mat 数据独立
}

FeatureTracker::FeatureTracker() {
    try {
        module = torch::jit::load("/home/nyamori/catkin_ws/src/LVI-SAM-Easyused/model_script.pt");
    }
    catch (const c10::Error& e) {
        ROS_WARN("Error loading the model\n");
    }
    ROS_WARN("Model loaded successfully\n");
    module.eval();

}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time, uint seq)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
        prev_seq = cur_seq = forw_seq = seq;
    }
    else
    {
        forw_img = img;
        forw_seq = seq;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);
        for (int i = 0; i < int(forw_pts.size()); i++) {
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        }
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());


        _cur_pts.clear(); _prev_pts.clear();
        for(const auto & point: forw_pts) {
            _cur_pts.push_back(cv::KeyPoint(point, 1.0));
        }
        for(const auto & point: cur_pts) {
            _prev_pts.push_back(cv::KeyPoint(point, 1.0));
        }
    }



    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        
        torch::NoGradGuard no_grad;
        int64_t model_h = 376, model_w = 1232, model_c = 19;
        cv::Mat ex_forw_img;
        cv::cvtColor(forw_img, ex_forw_img, cv::COLOR_GRAY2BGR);
        cv::copyMakeBorder(ex_forw_img, ex_forw_img, 0, 376-370, 0, 1232-1226, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        torch::Tensor forw_tensor = matToTensor(ex_forw_img);
        forw_tensor = forw_tensor.unsqueeze(0);

        std::vector<torch::jit::IValue> input;
        input.push_back(forw_tensor);

        auto start = std::chrono::high_resolution_clock::now();
        at::Tensor output = module.forward(input).toTensor();
        auto end = std::chrono::high_resolution_clock::now();

        std::vector<int64_t> output_size = {model_h, model_w};
        at::Tensor forw_feature = torch::nn::functional::interpolate(
                output,
                torch::nn::functional::InterpolateFuncOptions().size(output_size).mode(torch::kBilinear).align_corners(false)
        );
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Function execution time: " << duration.count() << " ms" << std::endl;


        

        // if(cur_feature.defined()) {
        //     double euc_dis_before = 0.0, cos_dis_before = 0.0;
        //     for(int i = 0; i < int(forw_pts.size()); i++) {
        //         double t_euc_dis = 0.0, t_cos_dis0 = 0.0, t_cos_dis1 = 0.0, t_cos_dis2 = 0.0;
        //         for(int j = 0; j < model_c; j++) {
        //             float x = forw_feature.index({0, j, cvRound(forw_pts[i].y), cvRound(forw_pts[i].x)}).item<float>();
        //             float y = cur_feature.index({0, j, cvRound(cur_pts[i].y), cvRound(cur_pts[i].x)}).item<float>();
        //             t_euc_dis += (x-y) * (x-y);
        //             t_cos_dis0 += x*y;
        //             t_cos_dis1 += x*x;
        //             t_cos_dis2 += y*y;
        //         }
        //         euc_dis_before += std::sqrt(t_euc_dis);
        //         cos_dis_before += t_cos_dis0 / std::sqrt(t_cos_dis1) / std::sqrt(t_cos_dis2);
        //     }
        //     euc_dis_before /= forw_pts.size() + 0.0;
        //     cos_dis_before /= forw_pts.size() + 0.0;
        //     ROS_WARN("euc_dis_before: %lf cos_dis_before: %lf\n", euc_dis_before, cos_dis_before);

        // }



        start = std::chrono::high_resolution_clock::now();
        rejectWithF();
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        std::cout << "ransac time: " << duration.count() << " ms" << std::endl;


        // if(cur_feature.defined()) {
        //     double euc_dis_after = 0.0, cos_dis_after = 0.0;
        //     for(int i = 0; i < int(forw_pts.size()); i++) {
        //         double t_euc_dis = 0.0, t_cos_dis0 = 0.0, t_cos_dis1 = 0.0, t_cos_dis2 = 0.0;
        //         for(int j = 0; j < model_c; j++) {
        //             float x = forw_feature.index({0, j, cvRound(forw_pts[i].y), cvRound(forw_pts[i].x)}).item<float>();
        //             float y = cur_feature.index({0, j, cvRound(cur_pts[i].y), cvRound(cur_pts[i].x)}).item<float>();
        //             t_euc_dis += (x-y) * (x-y);
        //             t_cos_dis0 += x*y;
        //             t_cos_dis1 += x*x;
        //             t_cos_dis2 += y*y;
        //         }
        //         euc_dis_after += std::sqrt(t_euc_dis);
        //         cos_dis_after += t_cos_dis0 / std::sqrt(t_cos_dis1) / std::sqrt(t_cos_dis2);
        //     }
        //     euc_dis_after /= forw_pts.size() + 0.0;
        //     cos_dis_after /= forw_pts.size() + 0.0;
        //     ROS_WARN("euc_dis_after: %lf cos_dis_after: %lf\n", euc_dis_after, cos_dis_after);

        // }

        cur_feature = forw_feature;


        std::string cur_label_name = intToStringWithLeadingZeros(cur_seq) + ".npy";
        std::string forw_label_name = intToStringWithLeadingZeros(forw_seq) + ".npy";

        cnpy::NpyArray cur_label = cnpy::npy_load("/media/nyamori/8856D74A56D73820/vslam/dataset/kitti/2011_09_30/04_npy/" + cur_label_name);
        cnpy::NpyArray forw_label = cnpy::npy_load("/media/nyamori/8856D74A56D73820/vslam/dataset/kitti/2011_09_30/04_npy/" + forw_label_name);
        int* cur_label_data = cur_label.data<int>();
        int* forw_label_data = forw_label.data<int>();
        std::vector<size_t> shape = cur_label.shape;
        vector<uchar> status;
        int del_bylabel = 0, before_label = 0;

        for (int i = 0; i < int(forw_pts.size()); i++) {
            ++before_label;
            status.push_back(0);
            for(int tx = -1; tx <= 1; ++tx) for(int ty = -1; ty <= 1; ++ty)
            if(abs(tx) + abs(ty) <= 1) {
                cv::Point2f forw_pts_new(forw_pts[i].x+tx, forw_pts[i].y+ty);
                cv::Point2f cur_pts_new(cur_pts[i].x+tx, cur_pts[i].y+ty);
                if(!inBorder(forw_pts_new) || !inBorder(cur_pts_new)) continue;
                if (forw_label_data[getLabelId(forw_pts_new, shape[1])] == cur_label_data[getLabelId(cur_pts_new, shape[1])]) {
                    status[i] = 1;
                }
            }
            del_bylabel += (1-status[i]);
        }
        ROS_WARN("forw = %d cur = %d (%d/%d)\n", forw_seq, cur_seq, del_bylabel, before_label);

        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_seq = cur_seq;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_seq = forw_seq;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_WARN("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
