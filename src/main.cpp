
/* STD */
#include <iostream>
#include <string>
#include <vector>
#include <functional>

/* OpenCV */
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"

/* Custom OpenCV */
#include "stereo_filter/rectifier.h"
#include "stereo_filter/f_sgbm.h"

/* ROS */
#include "ros/ros.h"
#include "ros/package.h"

#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <image_geometry/pinhole_camera_model.h>

#include <cv_bridge/cv_bridge.h>


#include "stereo_msgs/DisparityImage.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"

#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/PointField.h"
#include "sensor_msgs/point_cloud2_iterator.h"

#include "geometry_msgs/PointStamped.h"



using namespace cv;
using namespace std;

struct pcl_layout{
	float x;
	float y;
	float z;
	union{
		float rgb;
		struct{ //endian-ness
			uint8_t b;
			uint8_t g;
			uint8_t r;
		};
	};
};

class Parallel_cvImg2ROSPCL: public cv::ParallelLoopBody{
	private:
		const Mat& xyz;
		const Mat& bgr;
		sensor_msgs::PointCloud2& msg;
	public:
		Parallel_cvImg2ROSPCL(const Mat& xyz, const Mat& bgr, sensor_msgs::PointCloud2& msg):
			xyz(xyz),bgr(bgr),msg(msg){}
		virtual void operator()(const cv::Range &r) const {
			pcl_layout* msg_ptr = (pcl_layout*) msg.data.data();
			const uint8_t* bgr_ptr = (const uint8_t*) bgr.data;
			const float* xyz_ptr = (const float*) xyz.data;

			float bad = std::numeric_limits<float>::quiet_NaN();

			for(int i = r.start; i < r.end; ++i){ // essentially index of point
				int i_3 = i*3;
				float z = xyz_ptr[i_3+2];
				auto& m= msg_ptr[i];
				if (z <= 10 && !std::isinf(z)){ // 10000 = MISSING_Z
					m.x = xyz_ptr[i_3];
					m.y = xyz_ptr[i_3+1];
					m.z = xyz_ptr[i_3+2];
					m.b = bgr_ptr[i_3];
					m.g = bgr_ptr[i_3+1];;
					m.r = bgr_ptr[i_3+2];;
				}else{
					m.x = m.y = m.z = bad;
				}
			}
		}

};

#define INIT_PFIELD(n, o, t) \
	sensor_msgs::PointField n; \
n.datatype = n.t; \
n.offset = o; \
n.count = 1; \
n.name = #n;

void format_pointfields(std::vector<sensor_msgs::PointField>& fields){
	fields.clear();

	//sensor_msgs::PointField x,y,z,b,g,r;
	/* Instantiate Fields */
	INIT_PFIELD(x,0,FLOAT32);
	INIT_PFIELD(y,4,FLOAT32);
	INIT_PFIELD(z,8,FLOAT32);

	INIT_PFIELD(rgb,12,FLOAT32);
	//INIT_PFIELD(a,15,UINT8);

	fields.push_back(x);
	fields.push_back(y);
	fields.push_back(z);
	fields.push_back(rgb);
}

void dist2pcl(Mat& dist, Mat& frame, sensor_msgs::PointCloud2& msg){
	//dist = (row,col,(x,y,z))
	msg.header.stamp = ros::Time::now();
	msg.height = dist.rows;
	msg.width  = dist.cols;

	format_pointfields(msg.fields);

	msg.is_bigendian = false;
	msg.point_step = sizeof(pcl_layout);
	msg.row_step = msg.width * sizeof(pcl_layout);

	int n = dist.rows * dist.cols;
	msg.data.resize(n*sizeof(pcl_layout));

	msg.is_dense = false; // there may be invalid points	

	parallel_for_(Range{0,dist.rows * dist.cols}, Parallel_cvImg2ROSPCL(dist, frame, msg), 8);
}

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch ( depth ) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
}

class StereoFilter{
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::CameraInfo,
		sensor_msgs::Image, sensor_msgs::CameraInfo
        > StereoSyncPolicy;
    typedef message_filters::Synchronizer<StereoSyncPolicy> ApproximateSync;

    protected:
        Mat img_l, img_r;
        Mat img_l_g, img_r_g;
        Mat disp, dist, raw_disp;
        Mat conf;

		float baseline;
		bool override_baseline;
        bool _rectify;

        FilteredSGBM block_matcher; // disparity matching

        image_geometry::PinholeCameraModel model;
		boost::shared_ptr<Rectifier> rectifier;

        ros::NodeHandle nh;
        image_transport::ImageTransport it;
        boost::shared_ptr<ApproximateSync> sync;

        image_transport::SubscriberFilter img_l_sub, img_r_sub;
		message_filters::Subscriber<sensor_msgs::CameraInfo> img_l_info_sub, img_r_info_sub;

        image_transport::Publisher disp_pub;
        ros::Publisher pcl_pub;

    public:
        StereoFilter():
            nh(),
            it(nh)
        {
            ROS_INFO("Stereo Filter Starting Up");
            disp_pub = it.advertise("disparity", 1);
            image_transport::TransportHints hints("raw", ros::TransportHints(), nh);

			ros::param::get("~baseline", baseline);
			ros::param::get("~override_baseline", override_baseline);
            ros::param::get("~rectify", _rectify);

            img_l_sub.subscribe(it, "left", 1, hints);
            img_r_sub.subscribe(it, "right", 1, hints);
			img_l_info_sub.subscribe(nh, "left_info", 1);
			img_r_info_sub.subscribe(nh, "right_info", 1);
			pcl_pub=nh.advertise<sensor_msgs::PointCloud2>("pcl", 1);

            sync.reset(new ApproximateSync(StereoSyncPolicy(1),
                        img_l_sub, img_l_info_sub, img_r_sub, img_r_info_sub));
            sync->registerCallback(
                    boost::bind(&StereoFilter::callback, this, _1, _2, _3, _4));
        }
        void callback(
                const sensor_msgs::ImageConstPtr& left_msg,
				const sensor_msgs::CameraInfoConstPtr& left_info_msg,
                const sensor_msgs::ImageConstPtr& right_msg,
				const sensor_msgs::CameraInfoConstPtr& right_info_msg
				){

            //ROS_INFO_THROTTLE(1.0, "_rectify : %s", (_rectify? "True" : "False"));

			if(!rectifier){

				cv::Mat m_l(3,3, CV_64FC1, (double*)left_info_msg->K.data()),
						d_l(1,5, CV_64FC1, (double*)left_info_msg->D.data()), 
						r_l(3,3, CV_64FC1, (double*)left_info_msg->R.data()),
						p_l(3,4, CV_64FC1, (double*)left_info_msg->P.data()),
						m_r(3,3, CV_64FC1, (double*)right_info_msg->K.data()),
						d_r(1,5, CV_64FC1, (double*)right_info_msg->D.data()), 
						r_r(3,3, CV_64FC1, (double*)right_info_msg->R.data()),
						p_r(3,4, CV_64FC1, (double*)right_info_msg->P.data());

				if(override_baseline){
					p_r.at<double>(0, 3) = -p_r.at<double>(0,0) * baseline;
				}

				// apply corrections
				rectifier.reset(new Rectifier(m_l, d_l, r_l, p_l,
							m_r, d_r, r_r, p_r,
							left_info_msg->width,
							left_info_msg->height
							));

                model.fromCameraInfo(left_info_msg);
			}

			if(!rectifier)
				return;

            cv_bridge::CvImagePtr cv_l_ptr, cv_r_ptr, cv_d_ptr;
            cv_l_ptr = cv_bridge::toCvCopy(left_msg, sensor_msgs::image_encodings::BGR8);
            cv_r_ptr = cv_bridge::toCvCopy(right_msg, sensor_msgs::image_encodings::BGR8);

            if(_rectify){
                //rectifier->apply(cv_l_ptr->image, cv_r_ptr->image, img_l, img_r);
                model.rectifyImage(cv_l_ptr->image, img_l);
                model.rectifyImage(cv_r_ptr->image, img_r);
            }else{
                img_l = cv_l_ptr->image;
                img_r = cv_r_ptr->image;
            }
			
			//cv::GaussianBlur(img_l, img_l, cv::Size(3,3), 0.0, 0.0);
			//cv::GaussianBlur(img_r, img_r, cv::Size(3,3), 0.0, 0.0);

            //here, assume rectified
            cv::Rect roi;
            block_matcher.compute(img_l, img_r,
                    disp, &raw_disp, &roi, &conf);
            disp.setTo(FLT_MAX, conf < 128); // TODO : set as configurable parameter
			rectifier->convert(disp, dist);

            // hack specific to current configuration:
            // kill bottom few pixels cuz invalid
            // xywh
            
            // cv::Rect roi_hack(0, 0, dist.cols, 455);
            // dist = dist(roi_hack);

            // TODO : better disparity, etc.
            // max disparity 
            double min, max;
            cv::minMaxLoc(dist, &min, &max);
			//ROS_INFO("min/max %f %f", min, max);
            //disp.convertTo(disp, CV_8UC1, 255.0/(max-min), -min);
            //disp.convertTo(disp, CV_8UC1, 0.5 / 16.0, 0.0);
            //disp = disp(roi);

            // output ...
            cv_bridge::CvImage disp_msg;
            disp_msg.header = left_msg->header;
            disp_msg.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
            disp_msg.image = disp;
            disp_pub.publish(disp_msg.toImageMsg());

			sensor_msgs::PointCloud2 pcl_msg;
			pcl_msg.header.frame_id = left_msg->header.frame_id;
			dist2pcl(dist, img_l, pcl_msg);
			pcl_pub.publish(pcl_msg);

            //cv::imshow("conf", conf);
            //cv::waitKey(1);

            //stereo_msgs::DisparityImage disp_out;
            //disp_out.
        }
};

int main(int argc, char* argv[])
{
	ros::init(argc, argv, "stereo_filter");
    StereoFilter app;
    while(ros::ok()){
        ros::spin();
    }
	return 0;
}
