
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
#include <message_filters/sync_policies/approximate_time.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <cv_bridge/cv_bridge.h>


#include "stereo_msgs/DisparityImage.h"
#include "sensor_msgs/Image.h"

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
					m.x = xyz_ptr[i_3+2];
					m.y = -xyz_ptr[i_3];
					m.z = -xyz_ptr[i_3+1];
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
        sensor_msgs::Image, sensor_msgs::Image
        > StereoSyncPolicy;
    typedef message_filters::Synchronizer<StereoSyncPolicy> ApproximateSync;

    protected:
        Mat img_l, img_r;
        Mat disp, raw_disp;
        FilteredSGBM block_matcher; // disparity matching
        //Rectifier rectifier;

        ros::NodeHandle nh;
        image_transport::ImageTransport it;
        boost::shared_ptr<ApproximateSync> sync;

        image_transport::SubscriberFilter img_l_sub;
        image_transport::SubscriberFilter img_r_sub;

        image_transport::Publisher disp_pub;
        image_transport::Publisher pcl_pub;

    public:
        StereoFilter():
            nh(),
            it(nh)
        {
            ROS_INFO("Stereo Filter Starting Up");
            disp_pub = it.advertise("disparity", 1);
            image_transport::TransportHints hints("raw", ros::TransportHints(), nh);
            img_l_sub.subscribe(it, "left", 1, hints);
            img_r_sub.subscribe(it, "right", 1, hints);
            sync.reset(new ApproximateSync(StereoSyncPolicy(1),
                        img_l_sub, img_r_sub));
            sync->registerCallback(
                    boost::bind(&StereoFilter::callback, this, _1, _2));
        }
        void callback(
                const sensor_msgs::ImageConstPtr& left_msg,
                const sensor_msgs::ImageConstPtr& right_msg){
            cv_bridge::CvImagePtr cv_l_ptr, cv_r_ptr, cv_d_ptr;

            cv_l_ptr = cv_bridge::toCvCopy(left_msg, sensor_msgs::image_encodings::BGR8);
            cv_r_ptr = cv_bridge::toCvCopy(right_msg, sensor_msgs::image_encodings::BGR8);

            //here, assume rectified
            cv::Rect roi;
            block_matcher.compute(cv_l_ptr->image, cv_r_ptr->image, disp, &raw_disp, &roi);

            // TODO : better disparity, etc.
            // max disparity 
            double min, max;
            cv::minMaxLoc(disp, &min, &max);
            disp.convertTo(disp, CV_8UC1, 255.0/(max-min), -min);
            disp = disp(roi);

            // output ...
            cv_bridge::CvImage disp_msg;
            disp_msg.header = left_msg->header;
            disp_msg.encoding = sensor_msgs::image_encodings::MONO8;
            disp_msg.image = disp;
            disp_pub.publish(disp_msg.toImageMsg());

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
