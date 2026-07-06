
#include <opencv2/opencv.hpp>

#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"

using namespace std;

// water_segment class contract (docs/ROADMAP.md) -- ids are fixed downstream
static const char *classlabels[] = {"water", "bridge", "person", "boat"};

yolo::Image cvimg(const cv::Mat &image) {
    return yolo::Image(image.data, image.cols, image.rows);
}

void single_inference(const char *engine_path, const char *image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        printf("failed to read image: %s\n", image_path);
        return;
    }
    auto yolo = yolo::load(engine_path, yolo::Type::V26Seg, 0.4f);
    if (yolo == nullptr) return;

    auto objs = yolo->forward(cvimg(image));
    int i = 0;
    for (auto &obj : objs) {
        printf("  %-6s conf=%.3f box=(%.0f,%.0f,%.0f,%.0f)\n",
               classlabels[obj.class_label], obj.confidence, obj.left, obj.top,
               obj.right, obj.bottom);
        uint8_t b, g, r;
        tie(b, g, r) = yolo::random_color(obj.class_label);
        cv::Vec3b color(b,g,r);

        auto obj_width = obj.right - obj.left;
        auto obj_height = obj.bottom - obj.top;

        // auto name = cocolabels[obj.class_label];
        // auto caption = cv::format("%s %.2f", name, obj.confidence);
        // int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        // cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
        //               cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r),
        //               -1);
        // cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1,
        //             cv::Scalar::all(0), 2, 16);

        if (obj.seg) {
            cv::Mat mask(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data);
            cv::Mat resized_mask;

            cv::resize(mask, resized_mask, cv::Size(obj_width, obj_height));
            cv::imwrite(
                cv::format("%d_mask.jpg", i),
                resized_mask);
            i++;

            // for 循环逐个绘制
            auto x = obj.left;
            auto y = obj.top;

            for (int i =0; i < resized_mask.rows; i++) {
                for (int j =0; j < resized_mask.cols; j++) {
                    if(resized_mask.at<uchar>(i,j) > 128 ) {
                        image.at<cv::Vec3b>(y+i, x+j) = color;
                    }
              }
            }
        }
    }

    printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
    cv::imwrite("Result.jpg", image);
}

int main(int argc, char **argv) {
    const char *engine_path = argc > 1 ? argv[1] : "best.engine";
    const char *image_path = argc > 2 ? argv[2] : "1720.jpg";
    single_inference(engine_path, image_path);
    return 0;
}