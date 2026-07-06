
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

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

// The seg model systematically leaves a ~5 px unlabeled sliver along frame
// borders (mask-proto downsampling + boxes rarely flush with the edge). If
// water reaches within snap_px of the left/right/bottom edge, extend it to
// the edge -- mirrors the label-side border snap in sam3_annotate.py. The
// top edge is left alone (water never ends there).
static void snap_water_to_borders(cv::Mat &mask, int snap_px) {
    int w = mask.cols, h = mask.rows;
    for (int y = 0; y < h; ++y) {
        uchar *row = mask.ptr<uchar>(y);
        for (int x = 1; x <= snap_px && x < w; ++x) {
            if (row[x]) { memset(row, 255, x); break; }
        }
        for (int x = w - 2; x >= w - 1 - snap_px && x >= 0; --x) {
            if (row[x]) { memset(row + x + 1, 255, w - 1 - x); break; }
        }
    }
    for (int x = 0; x < w; ++x) {
        for (int y = h - 2; y >= h - 1 - snap_px && y >= 0; --y) {
            if (mask.at<uchar>(y, x)) {
                for (int yy = y + 1; yy < h; ++yy) mask.at<uchar>(yy, x) = 255;
                break;
            }
        }
    }
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

    // water accumulates into one frame-sized mask (so border snapping works
    // in frame space); other classes are painted per instance afterwards
    cv::Mat water_mask = cv::Mat::zeros(image.rows, image.cols, CV_8U);
    struct Painted { cv::Mat mask; int x, y; cv::Vec3b color; };
    std::vector<Painted> others;

    int i = 0;
    for (auto &obj : objs) {
        printf("  %-6s conf=%.3f box=(%.0f,%.0f,%.0f,%.0f)\n",
               classlabels[obj.class_label], obj.confidence, obj.left, obj.top,
               obj.right, obj.bottom);
        if (!obj.seg) continue;

        int obj_width = std::max(1, (int)std::round(obj.right - obj.left));
        int obj_height = std::max(1, (int)std::round(obj.bottom - obj.top));
        cv::Mat mask(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data);
        cv::Mat resized_mask;
        cv::resize(mask, resized_mask, cv::Size(obj_width, obj_height));
        cv::imwrite(cv::format("%d_mask.jpg", i), resized_mask);
        i++;

        int x = (int)std::round(obj.left);
        int y = (int)std::round(obj.top);

        if (obj.class_label == 0) {
            // boxes from the end-to-end head may stick a few px out of the
            // image, so clamp every write into bounds
            for (int r = 0; r < resized_mask.rows; r++) {
                int py = y + r;
                if (py < 0 || py >= image.rows) continue;
                for (int c = 0; c < resized_mask.cols; c++) {
                    int px = x + c;
                    if (px < 0 || px >= image.cols) continue;
                    if (resized_mask.at<uchar>(r, c) > 128)
                        water_mask.at<uchar>(py, px) = 255;
                }
            }
        } else {
            uint8_t b, g, r;
            tie(b, g, r) = yolo::random_color(obj.class_label);
            others.push_back({resized_mask, x, y, cv::Vec3b(b, g, r)});
        }
    }

    snap_water_to_borders(water_mask, 8);

    uint8_t wb, wg, wr;
    tie(wb, wg, wr) = yolo::random_color(0);
    image.setTo(cv::Scalar(wb, wg, wr), water_mask);

    for (auto &p : others) {
        for (int r = 0; r < p.mask.rows; r++) {
            int py = p.y + r;
            if (py < 0 || py >= image.rows) continue;
            for (int c = 0; c < p.mask.cols; c++) {
                int px = p.x + c;
                if (px < 0 || px >= image.cols) continue;
                if (p.mask.at<uchar>(r, c) > 128) image.at<cv::Vec3b>(py, px) = p.color;
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