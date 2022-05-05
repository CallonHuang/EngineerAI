// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <vector>
#include <set>
#include "postprocess.h"
#include <stdint.h>
#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"


static const int strides[] = {8, 16, 32};
static const int anchors[] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
static const char *labels[] = {
	"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light", 
	"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
	"elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
	"skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
	"tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
	"orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
	"dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
	"toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<prob_with_idx_t> &objProbs, int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i)
    {
		int n = objProbs[i].index;
        if (n == -1 || classIds[i] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = objProbs[j].index;
            if (m == -1 || classIds[i] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                objProbs[j].index = -1;
            }
        }
    }
    return 0;
}

static float sigmoid(float x)
{
    return 1.0 / (1.0 + expf(-x));
}

static float unsigmoid(float y)
{
    return -1.0 * logf((1.0 / y) - 1.0);
}

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static uint8_t qnt_f32_to_affine(float f32, uint32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

static float deqnt_affine_to_f32(uint8_t qnt, uint32_t zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

static int process(uint8_t *input, int *anchor, int index, int height, int width,
                   std::vector<float> &boxes, std::vector<prob_with_idx_t> &objProbs, std::vector<int> &classId,
                   float threshold, uint32_t zp, float scale)
{

    int validCount = 0;
	int grid_h = height / strides[index];
	int grid_w = width / strides[index];
    int grid_len = grid_h * grid_w;
    float thres = unsigmoid(threshold);
    uint8_t thres_u8 = qnt_f32_to_affine(thres, zp, scale);
    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                uint8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_u8)
                {
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    uint8_t *in_ptr = input + offset;
                    float box_x = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)strides[index];
                    box_y = (box_y + i) * (float)strides[index];
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);
                    boxes.push_back(box_x);
                    boxes.push_back(box_y);
                    boxes.push_back(box_w);
                    boxes.push_back(box_h);

                    uint8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k)
                    {
                        uint8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
					prob_with_idx_t p;
					p.score = sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale));
					p.index = validCount;
                    objProbs.push_back(p);
                    classId.push_back(maxClassId);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

static int cmp(const void *x, const void *y)
{
	return ((prob_with_idx_t *)x)->score > ((prob_with_idx_t *)y)->score ? 1 : 0;
}

int post_process(uint8_t *input0, uint8_t *input1, uint8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 std::vector<uint32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group)
{
    memset(group, 0, sizeof(detect_result_group_t));

    std::vector<float> filterBoxes;
    std::vector<prob_with_idx_t> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    uint8_t *input[] = {input0, input1, input2};
    for (int i = 0; i < 3; i++)
        validCount += process(input[i], (int *)&anchors[i * 6], i, model_in_h, model_in_w,
                          filterBoxes, objProbs, classId, conf_threshold, qnt_zps[i], qnt_scales[i]);
    // no object detect
    if (validCount <= 0)
    {
        return 0;
    }

	qsort(&objProbs[0], objProbs.size(), sizeof(prob_with_idx_t), cmp);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set)
    {
        nms(validCount, filterBoxes, classId, objProbs, c, nms_threshold);
    }

    int last_count = 0;
    group->count = 0;
    /* box valid detect target */
    for (int i = 0; i < validCount; ++i)
    {

        if (objProbs[i].index == -1 || i >= OBJ_NUMB_MAX_SIZE)
        {
            continue;
        }
        int n = objProbs[i].index;

        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i].score;

        group->results[last_count].box.left = (int)(x1 / scale_w);
        group->results[last_count].box.top = (int)(y1 / scale_h);
        group->results[last_count].box.right = (int)(x2 / scale_w);
        group->results[last_count].box.bottom = (int)(y2 / scale_h);
        group->results[last_count].prop = obj_conf;
        strncpy(group->results[last_count].name, labels[id], OBJ_NAME_MAX_SIZE);

        // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left, group->results[last_count].box.top,
        //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
        last_count++;
    }
    group->count = last_count;

    return 0;
}
