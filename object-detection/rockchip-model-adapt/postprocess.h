#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#include <stdint.h>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM     80
#define NMS_THRESH        0.6
#define BOX_THRESH        0.5
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

typedef struct _prob_with_idx_t {
	float score;
	int index;
} prob_with_idx_t;

int post_process(uint8_t *input0, uint8_t *input1, uint8_t *input2, 
                 int model_in_h, int model_in_w, float conf_threshold, float nms_threshold, 
                 std::vector<uint32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

#endif //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
