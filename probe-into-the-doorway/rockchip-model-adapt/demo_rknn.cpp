#include <stdio.h>
#include <stdlib.h>
#include "rknn_api.h"
#include <cstring>
#include <sys/time.h>
#include <opencv2/imgcodecs.hpp>

static void printRKNNTensor(rknn_tensor_attr *attr)
{
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d "
           "fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2],
           attr->dims[1], attr->dims[0], attr->n_elems, attr->size, 0, attr->type,
           attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{

    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}
using namespace cv;
int main(int argc, char **argv)
{
	rknn_context ctx;
	if (argc != 3)
    {
        printf("Usage: %s <rknn model> <bmp> \n", argv[0]);
        return -1;
    }
	char *model_name = NULL;
	model_name = (char *)argv[1];
	char *image_name = argv[2];
	printf("Loading model...\n");
	int model_data_size = 0;
	unsigned char *model_data = load_model(model_name, &model_data_size);
	int ret = rknn_init(&ctx, (void *)model_data, model_data_size, 0);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
	rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version,
           version.drv_version);
	
	rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input,
           io_num.n_output);
	rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                         sizeof(rknn_tensor_attr));
        printRKNNTensor(&(output_attrs[i]));
    }
	struct timeval start_time, stop_time;
	rknn_input inputs[1];
	memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = 28 * 28 * 1;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
	int32_t width, height, channel;
	Mat image = imread(image_name, 0);
	printf("cols: %d, rows: %d\n", image.cols, image.rows);
	int image_size = image.cols * image.rows;
	unsigned char* resize_buf = new unsigned char[image_size];
	int a = 0;
	for (int i = 0;i<image.rows;i++)
	{
		for (int j = 0;j<image.cols;j++)
		{
			 resize_buf[a]= image.at<uchar>(i, j);
			 a++;
		}
	}
	inputs[0].buf = resize_buf;
	//printf("inputvalue: [");
	//for (int i = 0; i < inputs[0].size; i++) {
	//	printf("%d ", ((char *)resize_buf)[i]);
	//}
	//printf("]\n");
	gettimeofday(&start_time, NULL);
	rknn_inputs_set(ctx, io_num.n_input, inputs);
	
	rknn_output outputs[io_num.n_output];
	memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 1;
    }
	ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    gettimeofday(&stop_time, NULL);
	delete []resize_buf;
    printf("once run use %f ms\n", (stop_time.tv_sec - start_time.tv_sec) * 1000.0 + (stop_time.tv_usec - start_time.tv_usec) / 1000.0);
	printf("out want_float: %d, index: %d, size: %d\n", outputs[0].want_float, outputs[0].index, outputs[0].size);
	printf("outvalue(%d): ", sizeof(float));
	int result = 0;
	float max = FLT_MIN;
	for (int i = 0; i < outputs[0].size; i += 4) {
		float outvalue = *((float *)(((char *)outputs[0].buf + i)));
		if (outvalue > max) {
			max = outvalue;
			result = i / 4;
		}
		printf("%f ", outvalue);
	}
	printf("\n");
	printf("result number: %d\n", result);
	
	ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
	ret = rknn_destroy(ctx);
	return 0;
}