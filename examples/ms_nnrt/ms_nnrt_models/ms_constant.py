"""
Predefined code for pyACL
"""

# error code
ACL_ERROR_NONE = 0

# rule for memory copy
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3

# rule for mem
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEM_MALLOC_HUGE_ONLY = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2

# NUMPY data type
NPY_BOOL = 0
NPY_BYTE = 1
NPY_UBYTE = 2
NPY_SHORT = 3
NPY_USHORT = 4
NPY_INT = 5
NPY_UINT = 6
NPY_LONG = 7
NPY_ULONG = 8
NPY_FLOAT32 = 11

# data format
ACL_FORMAT_UNDEFINED = -1
ACL_FORMAT_NCHW = 0
ACL_FORMAT_NHWC = 1
ACL_FORMAT_ND = 2
ACL_FORMAT_NC1HWC0 = 3
ACL_FORMAT_FRACTAL_Z = 4

# data type
ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT8 = 2
ACL_INT32 = 3

# dvpp pixel format
PIXEL_FORMAT_YUV_400 = 0  # YUV400 8bit
PIXEL_FORMAT_YUV_SEMIPLANAR_420 = 1  # YUV420SP NV12 8bit
PIXEL_FORMAT_YVU_SEMIPLANAR_420 = 2  # YUV420SP NV21 8bit
PIXEL_FORMAT_YUV_SEMIPLANAR_422 = 3  # YUV422SP NV12 8bit
PIXEL_FORMAT_YVU_SEMIPLANAR_422 = 4  # YUV422SP NV21 8bit
PIXEL_FORMAT_YUV_SEMIPLANAR_444 = 5  # YUV444SP NV12 8bit
PIXEL_FORMAT_YVU_SEMIPLANAR_444 = 6  # YUV444SP NV21 8bit
PIXEL_FORMAT_YUYV_PACKED_422 = 7  # YUV422P YUYV 8bit
PIXEL_FORMAT_UYVY_PACKED_422 = 8  # YUV422P UYVY 8bit
PIXEL_FORMAT_YVYU_PACKED_422 = 9  # YUV422P YVYU 8bit
PIXEL_FORMAT_VYUY_PACKED_422 = 10  # YUV422P VYUY 8bit
PIXEL_FORMAT_YUV_PACKED_444 = 11  # YUV444P 8bit
PIXEL_FORMAT_RGB_888 = 12  # RGB888
PIXEL_FORMAT_BGR_888 = 13  # BGR888
PIXEL_FORMAT_ARGB_8888 = 14  # ARGB8888
PIXEL_FORMAT_ABGR_8888 = 15  # ABGR8888
PIXEL_FORMAT_RGBA_8888 = 16  # RGBA8888
PIXEL_FORMAT_BGRA_8888 = 17  # BGRA8888
PIXEL_FORMAT_YUV_SEMI_PLANNER_420_10BIT = 18  # YUV420SP 10bit
PIXEL_FORMAT_YVU_SEMI_PLANNER_420_10BIT = 19  # YVU420sp 10bit
PIXEL_FORMAT_YVU_PLANAR_420 = 20  # YUV420P 8bit

# images format
IMG_EXT = ['.jpg', '.JPG', '.png', '.PNG', '.bmp', '.BMP', '.jpeg', '.JPEG']