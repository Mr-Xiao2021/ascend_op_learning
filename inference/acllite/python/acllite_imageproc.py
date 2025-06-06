"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2021-01-20 20:12:13
MODIFIED: 2021-01-29 14:04:45
"""
import numpy as np
import acl
import acllite_utils as utils
from acllite_image import AclLiteImage
from acllite_logger import log_error, log_info
from acllite_resource import resource_list
import constants as constants

class AclLiteImageProc(object):
    """
    dvpp class
    """

    def __init__(self, acl_resource=None):
        if acl_resource is None:
            self._stream, ret = acl.rt.create_stream()
            utils.check_ret("acl.rt.create_stream", ret)
            self._run_mode, ret = acl.rt.get_run_mode()
            utils.check_ret("acl.rt.get_run_mode", ret)
        else:
            self._stream = acl_resource.stream
            self._run_mode = acl_resource.run_mode
        self._dvpp_channel_desc = None
        self._crop_config = None
        self._paste_config = None

        self._init_resource()

        # AclLiteImageProc involves acl resources, which need to be released \
        # before the acl ends when the program exits, \
        # register here to the resource table to ensure the release timing
        self._is_destroyed = False
        resource_list.register(self)

    def _init_resource(self):
        # Create dvpp channel
        self._dvpp_channel_desc = acl.media.dvpp_create_channel_desc()
        ret = acl.media.dvpp_create_channel(self._dvpp_channel_desc)
        utils.check_ret("acl.media.dvpp_create_channel", ret)

        # Create a resize configuration
        self._resize_config = acl.media.dvpp_create_resize_config()

        # Create yuv to jpeg configuration
        self._jpege_config = acl.media.dvpp_create_jpege_config()
        ret = acl.media.dvpp_set_jpege_config_level(self._jpege_config, 100)
        utils.check_ret("acl.media.dvpp_set_jpege_config_level", ret)

    def _gen_input_pic_desc(self, image,
                            width_align_factor=16, height_align_factor=2):
        image.width = utils.align_up2(image.width)
        image.height = utils.align_up2(image.height)
        if image.alignWidth == 0 or image.alignHeight == 0:
            image.alignWidth = utils.align_up(image.width, width_align_factor)
            image.alignHeight = utils.align_up(image.height, height_align_factor)
        image.size = utils.yuv420sp_size(image.alignWidth, image.alignHeight)

        pic_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(pic_desc, image.data())
        acl.media.dvpp_set_pic_desc_format(
            pic_desc, constants.PIXEL_FORMAT_YUV_SEMIPLANAR_420)
        acl.media.dvpp_set_pic_desc_width(pic_desc, image.width)
        acl.media.dvpp_set_pic_desc_height(pic_desc, image.height)
        acl.media.dvpp_set_pic_desc_width_stride(pic_desc, image.alignWidth)
        acl.media.dvpp_set_pic_desc_height_stride(pic_desc, image.alignHeight)
        acl.media.dvpp_set_pic_desc_size(pic_desc, image.size)

        return pic_desc

    def _gen_output_pic_desc(self, width, height,
                             output_buffer, output_buffer_size,
                             width_align_factor=16, height_align_factor=2):
        # Create output image
        stride_width = utils.align_up(width, width_align_factor)
        stride_height = utils.align_up(height, height_align_factor)

        pic_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(pic_desc, output_buffer)
        acl.media.dvpp_set_pic_desc_format(
            pic_desc, constants.PIXEL_FORMAT_YUV_SEMIPLANAR_420)
        acl.media.dvpp_set_pic_desc_width(pic_desc, width)
        acl.media.dvpp_set_pic_desc_height(pic_desc, height)
        acl.media.dvpp_set_pic_desc_width_stride(pic_desc, stride_width)
        acl.media.dvpp_set_pic_desc_height_stride(pic_desc, stride_height)
        acl.media.dvpp_set_pic_desc_size(pic_desc, output_buffer_size)

        return pic_desc

    def _stride_yuv_size(self, width, height,
                         width_align_factor=16, height_align_factor=2):
        stride_width = utils.align_up(width, width_align_factor)
        stride_height = utils.align_up(height, height_align_factor)
        stride_size = utils.yuv420sp_size(stride_width, stride_height)

        return stride_width, stride_height, stride_size

    def jpegd(self, image):
        """
        jepg image to yuv image
        """
        image.width = utils.align_up2(image.width)
        image.height = utils.align_up2(image.height)
        soc_version = acl.get_soc_name()
        exist_Ascend310Bx = soc_version.find("Ascend310B")
        if soc_version == "Ascend310P3" or exist_Ascend310Bx > -1:
            stride_width = utils.align_up64(image.width)
            stride_height = utils.align_up16(image.height)
            stride_size = utils.yuv420sp_size(stride_width, stride_height)
        else:
            stride_width = utils.align_up128(image.width)
            stride_height = utils.align_up16(image.height)
            stride_size = utils.yuv420sp_size(stride_width, stride_height)
        # Create conversion output image desc
        output_desc, out_buffer = self._gen_jpegd_out_pic_desc(image, stride_size)
        ret = acl.media.dvpp_jpeg_decode_async(self._dvpp_channel_desc,
                                               image.data(),
                                               image.size,
                                               output_desc,
                                               self._stream)
        if ret != constants.ACL_SUCCESS:
            log_error("dvpp_jpeg_decode_async failed ret={}".format(ret))
            return None
        ret = acl.media.dvpp_destroy_pic_desc(output_desc)
        if ret != constants.ACL_SUCCESS:
            log_error("dvpp_destroy_pic_desc failed ret={}".format(ret))
            return None
        ret = acl.rt.synchronize_stream(self._stream)
        if ret != constants.ACL_SUCCESS:
            log_error("dvpp_jpeg_decode_async failed ret={}".format(ret))
            return None

        # Return the decoded AclLiteImage instance
        return AclLiteImage(out_buffer, image.width, image.height, stride_width,
                        stride_height, stride_size, constants.MEMORY_DVPP)

    def _gen_jpegd_out_pic_desc(self, image, stride_size):
        # Predict the memory size required to decode jpeg into yuv pictures
        ret, out_buffer_size = self._get_jpegd_memory_size(image, stride_size)
        if not ret:
            return None
        # Apply for memory for storing decoded yuv pictures
        out_buffer, ret = acl.media.dvpp_malloc(out_buffer_size)
        if ret != constants.ACL_SUCCESS:
            log_error("AclLiteImageProc malloc failed, error: ", ret)
            return None
        
        soc_version = acl.get_soc_name()
        if soc_version == "Ascend310P3" or soc_version == "Ascend310B1" :
            width_align_factor = 64
            height_align_factor = 16
        else:
            width_align_factor = 128
            height_align_factor = 16
        # Create output image desc
        pic_desc = self._gen_output_pic_desc(
            image.width,
            image.height,
            out_buffer,
            out_buffer_size,
            width_align_factor,
            height_align_factor)
        return pic_desc, out_buffer

    def _get_jpegd_memory_size(self, image, stride_size):
        if image.is_local():
            size, ret = acl.media.dvpp_jpeg_predict_dec_size(
                image.data(), image.size, constants.PIXEL_FORMAT_YUV_SEMIPLANAR_420)
            if ret != constants.ACL_SUCCESS:
                log_error("Predict jpeg decode size failed, return ", ret)
                return False, 0
            return True, size
        else:
            return True, int(stride_size)

    def resize(self, image, resize_width, resize_height):
        """
        Scale yuvsp420 picture to specified size
        """
        resize_width = utils.align_up2(resize_width)
        resize_height = utils.align_up2(resize_height)
        soc_version = acl.get_soc_name()
        if soc_version == "Ascend310B1" :
            width_align_factor = 2
            height_align_factor = 2
            stride_width = resize_width
            stride_height = utils.align_up2(resize_height)
        else:
            width_align_factor = 16
            height_align_factor = 2
            stride_width = utils.align_up16(resize_width)
            stride_height = utils.align_up2(resize_height)
        # Generate input picture desc
        input_desc = self._gen_input_pic_desc(image, width_align_factor, height_align_factor)
        # Calculate the image size after scaling
        output_size = utils.yuv420sp_size(stride_width, stride_height)
        # Request memory for the zoomed picture
        out_buffer, ret = acl.media.dvpp_malloc(output_size)
        if ret != constants.ACL_SUCCESS:
            log_error("AclLiteImageProc malloc failed, error: ", ret)
            return None
        # Create output image
        output_desc = self._gen_output_pic_desc(resize_width, resize_height,
                                                out_buffer, output_size,
                                                width_align_factor, height_align_factor)
        if output_desc is None:
            log_error("Gen resize output desc failed")
            return None
        # Call dvpp asynchronous zoom interface to zoom pictures
        ret = acl.media.dvpp_vpc_resize_async(self._dvpp_channel_desc,
                                              input_desc,
                                              output_desc,
                                              self._resize_config,
                                              self._stream)
        if ret != constants.ACL_SUCCESS:
            log_error("Vpc resize async failed, error: ", ret)
            return None
        # Wait for the zoom operation to complete
        ret = acl.rt.synchronize_stream(self._stream)
        if ret != constants.ACL_SUCCESS:
            log_error("Resize synchronize stream failed, error: ", ret)
            return None
        # Release the resources requested for scaling
        acl.media.dvpp_destroy_pic_desc(input_desc)
        acl.media.dvpp_destroy_pic_desc(output_desc)
        return AclLiteImage(out_buffer, resize_width, resize_height, stride_width,
                        stride_height, output_size, constants.MEMORY_DVPP)

    def _gen_resize_out_pic_desc(self, resize_width,
                                 resize_height, output_size):
        out_buffer, ret = acl.media.dvpp_malloc(output_size)
        if ret != constants.ACL_SUCCESS:
            log_error("AclLiteImageProc malloc failed, error: ", ret)
            return None
        pic_desc = self._gen_output_pic_desc(resize_width, resize_height,
                                             out_buffer, output_size)
        return pic_desc, out_buffer

    def crop_and_paste(
            self,
            image,
            width,
            height,
            crop_and_paste_width,
            crop_and_paste_height):
        """
        crop_and_paste
        """
        log_info('AclLiteImageProc vpc crop and paste stage:')
        crop_and_paste_width = utils.align_up2(crop_and_paste_width)
        crop_and_paste_height = utils.align_up2(crop_and_paste_height)
        soc_version = acl.get_soc_name()
        if soc_version == "Ascend310B1" :
            width_align_factor = 2
            height_align_factor = 2
            stride_width = crop_and_paste_width
            stride_height = utils.align_up2(crop_and_paste_height)
        else:
            width_align_factor = 16
            height_align_factor = 2
            stride_width = utils.align_up16(crop_and_paste_width)
            stride_height = utils.align_up2(crop_and_paste_height)
        input_desc = self._gen_input_pic_desc(image, width_align_factor, height_align_factor)
        out_buffer_size = utils.yuv420sp_size(stride_width, stride_height)
        out_buffer, ret = acl.media.dvpp_malloc(out_buffer_size)
        output_desc = self._gen_output_pic_desc(
            crop_and_paste_width,
            crop_and_paste_height,
            out_buffer,
            out_buffer_size,
            width_align_factor,
            height_align_factor)
        self._crop_config = acl.media.dvpp_create_roi_config(
            0, (width >> 1 << 1) - 1, 0, (height >> 1 << 1) - 1)
        # set crop area:
        rx = float(width) / float(crop_and_paste_width)
        ry = float(height) / float(crop_and_paste_height)
        if rx > ry:
            dx = 0
            r = rx
            dy = int((crop_and_paste_height - height / r) / 2)
        else:
            dy = 0
            r = ry
            dx = int((crop_and_paste_width - width / r) / 2)
        pasteRightOffset = int(crop_and_paste_width - 2 * dx)
        pasteBottomOffset = int(crop_and_paste_height - 2 * dy)
        if (pasteRightOffset % 2) == 0:
            pasteRightOffset = pasteRightOffset - 1
        if (pasteBottomOffset % 2) == 0:
            pasteBottomOffset = pasteBottomOffset - 1
        self._paste_config = acl.media.dvpp_create_roi_config(
            0, pasteRightOffset, 0, pasteBottomOffset)
        ret = acl.media.dvpp_vpc_crop_and_paste_async(self._dvpp_channel_desc,
                                                      input_desc,
                                                      output_desc,
                                                      self._crop_config,
                                                      self._paste_config,
                                                      self._stream)
        utils.check_ret("acl.media.dvpp_vpc_crop_and_paste_async", ret)
        ret = acl.rt.synchronize_stream(self._stream)
        utils.check_ret("acl.rt.synchronize_stream", ret)
        log_info('AclLiteImageProc vpc crop and paste stage success')
        stride_width = crop_and_paste_width - 2 * dx
        stride_height = crop_and_paste_height - 2 * dy
        acl.media.dvpp_destroy_pic_desc(input_desc)
        acl.media.dvpp_destroy_pic_desc(output_desc)

        return AclLiteImage(out_buffer, image.width, image.height, stride_width,
                        stride_height, out_buffer_size, constants.MEMORY_DVPP)

    def crop_and_paste_get_roi(
            self,
            image,
            width,
            height,
            crop_and_paste_width,
            crop_and_paste_height):
        """
        :image: input image
        :width: input image width
        :height: input image height
        :crop_and_paste_width: crop_and_paste_width
        :crop_and_paste_height: crop_and_paste_height
        :return: return AclLiteImage
        """
        log_info('AclLiteImageProc vpc crop and paste stage:')
        crop_and_paste_width = utils.align_up2(crop_and_paste_width)
        crop_and_paste_height = utils.align_up2(crop_and_paste_height)
        soc_version = acl.get_soc_name()
        if soc_version == "Ascend310B1" :
            width_align_factor = 2
            height_align_factor = 2
            stride_width = crop_and_paste_width
            stride_height = utils.align_up2(crop_and_paste_height)
        else:
            width_align_factor = 16
            height_align_factor = 2
            stride_width = utils.align_up16(crop_and_paste_width)
            stride_height = utils.align_up2(crop_and_paste_height)
        input_desc = self._gen_input_pic_desc(image, width_align_factor, height_align_factor)
        out_buffer_size = utils.yuv420sp_size(stride_width, stride_height)
        out_buffer, ret = acl.media.dvpp_malloc(out_buffer_size)
        output_desc = self._gen_output_pic_desc(
            crop_and_paste_width,
            crop_and_paste_height,
            out_buffer,
            out_buffer_size,
            width_align_factor,
            height_align_factor)
        self._crop_config = acl.media.dvpp_create_roi_config(
            0, (width >> 1 << 1) - 1, 0, (height >> 1 << 1) - 1)
        self._paste_config = acl.media.dvpp_create_roi_config(
            0, crop_and_paste_width - 1, 0, crop_and_paste_height - 1)
        ret = acl.media.dvpp_vpc_crop_and_paste_async(self._dvpp_channel_desc,
                                                      input_desc,
                                                      output_desc,
                                                      self._crop_config,
                                                      self._paste_config,
                                                      self._stream)
        utils.check_ret("acl.media.dvpp_vpc_crop_and_paste_async", ret)
        ret = acl.rt.synchronize_stream(self._stream)
        utils.check_ret("acl.rt.synchronize_stream", ret)
        log_info('AclLiteImageProc vpc crop and paste stage success')
        stride_width = utils.align_up16(crop_and_paste_width)
        stride_height = utils.align_up2(crop_and_paste_height)
        acl.media.dvpp_destroy_pic_desc(input_desc)
        acl.media.dvpp_destroy_pic_desc(output_desc)
        return AclLiteImage(out_buffer, image.width, image.height, stride_width,
                        stride_height, out_buffer_size, constants.MEMORY_DVPP)

    def jpege(self, image):
        """
        Convert yuv420sp pictures to jpeg pictures
        """
        # create input image
        input_desc = self._gen_input_pic_desc(image)
        # Predict the memory size required for conversion
        output_size, ret = acl.media.dvpp_jpeg_predict_enc_size(
            input_desc, self._jpege_config)
        if (ret != constants.ACL_SUCCESS):
            log_error("Predict jpege output size failed")
            return None
        # Request memory required for conversion
        output_buffer, ret = acl.media.dvpp_malloc(output_size)
        if (ret != constants.ACL_SUCCESS):
            log_error("Malloc jpege output memory failed")
            return None
        output_size_array = np.array([output_size], dtype=np.int32)
        if "bytes_to_ptr" in dir(acl.util):
            bytes_data = output_size_array.tobytes()
            output_size_ptr = acl.util.bytes_to_ptr(bytes_data)
        else:
            output_size_ptr = acl.util.numpy_to_ptr(output_size_array)

        # Call jpege asynchronous interface to convert pictures
        ret = acl.media.dvpp_jpeg_encode_async(self._dvpp_channel_desc,
                                               input_desc, output_buffer,
                                               output_size_ptr,
                                               self._jpege_config,
                                               self._stream)
        if (ret != constants.ACL_SUCCESS):
            log_error("Jpege failed, ret ", ret)
            return None
        # Wait for the conversion to complete
        ret = acl.rt.synchronize_stream(self._stream)
        if (ret != constants.ACL_SUCCESS):
            log_error("Jpege synchronize stream, failed, ret ", ret)
            return None
        # Release resources
        acl.media.dvpp_destroy_pic_desc(input_desc)
        if "bytes_to_ptr" in dir(acl.util):
            output_size_array=np.frombuffer(bytes_data,dtype=output_size_array.dtype).reshape(output_size_array.shape)
        return AclLiteImage(
            output_buffer, image.width, image.height, 0, 0, int(
                output_size_array[0]), constants.MEMORY_DVPP)

    def destroy(self):
        """
        dvpp resource release
        """
        if self._is_destroyed:
            return

        if self._resize_config:
            acl.media.dvpp_destroy_resize_config(self._resize_config)

        if self._dvpp_channel_desc:
            acl.media.dvpp_destroy_channel(self._dvpp_channel_desc)
            acl.media.dvpp_destroy_channel_desc(self._dvpp_channel_desc)

        if self._jpege_config:
            acl.media.dvpp_destroy_jpege_config(self._jpege_config)
        self._is_destroyed = True
        resource_list.unregister(self)
        log_info("dvpp resource release success")

    def __del__(self):
        self.destroy()

