import argparse
import re
from collections import namedtuple
from pathlib import Path
import math
import random
import subprocess
from dataclasses import dataclass
from typing import List, Any
# author: Jonathan Ta 8/3/2020


###EBML

# ebml id : (element name, level, type, dict)
EBML = namedtuple('EBML', 'name level data_type child')
EBML.__new__.__defaults__ = ("", 0, "Master", None)

# ebml struct 
# byte level data of the ebml element
EBML_Data = namedtuple('EBML_Data', 'position name name_bytes size size_bytes data_position data_bytes')  


# ebml stack item
class EBML_Record:
    def __init__(self, position=0, name=None, record_id=None, size=0, name_len=0, size_len=0):
        self.position, self.name, self.record_id, self.size, self.name_len, self.size_len = position, name, record_id, size, name_len, size_len
    def __str__(self):
        return str((self.position, self.name, self.record_id, self.size, self.name_len, self.size_len))



ebml_header = {
    "1a45dfa3" : EBML("EBML", 0, "Master"),
    "4286" : EBML("EBMLVersion", 1, "uint"),
    "42f7" : EBML("EBMLReadVersion", 1, "uint"),
    "42f2" : EBML("EBMLMaxIDLength", 1, "uint"),
    "42f3" : EBML("EBMLMaxSizeLength", 1, "uint"), 
    "4282" : EBML("DocType", 1, "string"),
    "4287" : EBML("DocTypeVersion", 1, "uint"),
    "4285" : EBML("DocTypeReadVersion", 1, "uint"),
}

ebml_special = {
    "bf" : EBML("CRC32", None, "binary"),
    "ec" : EBML("Void", None, "binary")
}
ebml = {

    "18538067" : EBML("Segment", 0, "Master", {
        "114d9b74" : EBML("SeekHead", 1, "Master", {
            "4dbb" : EBML("Seek", 2, "Master"),
            "53ab" : EBML("SeekID", 3, "uint"),
            "53ac" : EBML("SeekPosition", 3, "uint"),
            }),

        "1549a966" : EBML("SegmentInfo", 1, "Master", { 
            "73a4" : EBML("SegmentUID", 2, "char[16]"),
            "7384" : EBML("SegmentFilename", 2, "utf-8"),
            "3cb923" : EBML("PrevUID", 2, "char[16]"),
            "3c83ab" : EBML("PrevFilename", 2, "utf-8"),
            "3eb923" : EBML("NextUID", 2, "char[16]"),
            "3e83bb" : EBML("NextFilename", 2, "utf-8"),
            "2ad7b1" : EBML("TimecodeScale", 2, "uint"),
            "4489"   : EBML("Duration", 2, "float"),
            "7ba9"   : EBML("Title", 2, "utf-8"),
            "4d80"   : EBML("MuxingApp", 2, "string"),
            "5741"   : EBML("WritingApp", 2, "utf-8"),
            "4461"   : EBML("DateUTC", 2, "int")
            }),
        "1f43b675" : EBML("Cluster", 1, "Master", {
            "e7"   : EBML("Timestamp", 2, "uint"),
            "a7"   : EBML("Position", 2, "uint"),
            "ab"   : EBML("PrevSize", 2, "uint"),
            "a3"   : EBML("SimpleBlock", 2, "binary"),
            "a0"   : EBML("BlockGroup", 2, "Master"), 
            "a1"   : EBML("Block", 3, "binary"),
            "fb"   : EBML("Reference Block", 3, "int"), 
            "9b"   : EBML("Block duration", 3, "int")
            }),
        "1654ae6b" : EBML("Tracks", 1, "Master", {
              "ae" : EBML("TrackEntry", 2, "Master"),
              "d7" : EBML("TrackNumber", 3, "uint"),
            "73c5" : EBML("TrackUID", 3, "uint"), 
            "83"   : EBML("TrackType", 3, "uint"),
            "b9"   : EBML("FlagEnabled", 3, "bool"),
            "88"   : EBML("FlagDefault", 3, "bool"),
            "55aa" : EBML("FlagForced", 3, "bool"),
            "9c"   : EBML("FlagLacing", 3, "bool"),
            "6de7" : EBML("MinCache", 3, "uint"),
            "6df8" : EBML("MaxCache", 3, "uint"),
            "22b59c" : EBML("Language", 3, "string"),
            "86" : EBML("CodecID", 3, "string"),
            "63a2": EBML("CodecPrivate", 3, "binary"),
            "258688" : EBML("CodecName", 3, "utf-8"),
            "7446" : EBML("AttachmentLink", 3, "uint"),
            "e0" : EBML("Video", 3, "Master"),
            "9a" : EBML("FlagInterlaced", 4, "uint"),
            "b0" : EBML("PixelWidth", 4, "uint"),
            "ba" : EBML("PixelHeight", 4, "uint"),
            "54aa" : EBML("PixelCropBottom", 4, "uint"),
            "54bb" : EBML("PixelCropTop", 4, "uint"),
            "54b0" : EBML("DisplayWidth", 4, "uint"),
            "54ba" : EBML("DislayHeight", 4, "uint"),
            "54b2" : EBML("DisplayUnit", 4, "uint"),
            "e1" : EBML("Audio", 3, "Master")
            }),

        "1c53bb6b" : EBML("Cues", 1, "Master", {
            "bb"   : EBML("CuePoint", 2, "Master"),
            "b3"   : EBML("CueTime", 3, "uint"),
            "b7"   : EBML("CueTrackPositions", 3, "Master"),
            "f7"   : EBML("CueTrack", 4, "uint"),
            "f1"   : EBML("CueClusterPostion", 4, "uint"),
            "5378" : EBML("CueBlockNumber", 4, "uint"),
            "f0"   : EBML("CueRelativePosition", 4, "uint"),
            }),
        "1941a469" : EBML("Attachments", 1, "Master", {}),
        "1254c367" : EBML("Tags", 1, "Master", {
            "7373" : EBML("Tag", 2, "Master"),
            "63c0" : EBML("Targets", 3, "Master"),
            "67c8" : EBML("SimpleTag", 3, "Master")
            })}),
}


# (element_name, size, bytes_traversed)
ebml_stack = []
ebml_tree  = []

def read_vint(file, mask=False):
    byte = file.read(1)
    if byte == b'':
        return None, None #if there are no more bytes to be read
    else:
        bytestring = f"{int(byte.hex(), 16):08b}"
        size = bytestring.find('1') + 1

        if mask:    
            bytestring = bytestring[:size-1] + "0" + bytestring[size:]
            vint = bytearray(int(bytestring, 2).to_bytes(len(bytestring) // 8, byteorder='big'))
        else:
            vint = bytearray(byte)

        if size > 1:
            vint += file.read(size-1)
        return vint, size

def byte_to_int(byte):
    return int(byte.hex(), 16)
def byte_to_binary(byte):
    return f"{int(byte.hex(), 16):08b}"
def int_to_binary(integer):
    return f"{integer:08b}"
def binary_to_byte(binary_string):
    return int(binary_string, 2).to_bytes(len(binary_string) // 8, byteorder='big')

def parse_mkv(file):
    file_data = []
    nalu_length_bytes = 4

    position = 0
    with open(file, 'rb') as f:
        ebml_id, id_len = None, None
        size, size_len = None, None
        data, data_len = None, None

        ## EBML HEADER

        #read EBML attribute
        ebml_id, id_len = read_vint(f)
        size, size_len = read_vint(f, True)
        #position += id_len + size_len
        
        assert ebml_id.hex() == "1a45dfa3", print("Unrecognized EBML attribute.")
        header_size = int(size.hex(), 16)



        #ebml_item = EBML_Record(position = position, name = ebml_header[ebml_id.hex()].name, record_id = ebml_id.hex(), size=header_size, name_len=id_len, size_len=size_len)
        #ebml_stack.append(item)

        #if args.verbose:
        stack_print(f.tell()-id_len-size_len, 0, "EBML Header")
        bytes_read  = 0

        while header_size > bytes_read:
            ebml_id, id_len = read_vint(f)

  
            assert ebml_id.hex() in ebml_header, print("Unrecognized EBML header format.")

            size, size_len = read_vint(f, True)
            data_len = int(size.hex(), 16)
            data = f.read(int(size.hex(),16))

            #file_data.append(EBML_Data(position, ebml_header[ebml_id.hex()].name, ebml_id, data_len, size, data))
            if ebml_header[ebml_id.hex()].data_type == "uint":
                data = f"{int(data.hex(), 16):1d}"
            else:
                data = str(data, "utf-8")
            #if args.verbose:
            stack_print(f.tell()-data_len-size_len-id_len, 1, f"{ebml_header[ebml_id.hex()].name:20s}={data}")
            bytes_read += id_len + size_len + data_len
            #position += id_len + size_len + data_len
        
                
        ## EBML BODY

        void_buffer = None
        while True:

            ebml_start = f.tell()
            ebml_id, id_len = read_vint(f)

            if ebml_id is None:
                break
            
            #position += id_len


            lut = ebml
            for i in range(len(ebml_stack)): #traverse ebml tree
                if i >= 2:
                    break
                else:
                    lut = lut[ebml_stack[i].record_id].child

            if ebml_id.hex() in lut:
                if void_buffer is not None:
                    if args.verbose:
                        stack_print(f.tell(), len(ebml_stack), f"Void binary data of size {len(void_buffer)}.")
                    void_buffer = None

                size, size_len = read_vint(f, True)
                #position += size_len

                data_len = int(size.hex(), 16)

                ebml_ref = lut[ebml_id.hex()]

                # if ebml_ref.name == "SimpleBlock":
                #     block_count += 1
                if ebml_ref.level:
                    assert ebml_ref.level == len(ebml_stack), print("EBML level error.")
                ebml_item = EBML_Record(position = ebml_start, name = ebml_ref.name, record_id = ebml_id.hex(), size=data_len, name_len=id_len, size_len=size_len)
                
                

                data_type = ebml_ref.data_type

                if data_type == "binary" and ebml_item.name == "SimpleBlock":
                    ts, ts_size = read_vint(f)
                    sb_header = f.read(3)

                    actual_size = data_len - (ts_size+3)
                    #if (data_len-ts_size-3 in size_map):
                    #    print(f"{data_len} block, {size_map[actual_size]} frame")
                    data_position = f.tell()
                    data = f.read(actual_size)
                    #nalu_bytes = 0
                    #while nalu_bytes < actual_size:
                    #    data = f.read(4)

                    file_data.append(EBML_Data(position = ebml_start, 
                        name = ebml_item.name, 
                        name_bytes = ebml_id, 
                        size = data_len, 
                        size_bytes = size,
                        data_position = data_position,
                        data_bytes = bytearray(data)))
                    #position += data_len
                    #if args.verbose:
                    stack_print(ebml_start, len(ebml_stack), f"{ebml_item.name}({ebml_item.size})")
                elif data_type == "binary" and ebml_item.name == "CodecPrivate":
                    data_position = f.tell()
                    data = f.read(data_len)
                    file_data.append(EBML_Data(position = ebml_start, 
                        name = ebml_item.name, 
                        name_bytes = ebml_id, 
                        size = data_len, 
                        size_bytes = size, 
                        data_position = data_position, 
                        data_bytes = bytearray(data)))
                    #position += data_len
                    #print(data)
                elif data_type != "Master":
                    #print(ebml_item[2])
                    data = f.read(data_len)
                    #position += data_len

                    if data_type == "uint" or data_type == "int":
                        data = int(data.hex(), 16)
                    elif data_type == "string":
                        data = data.decode("ascii")
                    elif data_type == "utf-8":
                        data = str(data, "utf-8")
                    elif data_type == "char[16]":
                        data = int(data.hex(), 16)
                    elif data_type == "bool":
                        data = False if data == b'\x00' else True
                    elif data_type == "binary":
                        data = data.hex()
                    elif data_type == "float":
                        pass #unimplemented
                    else:
                        print(f"Unknown data type {data_type} found.")

                    if data_type == "binary" and args.verbose:
                        stack_print(ebml_start, len(ebml_stack), f"{ebml_item.name}({ebml_item.size})")

                        #print(f"\n{data}")
                        
                    else:
                        #if args.verbose:
                        stack_print(ebml_start, len(ebml_stack), f"{ebml_item.name}({ebml_item.size})={data}")
                else:
                    #if args.verbose:
                    stack_print(ebml_start, len(ebml_stack), f"{ebml_item.name}({id_len}, {size_len}, {data_len})=(")
                    #stack_print(ebml_start, len(ebml_stack), f"size[{id_len}, {size_len}, {data_len}]=" + f"({ebml_id.hex()}, size {size.hex()})")
                
                ebml_stack.append(ebml_item)

            elif ebml_id == b'\xec': #void values
                if void_buffer is None:
                    void_buffer = bytearray(ebml_id)
                else:
                    void_buffer += ebml_id
            elif ebml_id == b'\xbf': #crc32 
                crc = f.read(5)
                #if args.verbose:
                stack_print(ebml_start, len(ebml_stack), f"CRC32 : {crc.hex()}")
                #position += 5
            else:
                if ebml_id != b'\x00': #ignore null bytes
                    #if args.verbose:
                    stack_print(ebml_start, len(ebml_stack), f"Unrecognized EBML symbol: {ebml_id.hex()}")

                    size, size_len = read_vint(f, True)
                    #position += size_len

                    data_len = int(size.hex(), 16)
                    #position += data_len
                    f.read(data_len)



            while ebml_stack:
                if ebml_stack[-1].size + ebml_stack[-1].position + ebml_stack[-1].name_len + ebml_stack[-1].size_len <= f.tell():

                    assert ebml_stack[-1].size + ebml_stack[-1].position + ebml_stack[-1].name_len + ebml_stack[-1].size_len == f.tell(), f"Misalignment issue with {ebml_stack[-1].name} where {ebml_stack[-1].size} + {ebml_stack[-1].position} + {ebml_stack[-1].name_len} + {ebml_stack[-1].size_len} < {f.tell()}."

                    ebml_item = ebml_stack.pop()

                    #if args.debug:
                    #print(f"{f.tell():8d} :: " +  "\t"*len(ebml_stack) + f"{ebml_item.name} end with total element size {ebml_item.size + ebml_item.name_len + ebml_item.size_len}.")
                else:
                    break
        return file_data

###H264

@dataclass
class SPS:
    seq_scaling_list_present_flag : List
    offset_for_ref_frame : List 
    profile_idc : int = -1
    level_idc : int   = -1
    sps_id : int      = -1

    chroma_format_idc : int = -1
    separate_colour_plane_flag : bool = False #chroma array type is equal to chroma_format_idc if separate_colour_plane is false, else 0

    bit_depth_luma_minus8  : int = -1
    bit_depth_chroma_minus8 : int = -1
    qpprime_y_zero_transform_bypass_flag : bool = False
    seq_scaling_matrix_present_flag : int = -1
    #seq_scaling_list_present_flag : List

    log2_max_frame_num_minus4 : int = -1
    pic_order_cnt_type : int = -1
    log2_max_pic_order_cnt_lsb_minus4 : int = -1

    delta_pic_order_always_zero_flag : bool = False
    offset_for_non_ref_pic : int = -1
    offset_for_top_to_bottom_field : int = -1
    num_ref_frames_in_pic_order_cnt_cycle : int = -1
    #offset_for_ref_frame : List 

    max_num_ref_frames : int = -1
    gaps_in_frame_num_value_allowed_flag : bool = False
    pic_width_in_mbs_minus1 : int = -1
    pic_height_in_map_units_minus1 : int = -1
    frame_mbs_only_flag : bool = False

    mb_adaptive_frame_field_flag : bool = False

    direct_8x8_inference: int = -1
    frame_cropping_flag: bool = False

    frame_crop_left_offset : int = -1
    frame_crop_right_offset : int = -1
    frame_crop_top_offset : int = -1
    frame_crop_bottom_offset : int = -1

    vui_parameters_present_flag: bool = False

    aspect_ratio_info_present_flag: bool = False
    aspect_ratio_idc : int = -1
    sar_width : int = -1
    sar_height : int = -1

    overscan_info_present : bool = False
    overscan_appropriate_flag : bool = False

    video_signal_type_present_flag : bool = False
    video_format : int = -1
    video_full_range_flag : bool = False
    colour_description_present_flag : bool = False
    colour_primaries : int = -1
    transfer_characteristics : int = -1
    matrix_coefficients : int = -1

    chroma_loc_info_present_flag : bool = False
    chroma_sample_loc_type_top_field : int = -1
    chroma_sample_loc_type_bottom_field : int = -1

    timing_info_present_flag : bool = False
    num_units_in_tick : int = -1
    time_scale : int = -1
    fixed_frame_rate_flag : bool = False

    nal_hrd_parameter_present_flag : bool = False
    #hrd_parameter
    vcl_hrd_parameters_present_flag : bool = False
    #hrd_parameters()
    low_delay_hrd_flag : bool = False
    pic_struct_present_flag : bool = False
    bitstream_restriction_flag : bool = False

    motion_vectors_over_pic_boundaries_flag : bool = False
    max_bytes_per_pic_denom : int = -1
    max_bits_per_mb_denom : int = -1
    log2_max_mv_length_horizontal : int = -1
    log2_max_mv_length_vertical : int = -1
    max_num_reorder_frames : int = -1
    max_dec_frame_buffering : int = -1

@dataclass
class PPS:
    run_length_minus1 : List
    top_left : List
    bottom_right : List
    slice_group_id : List

    pps_id : int = -1
    sps_id : int = -1
    entropy_coding_mode_flag : bool = False
    bottom_field_pic_order_in_frame_present_flag : bool = False
    num_slice_groups_minus1 : int = -1
    slice_group_map_type : int = -1
    # run_length_minus1 : List
    # top_left : List
    # bottom_right : List

    slice_group_change_direction_flag : bool = False
    slice_group_change_rate_minus1 : int = -1

    pic_size_in_map_units_minus1 : int = -1
    #slice_group_id : List

    num_ref_idx_l0_default_active_minus1 : int = -1
    num_ref_idx_l1_default_active_minus1 : int = -1
    weighted_pred_flag : bool = False
    weighted_bipred_idc : int = -1
    pic_init_qp_minus26 : int = -1
    pic_init_qs_minus26 : int = -1

    chroma_qp_index_offset : int = -1
    deblocking_filter_control_present_flag : bool = False
    constrained_intra_pred_flag : bool = False
    redundant_pic_cnt_present_flag : bool = False

    transform_8x8_mode_flag : bool = False
    pic_scaling_matrix_present_flag : bool = False
    pic_scaling_lislice_type_present_flag : bool = False
    second_chroma_qp_index_offset : int = -1


class Parser:
    def __init__(self, byte_array, file_position):
        self.byte_array = byte_array
        self.file_position = file_position
        self.byte_index = 0
        self.bit_index = 0

    def read_u(self, v):
        if 8 > v + self.bit_index:
            read_value = int(int_to_binary(self.byte_array[self.byte_index])[self.bit_index:self.bit_index+v], 2)
            self.bit_index += v
            return read_value
        elif 8 == v + self.bit_index:
            read_value = int(int_to_binary(self.byte_array[self.byte_index])[self.bit_index:], 2)
            self.bit_index = 0
            self.byte_index += 1
            self.detect_emulation()
            return read_value
        else:
            bits = int_to_binary(self.byte_array[self.byte_index])[self.bit_index:]
            v -= len(bits)
            while v > 8:
                self.byte_index += 1
                self.detect_emulation()
                bits_to_add = int_to_binary(self.byte_array[self.byte_index])
                v -= len(bits_to_add)
                bits += bits_to_add

            self.byte_index += 1
            self.detect_emulation()
            bits += int_to_binary(self.byte_array[self.byte_index])[:v]
            if v == 8:
                self.byte_index += 1
                v = 0

            self.bit_index = v
            return int(bits, 2)

    def read_ue(self):
        curr_byte = self.byte_array[self.byte_index]
        binary_string = int_to_binary(curr_byte)
        #stack_print(byte_index, 2, f"read_ue(data_array, byte_index{byte_index}, bit_index:{bit_index})")
        #stack_print(byte_index, 2, binary_string)
        index = binary_string.find('1', self.bit_index)
        #print(f"string:{binary_string}, index:{index}, bit_index:{bit_index}")
        leading_zero = 0
        count = 0
        if index == -1:
            #print(f"did not find terminating 1 in bitstring {binary_string} with bit_index {bit_index}")
            leading_zero += 8-self.bit_index
            while index == -1:
                self.byte_index += 1
                self.detect_emulation()
                binary_string = int_to_binary(self.byte_array[self.byte_index])
                #stack_print(byte_index, 2, binary_string)
                #print(binary_string)
                index = binary_string.find('1')
                #print(f"string:{binary_string}, index:{index}")

                leading_zero += 8 if index == -1 else index
        else:
            leading_zero = index-self.bit_index
        #print(f"leading_zeroes:{leading_zero}")
        bits_to_read = leading_zero


        if leading_zero != 0:
            if 8-(index+1) >= bits_to_read:
                bits = binary_string[index+1:index+1+bits_to_read]
                bit_offset = index + 1 + bits_to_read
                bits_to_read = 0
            else:
                bit_offset = 0
                if index != 8:
                    bits = binary_string[index+1:]
                    bits_to_read -= len(bits)

                else:
                    bits = ''
                    #print(f"bits in func:{bits}")


            while bits_to_read > 0:
                self.byte_index += 1
                self.detect_emulation()
                #print(count)
                #print(f"bits_to_read:{bits_to_read}")
                binary_string = int_to_binary(self.byte_array[self.byte_index])
                #stack_print(byte_index, 2, binary_string)

                if bits_to_read >= 8:
                    bits += binary_string
                    bits_to_read -= 8
                else:
                    bits += binary_string[:bits_to_read]
                    bit_offset = bits_to_read
                    bits_to_read = 0
        else:
            bit_offset = index + 1
        if bit_offset == 8:
            self.byte_index += 1
            bit_offset = 0
        self.bit_index = bit_offset
        return 2**(leading_zero) - 1 + int(bits, 2) if leading_zero != 0 else 0
    def read_se(self):
        code_num = self.read_ue()
        return (-1)**(code_num+1)*math.ceil(code_num/2)

    def is_empty(self):
        return self.byte_index >= len(self.byte_array)
    def detect_emulation(self):
        if self.byte_index >= 2:
            if self.byte_array[self.byte_index-2] == 0 and self.byte_array[self.byte_index-1] == 0 and self.byte_array[self.byte_index] == 3:
                stack_print(self.byte_index, 3, "emulation byte detected.")
                self.byte_index += 1

def verbose_print(string):
    if args.verbose:
        print(string)
def stack_print(position, depth, string):
    if args.verbose:
        print(f"{position:8d} :: " +  "\t"*depth + string)
def debug_print(string):
    if args.debug:
        print(string)

sps = SPS(seq_scaling_list_present_flag = [], offset_for_ref_frame = [])
pps = PPS(run_length_minus1 = [], top_left = [], bottom_right = [], slice_group_id = [])
nalu_length = 4 #default


def parse_slice(parser, nal_ref_idc, nal_unit_type, nalu_end):
    try:
        chroma_array_type = sps.chroma_format_idc if not sps.separate_colour_plane_flag else 0 
        idr_pic_flag = True if nal_unit_type == 5 else False

        ##header 
        data = parser.read_ue()
        #print(f"data: {data}, bytes_parsed:{bytes_parsed}, bit_offset:{bit_offset}")
        slice_type = parser.read_ue()
        #print(f"data: {slice_type}, bytes_parsed:{bytes_parsed}, bit_offset:{bit_offset}")
        pps_id = parser.read_ue()
        #print(f"data: {data}, bytes_parsed:{bytes_parsed}, bit_offset:{bit_offset}")
        stack_print(parser.byte_index, 2, f"pps_id:{pps_id}")

        #data = int_to_binary(data_bytes[byte_position])
        frame_num = parser.read_u(sps.log2_max_frame_num_minus4+4)
        stack_print(parser.byte_index, 2, f"frame_num:{frame_num}")

        if not sps.frame_mbs_only_flag:
            field_pic_flag = parser.read_u(1)
            if field_pic_flag == 1:
                bottom_field_flag = parser.read_u(1)
        slice_type_p, slice_type_b, slice_type_i, slice_type_sp, slice_type_si = slice_type%5 == 0, slice_type%5 == 1, slice_type%5 == 2, slice_type%5 == 3, slice_type%5 == 4

        if slice_type%5 == 0:
            slice_type = "P"
        elif slice_type%5 == 1:
            slice_type = "B"
        elif slice_type%5 == 2:
            slice_type = "I"
        elif slice_type%5 == 3:
            slice_type = "SP"
        elif slice_type%5 == 4:
            slice_type = "SI"
        stack_print(parser.byte_index, 2, f"slice_type:{slice_type}")

        if idr_pic_flag:
            idr_pic_id = parser.read_ue()

        #print(f"data: {data}, bytes_parsed:{bytes_parsed}, bit_offset:{bit_offset}")
        #print(f"{data_bytes[byte_index:byte_index+10].hex()}")
        if sps.pic_order_cnt_type == 0 :
            pic_order_cnt_lsb =parser.read_u(sps.log2_max_pic_order_cnt_lsb_minus4+4)
            stack_print(parser.byte_index, 2, f"pic_order_cnt_lsb:{pic_order_cnt_lsb}")
            if pps.bottom_field_pic_order_in_frame_present_flag and not field_pic_flag:
                delta_pic_order_cnt_bottom = read_se()
        if pps.redundant_pic_cnt_present_flag:
            redundant_pic_cnt = parser.read_ue()
            stack_print(parser.byte_index, 2, f"redundant_pic_cnt:{redundant_pic_cnt}")

            #stack_print()
        if slice_type_b: #b slices
            direct_spatial_mv_pred_flag = parser.read_u(1)
            stack_print(parser.byte_index, 2, f"direct_spatial_mv_pred_flag:{direct_spatial_mv_pred_flag}")


        if slice_type_b or slice_type_p or slice_type_sp: #p b or sp slice
            num_ref_idx_active_override_flag = parser.read_u(1)
            stack_print(parser.byte_index, 2, f"num_ref_idx_active_override_flag:{num_ref_idx_active_override_flag}")


            if num_ref_idx_active_override_flag == 1:
                num_ref_idx_l0_active = parser.read_ue()
                if slice_type_b:
                    num_ref_idx_l1_active = parser.read_ue()

        if nal_unit_type == 20 or nal_unit_type == 21:
            print("list_mvc_modification()")
        else:
            #ref_pic_list_modification(), 7.3.3.1
            if not slice_type_i and not slice_type_si:
                ref_pic_list_modification_flag_10 = parser.read_u(1)
                if ref_pic_list_modification_flag_10 == 1:
                    while True:
                        modification_of_pics_nums_idc = parser.read_ue()

                        if modification_of_pics_nums_idc == 0 or modification_of_pics_nums_idc == 1:
                            abs_diff_pic_num_minus1 = parser.read_ue()
                        elif modification_of_pics_nums_idc == 2:
                            long_term_pic_num = parser.read_ue()
                        elif modification_of_pics_nums_idc == 3:
                            break
            if slice_type_b:
                ref_pic_list_modification_flag_11 = parser.read_u(1)
                if ref_pic_list_modification_flag_11 == 1:
                    while True:
                        modification_of_pics_nums_idc = parser.read_ue()
                        stack_print(parser.byte_index, 2, f"modification_of_pics_nums_idc:{modification_of_pics_nums_idc}")

                        if modification_of_pics_nums_idc == 0 or modification_of_pics_nums_idc == 1:
                            abs_diff_pic_num_minus1 = parser.read_ue()
                        elif modification_of_pics_nums_idc == 2:
                            long_term_pic_num = parser.read_ue()
                        if modification_of_pics_nums_idc == 3:
                            break
        if (pps.weighted_pred_flag and (slice_type_p or slice_type_sp)) or (pps.weighted_bipred_idc == 1 and slice_type_b):
            #pred weight table
            luma_log2_weight_denom = parser.read_ue()

            if chroma_array_type != 0:
                chroma_log2_weight_denom = parser.read_ue()

            for i in range(pps.num_ref_idx_l0_default_active_minus1+1):
                luma_weight_l0_flag = bool(parser.read_u(1))

                if luma_weight_l0_flag:
                    luma_weight_l0 = parser.read_se()
                    luma_offset_l0 = parser.read_se()
                if chroma_array_type != 0:
                    chroma_weight_l0_flag = parser.read_u(1)
                    if chroma_weight_l0_flag:
                        for j in range(2):
                            chroma_weight_l0 = parser.read_se()
                            chroma_offset_l0 = parser.read_se()

            if slice_type_b:
                for i in range(pps.num_ref_idx_l1_default_active_minus1 + 1):
                    luma_weight_l1_flag = bool(parser.read_u(1))

                    if luma_weight_l1_flag:
                        luma_weight_l1 = parser.read_se()
                        luma_offset_l1 = parser.read_se()
                    if chroma_array_type != 0:
                        chroma_weight_l1_flag = bool(parser.read_u(1))
                        if chroma_weight_l1_flag:
                            for j in range(2):
                                chroma_weight_l1 = parser.read_se()
                                chroma_offset_l1 = parser.read_se()



        if nal_ref_idc != 0:
            if idr_pic_flag:
                no_output_of_prior_pics_flag = parser.read_u(1)
                long_term_reference_flag = parser.read_u(1)
            else:
                adaptive_ref_pic_marking_mode_flag = parser.read_u(1)
                if adaptive_ref_pic_marking_mode_flag:
                    while True:
                        memory_management_control_operation = parser.read_ue()
                        if memory_management_control_operation == 0:
                            break
                        if memory_management_control_operation == 1 or memory_management_control_operation == 3:
                            difference_of_pic_nums_minus1 = parser.read_ue()
                        if memory_management_control_operation == 2:
                            long_term_pic_num = parser.read_ue()
                        if memory_management_control_operation == 3 or memory_management_control_operation == 6:
                            long_term_frame_idx = parser.read_ue()
                        if memory_management_control_operation == 4:
                            max_long_term_frame_idx_plus1 = parser.read_ue()

        if pps.entropy_coding_mode_flag and not slice_type_i and not slice_type_si:
            cabac_init_idc = parser.read_ue()

        slice_qp_delta = parser.read_se()

        if slice_type_sp or slice_type_si:
            if slice_type_sp:
                sp_for_switch_flag = parser.read_u()
            slice_qs_delta = parser.read_se()
        if pps.deblocking_filter_control_present_flag:
            disable_deblocking_filter_idc = parser.read_ue()
            if disable_deblocking_filter_idc != 1:
                slice_alpha_c0_offset_div2 = parser.read_se()
                slice_beta_offset_div2 = parser.read_se()

        if pps.num_slice_groups_minus1 > 0 and slice_group_map_type >= 3 and slice_group_map_type <= 5:
            print("slice_group_change_cycle")

        #if weighted predict flag
        #data_block = byte_block[byte_position:]
        #data_block = simulate_error(data_block, slice_type)
        #print(binary_string[bit_offset:])
        #approximate_data.append((byte_position, data_block))

        ##begin slice data

        if pps.entropy_coding_mode_flag:
            while parser.bit_index != 0:
                parser.read_u(1)

        stack_print(parser.byte_index, 2, f"end of header:: bit_offset={parser.bit_index}")

        #print(f"compare {parser.byte_index} and {nalu_end}")
        data_block = parser.byte_array[parser.byte_index:nalu_end]
        debug_print("data hexdump:")
        debug_print(data_block.hex())
    except:
        verbose_print("Parsing error. Aborting parse on slice.")
        data_block = parser.byte_array[parser.byte_index:nalu_end]
        debug_print("block buffer dump:")
        debug_print(data_block.hex())



def parse_sps(parser, nal_ref_idc, nal_unit_type):

    sps.profile_idc = parser.read_u(8)
    stack_print(parser.byte_index, 2, f"profile_idc:{sps.profile_idc}")
    
    parser.read_u(8)

    sps.level_idc = parser.read_u(8)

    sps.sps_id = parser.read_ue()
    stack_print(parser.byte_index, 2, f"sps_id:{sps.sps_id}")

    if sps.profile_idc == 100 or sps.profile_idc == 110 or sps.profile_idc == 122 or sps.profile_idc == 244 or sps.profile_idc == 44 or \
    sps.profile_idc == 83 or sps.profile_idc == 86 or sps.profile_idc == 118 or sps.profile_idc == 128 or sps.profile_idc == 138 or \
    sps.profile_idc == 139 or sps.profile_idc == 134 or sps.profile_idc == 135:
        sps.chroma_format_idc = parser.read_ue()
        stack_print(parser.byte_index, 3, f"chroma_format_idc:{sps.chroma_format_idc}")

        if sps.chroma_format_idc == 3:
            sps.separate_colour_plane_flag = parser.read_u(1) 
        sps.bit_depth_luma_minus8 = parser.read_ue()
        stack_print(parser.byte_index, 3, f"bit_depth_luma_minus8:{sps.bit_depth_luma_minus8}")
        sps.bit_depth_chroma_minus8 = parser.read_ue()
        stack_print(parser.byte_index, 3, f"bit_depth_chrom_minus8:{sps.bit_depth_chroma_minus8}")
        sps.qpprime_y_zero_transform_bypass_flag = (parser.read_u(1) == 1)
        sps.seq_scaling_matrix_present_flag = bool(parser.read_u(1))
        stack_print(parser.byte_index, 3, f"seq_scaling_matrix_present_flag:{sps.seq_scaling_matrix_present_flag}")

        if sps.seq_scaling_matrix_present_flag:

            num_iterations = 8 if sps.chroma_format_idc != 3 else 12

            for i in range(num_iterations):
                seq_scaling_lislice_type_present_flag = bool(parser.read_u(1))
                stack_print(parser.byte_index, 3, f"seq_scaling_lislice_type_present_flag:{seq_scaling_lislice_type_present_flag}")
                sps.seq_scaling_lislice_type_present_flag.append(seq_scaling_lislice_type_present_flag)
                if seq_scaling_lislice_type_present_flag:
                    if i < 6:
                        print("scaling list")
                    else:
                        print("scaling list")


    sps.log2_max_frame_num_minus4 = parser.read_ue()
    stack_print(parser.byte_index, 2, f"log2_max_frame_num_minus4:{sps.log2_max_frame_num_minus4+4}")

    sps.pic_order_cnt_type = parser.read_ue()
    stack_print(parser.byte_index, 2, f"pic_order_cnt_type:{sps.pic_order_cnt_type}")

    if sps.pic_order_cnt_type == 0:
        sps.log2_max_pic_order_cnt_lsb_minus4 = parser.read_ue()
        stack_print(parser.byte_index, 3, f"log2_max_pic_order_cnt_lsb_minus4:{sps.log2_max_pic_order_cnt_lsb_minus4}")

    elif sps.pic_order_cnt_type == 1:
        sps.delta_pic_order_always_zero_flag = parser.read_u(1)   
        sps.offset_for_non_ref_pic = parser.read_se()
        sps.offset_for_top_to_bottom_field = parser.read_se()

        sps.num_ref_frames_in_pic_order_cnt_cycle = parser.read_ue()

        for i in range(sps.num_ref_frames_in_pic_order_cnt_cycle):
            offset = read_se()
            sps.offset_for_ref_frame.append(offset)
            
            stack_print(parser.byte_index, 1, f"offset_for_ref_frame:{offset}")


    #print(f"{data_bytes[byte_offset]:08b}, bit_offset:{bit_offset}")
    

    sps.max_num_ref_frames= parser.read_ue()
    
    stack_print(parser.byte_index, 2, f"max_num_ref_frames:{sps.max_num_ref_frames}")

    sps.gaps_in_frame_num_value_allowed_flag = parser.read_u(1)
    
    stack_print(parser.byte_index, 2, f"gaps_in_frame_num_value_allowed_flag:{sps.gaps_in_frame_num_value_allowed_flag}")

    sps.pic_width_in_mbs_minus1 = parser.read_ue()
    
    stack_print(parser.byte_index, 2, f"pic_width_in_mbs_minus1:{sps.pic_width_in_mbs_minus1}")
    

    sps.pic_height_in_map_units_minus1 = parser.read_ue()
    
    stack_print(parser.byte_index, 2, f"pic_height_in_map_units_minus1:{sps.pic_height_in_map_units_minus1}")


    sps.frame_mbs_only_flag = parser.read_u(1)  == 1
    stack_print(parser.byte_index, 2, f"frame_mbs_only_flag:{sps.frame_mbs_only_flag}")


    if not sps.frame_mbs_only_flag:
        sps.mb_adaptive_frame_field_flag = parser.read_u(1) == 1        
    
    sps.direct_8x8_inference = bool(parser.read_u(1))
    stack_print(parser.byte_index, 2, f"direct_8x8_inference:{sps.direct_8x8_inference}")

    sps.frame_cropping_flag = bool(parser.read_u(1))
    stack_print(parser.byte_index, 2, f"frame_cropping_flag:{sps.frame_cropping_flag}")
    
    if sps.frame_cropping_flag:
        sps.frame_crop_left_offset = parser.read_ue()
        sps.frame_crop_right_offset = parser.read_ue()
        sps.frame_crop_top_offset = parser.read_ue()
        sps.frame_crop_bottom_offset = parser.read_ue()

    sps.vui_parameters_present_flag = bool(parser.read_u(1))
    stack_print(parser.byte_index, 2, f"vui_parameters_present_flag:{sps.vui_parameters_present_flag}")
    if sps.vui_parameters_present_flag:

        sps.aspect_ratio_info_present_flag = bool(parser.read_u(1))
        
        if sps.aspect_ratio_info_present_flag:
            sps.aspect_ratio_idc = parser.read_u(8)
            
            if sps.aspect_ratio_idc == 255: #extended_SAR
                sps.sar_width =parser.read_u(16)
                
                sps.sar_height =parser.read_u(16)
                 

        overscan_info_present_flag = parser.read_u(1)
        
        sps.overscan_info_present_flag = overscan_info_present_flag == 1

        if sps.overscan_info_present_flag:
            overscan_appropriate_flag = parser.read_u(1)
            
            sps.overscan_appropriate_flag = overscan_appropriate_flag == 1

        video_signal_type_present_flag = parser.read_u(1)
        
        sps.video_signal_type_present_flag = video_signal_type_present_flag == 1
        if  sps.video_signal_type_present_flag:
            sps.video_format = parser.read_u(3)
            

            video_full_range_flag = parser.read_u(1)
             
            sps.video_full_range_flag = video_full_range_flag == 1

            colour_description_present_flag = parser.read_u(1)
            
            sps.colour_description_present_flag = colour_description_present_flag == 1

            if sps.colour_description_present_flag:
                sps.colour_primaries =parser.read_u(16)
                

                sps.transfer_characteristics =parser.read_u(16)
                

                sps.matrix_coefficients =parser.read_u(16)
                

        sps.chroma_loc_info_present_flag = bool(parser.read_u(1))
        
        if sps.chroma_loc_info_present_flag:
            sps.chroma_sample_loc_type_top_field = parser.read_ue()
            

            sps.chroma_sample_loc_type_bottom_field = parser.read_ue()
            

        sps.timing_info_present_flag = parser.read_u(1) == 1        
        if sps.timing_info_present_flag:
            sps.num_units_in_tick =parser.read_u(32)
            sps.time_scale =parser.read_u(32)
            sps.fixed_frame_rate_flag = bool(parser.read_u(1))
        sps.nal_hrd_parameter_present_flag = bool(parser.read_u(1))
        if sps.nal_hrd_parameter_present_flag:
            print("nal_hrd_parameter")
            pass

        sps.vcl_hrd_parameters_present_flag = parser.read_u(1) == 1        
        ## MORE PARSING
        if sps.nal_hrd_parameter_present_flag == 1 or sps.vcl_hrd_parameters_present_flag == 1:
            print("vcl_hrd_parameter)")
            pass
        sps.pic_struct_present_flag = bool(parser.read_u(1))

        sps.bitstream_restriction_flag = bool(parser.read_u(1))
        
        if sps.bitstream_restriction_flag:
            motion_vectors_over_pic_boundaries_flag = parser.read_u(1)
            
            sps.motion_vectors_over_pic_boundaries_flag = motion_vectors_over_pic_boundaries_flag == 1

            sps.max_bytes_per_pic_denom = parser.read_ue()
            

            sps.max_bits_per_mb_denom = parser.read_ue()
            

            sps.log2_max_mv_length_horizontal = parser.read_ue()
            

            sps.log2_max_mv_length_vertical = parser.read_ue()
            

            sps.max_num_reorder_frames = parser.read_ue()
            

            sps.max_dec_frame_buffering = parser.read_ue()
            

def parse_pps(parser, nal_ref_idc, nal_unit_type):
    pps.pps_id = parser.read_ue()
    
    stack_print(parser.byte_index, 2, f"pps_id:{pps.pps_id}")

    pps.sps_id =  parser.read_ue()
    
    stack_print(parser.byte_index,2, f"sps_id:{pps.sps_id}")   

    pps.entropy_coding_mode_flag = bool(parser.read_u(1))
    
    stack_print(parser.byte_index,2, f"entropy_coding_mode_flag:{pps.entropy_coding_mode_flag}")

    pps.bottom_field_pic_order_in_frame_present_flag = bool(parser.read_u(1))
    
    stack_print(parser.byte_index,2, f"bottom_field_pic_order_in_frame_present_flag:{pps.bottom_field_pic_order_in_frame_present_flag}")

    #print(f"byte_offset:{byte_offset}")
    pps.num_slice_groups_minus1 = parser.read_ue()
    
    stack_print(parser.byte_index,2, f"num_slice_groups_minus1:{pps.num_slice_groups_minus1}")
    

    if pps.num_slice_groups_minus1 > 0:
        pps.slice_group_map_type = parser.read_ue()
        
        stack_print(parser.byte_index,3, f"slice_group_map_type:{pps.slice_group_map_type}")
        if pps.slice_group_map_type == 0:
            pass
        elif pps.slice_group_map_type == 2:
            pass
        elif pps.slice_group_map_type == 3 or pps.slice_group_map_type == 4 or pps.slice_group_map_type == 5:
            pass
        elif pps.slice_group_map_type == 6:
            pass

    pps.num_ref_idx_l0_default_active_minus1 = parser.read_ue()
    

    pps.num_ref_idx_l1_default_active_minus1 = parser.read_ue()
    

    weighted_pred_flag = parser.read_u(1)
    
    pps.weighted_pred_flag = weighted_pred_flag == 1
    stack_print(parser.byte_index,2, f"weighted_pred_flag:{weighted_pred_flag ==1}")
    pps.weighted_bipred_idc = parser.read_u(2)
    
    stack_print(parser.byte_index,2, f"weighted_bipred_idc:{pps.weighted_bipred_idc}")

    pps.pic_init_qp_minus26 = parser.read_se()
    
    pps.pic_init_qs_minus26 = parser.read_se()
    

    pps.chroma_qp_index_offset = parser.read_se()
    

    pps.deblocking_filter_control_present_flag = bool(parser.read_u(1))
    
    pps.constrained_intra_pred_flag = bool(parser.read_u(1))
    
    pps.redundant_pic_cnt_present_flag = bool(parser.read_u(1))


    #print(f"compare {len(item.data_bytes)} and {byte_offset}")
    if len(parser.byte_array) > parser.byte_index:
        pps.transform_8x8_mode_flag = bool(parser.read_u(1))
        
        pps.pic_scaling_matrix_present_flag = bool(parser.read_u(1))

        num_iterations = (2 if sps.chroma_format_idc != 3 else 6)*pps.transform_8x8_mode_flag

        #for i in range(6+num_iterations):



def parse_nalu(parser, nalu_end):

    forbidden_zero_bit = parser.read_u(1)
    nal_ref_idc = parser.read_u(2)
    nal_unit_type = parser.read_u(5)
    if nal_unit_type == 14 or nal_unit_type == 20 or nal_unit_type == 21:
        svc_extension_flag, avc_3d_extension_flag = 0, 0
        if nal_unit_type != 21:
            svc_extension_flag = parser.read_u(1)
        else:
            avc_3d_extension_flag = parser.read_u(1)

        if svc_extension_flag == 1:
            print("nal_unit_header_svc_extension")
        elif avc_3d_extension_flag == 1:
            print("nal_unit_header_3davc_extension")
        else:
            print("nal_unit_header_mvc_extension")


    if nal_unit_type == 8:
        stack_print(parser.byte_index, 1, "PPS")
        parse_pps(parser, nal_ref_idc, nal_unit_type)
    elif nal_unit_type == 7:
        stack_print(parser.byte_index,1,  "SPS") 
        parse_sps(parser, nal_ref_idc, nal_unit_type)
    elif nal_unit_type == 6:
        stack_print(parser.byte_index, 1, "SEI")
        #skip.
    elif nal_unit_type >= 1 and nal_unit_type <= 5:
        stack_print(parser.byte_index, 1, "Slice")

        parse_slice(parser, nal_ref_idc, nal_unit_type, nalu_end)

    parser.byte_index = nalu_end
    parser.bit_index = 0



def main():
    position = 0

    video_file = Path(args.file)
    if not video_file.exists():
        raise Exception("Video file specified does not exist.")


    verbose_print("-"*20)
    verbose_print("Parsing mkv file...")
    file_data = parse_mkv(video_file)

    # cp_cmd = f"cp {args.file} ./{output_file}"
    # process = subprocess.Popen(cp_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # stdout, stderr = process.communicate()
    verbose_print("-"*20)
    verbose_print("Parsing h264 avcc blocks...")
    for item in file_data:
        stack_print(0, 0, f"{item.name} at {item.position} with size {len(item.data_bytes)}")
        
        data_bytes = bytearray(item.data_bytes)

        parser = Parser(byte_array=data_bytes, file_position=item.data_position)
        
        if item.name == "SimpleBlock":
            while not parser.is_empty():
                nalu_length = parser.read_u((nalu_length_minus_one+1)*8)
               
                stack_print(parser.byte_index, 0, f"nalu_length:{nalu_length}")
                parse_nalu(parser, parser.byte_index + nalu_length)

                # if original_pos + nalu_length != parser.byte_index:
                #     stack_print(parser.byte_index, 0, "misalignment found.")
        if item.name == "CodecPrivate": #extradata
            nalu_length_minus_one = int(f"{parser.byte_array[4]:08b}"[-2:], 2)
            nalu_length = nalu_length_minus_one + 1
            num_sps = int(f"{parser.byte_array[5]:08b}"[-5:], 2)
            stack_print(5, 0, f"num_sps:{num_sps}")
            
            parser.byte_index = 6

            for i in range(num_sps):
                sps_length = parser.read_u(16)

                parse_nalu(parser, parser.byte_index+sps_length)

            num_pps = parser.read_u(8)
            for i in range(num_pps):
                pps_length = parser.read_u(16)
                parse_nalu(parser, parser.byte_index+pps_length)


        
            #print(byte_index)
            #7.3.3 page 73
    verbose_print("-"*40)
    verbose_print("Finished parsing file.")
        #print(f"blocks:{block_count}") #blocks

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="This is a simple mkv parser that outputs the structure and contents of the mkv sections. \
        This parser also has basic x264 parsing, allowing for simple extraction of some x264 data.")
    parser.add_argument("file", type=str,
                        help="Mkv file to parse.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose print for parsing details.")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug print for parsing nitty gritty and debugging.")
    
    args = parser.parse_args()

    main()