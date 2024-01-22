import os
import cv2
import signal
import numpy as np
from PIL import Image
from functools import partial
from multiprocessing import Pool
from plyfile import PlyData, PlyElement

from datasets.data_io import read_pfm
from filter.tank_test_config import tank_cfg


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,
                         src_confidence, src_confidence2, src_confidence1):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    sample_confi1 = cv2.remap(src_confidence1, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    sample_confi2 = cv2.remap(src_confidence2, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    sample_confi3 = cv2.remap(src_confidence, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src, sample_confi1, sample_confi2, sample_confi3


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,
                                src_confidence, src_confidence2, src_confidence1):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src, sample_confi1, sample_confi2, sample_confi3 \
        = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                               depth_src, intrinsics_src, extrinsics_src,
                               src_confidence, src_confidence2, src_confidence1)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    # dist_map = 1 * sample_confi3
    # relative_depth_diff_map = 0.01 * sample_confi3

    mask2 = np.logical_and(dist < 0.6, depth_diff < 0.1)

    mask3 = np.logical_and(dist < 0.7, depth_diff < 0.5)  # 0.1 0.01
    # 147 135
    mask4 = np.logical_and(dist < 0.8, depth_diff < 0.75)  # 0.1 0.01

    mask5 = np.logical_and(dist < 0.85, depth_diff < 1)  # 0.1 0.01

    mask6 = np.logical_and(dist < 0.9, depth_diff < 1.25)  # 0.1 0.01

    mask7 = np.logical_and(dist < 0.95, depth_diff < 1.55)  # 0.1 0.01

    mask8 = np.logical_and(dist < 1, depth_diff < 1.75)  # 0.1 0.01

    mask9 = np.logical_and(dist < 1, depth_diff < 1.95)  # 0.1 0.01
    # depth_reprojected[~mask] = 0
    '''
    error = depth_diff[mask]
    print("max: ",error.max())
    print("min: ",error.min())
    print("mean: ",error.mean())
    print("msk_num: ",mask.astype(float).mean())
    print("<1: ",(error<1).astype(float).mean())
    '''
    return mask9, mask8, mask7, mask6, mask5, mask2, mask3, mask4, depth_reprojected, x2d_src, y2d_src, sample_confi1, sample_confi2, sample_confi3


def filter_depth(args, pair_folder, scan_folder, out_folder, plyfilename):
    num_stage = len(args.ndepths)

    # the pair file
    pair_file = os.path.join(pair_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # src_views = src_views[:args.num_view]
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        confidence2 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage2.pfm'.format(ref_view)))[0]
        confidence1 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage1.pfm'.format(ref_view)))[0]
        photo_mask = np.logical_and(np.logical_and(confidence > args.conf[2], confidence2 > args.conf[1]),
                                    confidence1 > args.conf[0])

        # all_srcview_x = []
        # all_srcview_y = []
        all_srcview_depth_ests2 = []
        all_srcview_geomask2 = []

        all_srcview_depth_ests3 = []
        all_srcview_geomask3 = []

        all_srcview_depth_ests4 = []
        all_srcview_geomask4 = []

        all_srcview_depth_ests5 = []
        all_srcview_geomask5 = []

        all_srcview_depth_ests6 = []
        all_srcview_geomask6 = []

        all_srcview_depth_ests7 = []
        all_srcview_geomask7 = []

        all_srcview_depth_ests8 = []
        all_srcview_geomask8 = []

        all_srcview_depth_ests9 = []
        all_srcview_geomask9 = []

        # compute the geometric mask
        geo_mask_sum5 = 0
        geo_mask_sum2 = 0
        geo_mask_sum3 = 0
        geo_mask_sum4 = 0
        geo_mask_sum6 = 0
        geo_mask_sum7 = 0
        geo_mask_sum8 = 0
        geo_mask_sum9 = 0
        views_num = 0
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            src_confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(src_view)))[0]
            src_confidence2 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage2.pfm'.format(src_view)))[0]
            src_confidence1 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage1.pfm'.format(src_view)))[0]
            #

            geo_mask9, geo_mask8, geo_mask7, geo_mask6, geo_mask5, geo_mask2, geo_mask3, geo_mask4, \
            depth_reprojected, x2d_src, y2d_src, \
            sample_confi1, sample_confi2, sample_confi3 = check_geometric_consistency(
                ref_depth_est, ref_intrinsics, ref_extrinsics,
                src_depth_est,
                src_intrinsics, src_extrinsics,
                src_confidence, src_confidence2,
                src_confidence1)
            # src_photo_mask5 = np.logical_and(np.logical_and(sample_confi3 > 0.5,
            # sample_confi2 > 0.15),
            # sample_confi1 > 0.1)
            # src_photo_mask2 = np.logical_and(np.logical_and(sample_confi3 > 0.5,
            ##sample_confi2 > 0.15),
            # sample_confi1 > 0.1)
            # src_photo_mask3 = np.logical_and(np.logical_and(sample_confi3 > 0.5,
            # sample_confi2 > 0.15),
            # sample_confi1 > 0.1)
            ##src_photo_mask4 = np.logical_and(np.logical_and(sample_confi3 > 0.5,
            # sample_confi2 > 0.15),
            # sample_confi1 > 0.1)
            # geo_mask5 = np.logical_and(src_photo_mask5, geo_mask5)
            # geo_mask2 = np.logical_and(src_photo_mask2, geo_mask2)
            # geo_mask3 = np.logical_and(src_photo_mask3, geo_mask3)
            # geo_mask4 = np.logical_and(src_photo_mask4, geo_mask4)
            # print("***************per view***********************")
            # print(geo_mask3.mean())
            # print(geo_mask4.mean())
            # print(geo_mask5.mean())
            # print(src_photo_mask.mean())
            # geo_mask = np.logical_and(src_photo_mask, geo_mask)
            # 2
            dp2 = depth_reprojected.copy()
            dp2[~geo_mask2] = 0
            geo_mask_sum2 += geo_mask2.astype(np.int32)
            all_srcview_depth_ests2.append(dp2)
            all_srcview_geomask2.append(geo_mask2)
            # 3
            dp3 = depth_reprojected.copy()
            dp3[~geo_mask3] = 0
            geo_mask_sum3 += geo_mask3.astype(np.int32)
            all_srcview_depth_ests3.append(dp3)
            all_srcview_geomask3.append(geo_mask3)
            # 4
            dp4 = depth_reprojected.copy()
            dp4[~geo_mask4] = 0
            geo_mask_sum4 += geo_mask4.astype(np.int32)
            all_srcview_depth_ests4.append(dp4)
            all_srcview_geomask4.append(geo_mask4)
            # 6
            dp6 = depth_reprojected.copy()
            dp6[~geo_mask6] = 0
            geo_mask_sum6 += geo_mask6.astype(np.int32)
            all_srcview_depth_ests6.append(dp6)
            all_srcview_geomask6.append(geo_mask6)
            # 7
            dp7 = depth_reprojected.copy()
            dp7[~geo_mask7] = 0
            geo_mask_sum7 += geo_mask7.astype(np.int32)
            all_srcview_depth_ests7.append(dp7)
            all_srcview_geomask7.append(geo_mask7)
            # 8
            dp8 = depth_reprojected.copy()
            dp8[~geo_mask8] = 0
            geo_mask_sum8 += geo_mask8.astype(np.int32)
            all_srcview_depth_ests8.append(dp8)
            all_srcview_geomask8.append(geo_mask8)
            # 9
            dp9 = depth_reprojected.copy()
            dp9[~geo_mask9] = 0
            geo_mask_sum9 += geo_mask9.astype(np.int32)
            all_srcview_depth_ests9.append(dp9)
            all_srcview_geomask9.append(geo_mask9)
            # 5
            depth_reprojected[~geo_mask5] = 0
            geo_mask_sum5 += geo_mask5.astype(np.int32)
            all_srcview_depth_ests5.append(depth_reprojected)
            all_srcview_geomask5.append(geo_mask5)
            '''
            print("viewsnum: ",views_num)
            error = np.abs(depth_reprojected - src_depth_est)
            error = error[geo_mask5]
            print("max: ",error.max())
            print("min: ",error.min())
            print("mean: ",error.mean())
            print("<1: ",(error<5).astype(float).mean())
            views_num = views_num +1 
            '''
        depth_est_averaged9 = (sum(all_srcview_depth_ests9) + ref_depth_est) / (geo_mask_sum9 + 1)
        geo_mask9 = geo_mask_sum9 >= 9

        depth_est_averaged8 = (sum(all_srcview_depth_ests8) + ref_depth_est) / (geo_mask_sum8 + 1)
        geo_mask8 = geo_mask_sum8 >= 8

        depth_est_averaged7 = (sum(all_srcview_depth_ests7) + ref_depth_est) / (geo_mask_sum7 + 1)
        geo_mask7 = geo_mask_sum7 >= 7

        depth_est_averaged6 = (sum(all_srcview_depth_ests6) + ref_depth_est) / (geo_mask_sum6 + 1)
        geo_mask6 = geo_mask_sum6 >= 6

        depth_est_averaged5 = (sum(all_srcview_depth_ests5) + ref_depth_est) / (geo_mask_sum5 + 1)
        geo_mask5 = geo_mask_sum5 >= 5

        depth_est_averaged4 = (sum(all_srcview_depth_ests4) + ref_depth_est) / (geo_mask_sum4 + 1)
        geo_mask4 = geo_mask_sum4 >= 4

        depth_est_averaged3 = (sum(all_srcview_depth_ests3) + ref_depth_est) / (geo_mask_sum3 + 1)
        geo_mask3 = geo_mask_sum3 >= 3

        depth_est_averaged2 = (sum(all_srcview_depth_ests2) + ref_depth_est) / (geo_mask_sum2 + 1)
        geo_mask2 = geo_mask_sum2 >= 2

        '''
        geo_mask_r =  geo_mask2.astype(np.float32)
        depth_ave = geo_mask_r * depth_est_averaged2 + (1 - geo_mask_r) * depth_est_averaged3
        mask23 = np.logical_or(geo_mask2, geo_mask3)

        geo_mask_r = mask23.astype(np.float32)
        depth_ave = geo_mask_r * depth_ave + (1 - geo_mask_r) * depth_est_averaged4
        mask34  = np.logical_or(mask23, geo_mask4)

        geo_mask_r = mask34.astype(np.float32)
        depth_ave = geo_mask_r * depth_ave + (1 - geo_mask_r) * depth_est_averaged5
        mask345 = np.logical_or(mask34, geo_mask5)
        '''
        geo_mask_r = geo_mask2.astype(np.float32)
        depth_ave = geo_mask_r * depth_est_averaged2 + (1 - geo_mask_r) * depth_est_averaged3
        mask23 = np.logical_or(geo_mask2, geo_mask3)


        geo_mask_r = mask23.astype(np.float32)
        depth_ave = geo_mask_r * depth_ave + (1 - geo_mask_r) * depth_est_averaged4
        mask234 = np.logical_or(mask23, geo_mask4)


        geo_mask_r = mask987.astype(np.float32)
        depth_ave = geo_mask_r * depth_ave + (1 - geo_mask_r) * depth_est_averaged5
        mask2345 = np.logical_or(mask234, geo_mask5)


        geo_mask_r = mask9876.astype(np.float32)
        depth_ave = geo_mask_r * depth_ave + (1 - geo_mask_r) * depth_est_averaged6
        mask23456 = np.logical_or(mask2345, geo_mask6)

        geo_mask_r = mask98765.astype(np.float32)
        depth_ave = geo_mask_r * depth_ave + (1 - geo_mask_r) * depth_est_averaged7
        mask234567 = np.logical_or(mask23456, geo_mask7)

        geo_mask_r = mask987654.astype(np.float32)
        depth_ave = geo_mask_r * depth_ave + (1 - geo_mask_r) * depth_est_averaged8
        mask2345678 = np.logical_or(mask234567, geo_mask8)

        geo_mask_r = mask543.astype(np.float32)
        depth_ave = geo_mask_r * depth_ave + (1 - geo_mask_r) * depth_est_averaged9
        mask23456789 = np.logical_or(mask2345678, geo_mask9)

        # at least args.thres_view source views matched
        geo_mask = mask23456789
        depth_est_averaged = depth_ave

        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))

        if args.display:
            import cv2
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est / 800)
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
            cv2.waitKey(0)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        # color = ref_img[1:-16:4, 1::4, :][valid_points]  # hardcoded for DTU dataset

        if num_stage == 1:
            color = ref_img[1::4, 1::4, :][valid_points]
        elif num_stage == 2:
            color = ref_img[1::2, 1::2, :][valid_points]
        elif num_stage == 3:
            color = ref_img[valid_points]

        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

        # # set used_mask[ref_view]
        # used_mask[ref_view][...] = True
        # for idx, src_view in enumerate(src_views):
        #     src_mask = np.logical_and(final_mask, all_srcview_geomask[idx])
        #     src_y = all_srcview_y[idx].astype(np.int)
        #     src_x = all_srcview_x[idx].astype(np.int)
        #     used_mask[src_view][src_y[src_mask], src_x[src_mask]] = True

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def pcd_filter_worker(args, scan):
    if args.testlist != "all":
        scan_id = int(scan[4:])
        save_name = 'mvsnet{:0>3}_l3.ply'.format(scan_id)
    else:
        save_name = '{}.ply'.format(scan)
    pair_folder = os.path.join(args.datapath, scan)
    scan_folder = os.path.join(args.outdir, scan)
    out_folder = os.path.join(args.outdir, scan)

    if scan in tank_cfg.scenes:
        scene_cfg = getattr(tank_cfg, scan)
        args.conf = scene_cfg.conf

    filter_depth(args, pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name))


def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pcd_filter(args, testlist, number_worker):
    partial_func = partial(pcd_filter_worker, args)

    p = Pool(number_worker, init_worker)
    try:
        p.map(partial_func, testlist)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()