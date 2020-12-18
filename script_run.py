import os

######################################### IMAGE ENCODING(TO MPEG) #########################################
working_state = 'demo'  # full, demo
working_dir = '/media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/Walking_Videos'

folder_list = os.listdir(working_dir)
folders = sorted(folder_list)

try:
    folder_list.remove('outputs')
except:
    pass

videos_list = list(range(len(folders)))

for i, objects in enumerate(folders):
    video_list = os.listdir(working_dir + '/%s' % objects)
    videos_list[i] = sorted(video_list)  # videos in first object(folder) => videos[0] and so on ..

for i, videos in enumerate(videos_list):
    for n in range(len(videos)):
        if working_state == 'demo':
            if not os.path.exists(working_dir + '/outputs/%s_demo' % folders[i]):
                os.makedirs(working_dir + '/outputs/%s_demo' %folders[i])
                os.system( "ffmpeg -y -i {0} -to 0:20 -c copy {1}.mp4".format(working_dir + '/%s/%s' % (folders[i], videos[n]), working_dir + '/outputs/%s_demo/%s_demo' % ( folders[i], videos[n])))
        elif working_state == 'full':
            if not os.path.exists(working_dir + '/outputs/%s_mp4' % folders[i]): os.makedirs(working_dir + '/outputs/%s_mp4' % folders[i])
            os.system( "ffmpeg -y -i {0} -c copy {1}.mp4".format(working_dir + '/%s/%s' % (folders[i], videos[n]), working_dir + '/outputs/%s_mp4/%s' % (folders[i], videos[n])))


######################################## SKELETON INFERENCE ###############################################
# os.chdir('/media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/inference')
# os.system("python infer_video_d2.py \
#     --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
#     --output-dir /media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/output_directory \
#     --image-ext mp4 \
#     /media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/input_directory")
#
# os.chdir('/media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/data') os.system( "python
# /media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/data/prepare_data_2d_custom.py -i
# /media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/output_directory -o myvideos") os.chdir(
# '/media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D')

os.chdir('/media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/inference')
os.system("python infer_video_d2.py \
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
    --output-dir /media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/output_directory \
    --image-ext mp4 \
    /media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/input_directory/demos")

os.chdir('/media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/data')
os.system(
    "python /media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/data/prepare_data_2d_custom.py -i "
    "/media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/output_directory -o myvideos_demo")
os.chdir('//VideoPose3D')

input_list = os.listdir('/media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/input_directory')
input_demo_list = os.listdir('/media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/input_directory/demos')
inputs = sorted(input_list)
inputs_demo = sorted(input_demo_list)

##################################### EXPORT 3D JOINT VIDEO OR NUMPY ARRAY ###################################
try:
    inputs.remove('demos')
except:
    pass

for i, n in enumerate(inputs):
    os.system("python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate "
              "pretrained_h36m_detectron_coco.bin --render --viz-subject {0} --viz-action custom --viz-camera 0 "
              "--viz-video /media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/input_directory/{0} "
              "--viz-export {0} --viz-size 6".format(n))
# --viz-export : making joint information to numpy array. almost immediate

for i, n in enumerate(inputs_demo):
    os.system("python run.py -d custom -k myvideos_demo -arc 3,3,3,3,3 -c checkpoint --evaluate "
              "pretrained_h36m_detectron_coco.bin --render --viz-subject {0} --viz-action custom --viz-camera 0 "
              "--viz-video /media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/input_directory/demos/{0} "
              "--viz-output /media/kit/f5021b31-011b-4cc0-8e8b-fa110a0f30ea/kist4/VideoPose3D/{0} --viz-size 6".format(n))
# --viz-output : making demo video. it takes time almost same as in inference step for full video length and please
# check video size because this process need massive memory this can kill the process
