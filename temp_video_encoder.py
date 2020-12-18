import os

######################################### IMAGE ENCODING(TO MPEG) #########################################
working_state = 'demo'  # full, demo
working_dir = '//Walking_Videos'

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
                os.makedirs(working_dir + '/outputs/%s_demo' % folders[i])
            os.system(
                "ffmpeg -y -i {0} -to 0:20 -c copy {1}.mp4".format(working_dir + '/%s/%s' % (folders[i], videos[n]),
                                                                   working_dir + '/outputs/%s_demo/%s_demo' % (
                                                                       folders[i], videos[n])))
        elif working_state == 'full':
            if not os.path.exists(working_dir + '/outputs/%s_mp4' % folders[i]):
                os.makedirs(working_dir + '/outputs/%s_mp4' % folders[i])
            os.system(
                "ffmpeg -y -i {0} -c copy {1}.mp4".format(working_dir + '/%s/%s' % (folders[i], videos[n]),
                                                          working_dir + '/outputs/%s_mp4/%s' % (folders[i], videos[n])))
