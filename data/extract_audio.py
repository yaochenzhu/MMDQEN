import os
import glob

from multiprocessing import Pool

def get_video_list(root_dir="video", ext=".mp4"):
	video_list = glob.glob(os.path.join(root_dir, "*"+ext))
	return video_list


def change_root_ext(ori_path_list, ori_root="video", new_root="audio", new_ext=".wav"):
	new_path_list = [os.path.join(new_root, os.path.split(ori_path)[-1]) for ori_path in ori_path_list]
	new_path_list = [new_path.split(".")[0] + new_ext for new_path in new_path_list]
	return new_path_list


def extract_wav_thread(video_list, audio_list):
	ffmpeg_fmt = "ffmpeg -i {} -f wav -ar 16000 {}"	
	[os.system(ffmpeg_fmt.format(video_path, audio_path)) for video_path, audio_path in zip(video_list, audio_list)]


def extract_wav(video_root, audio_root, n_jobs=8):
	if not os.path.exists(audio_root):
		os.mkdir(audio_root)
	assert(os.path.exists(video_root)), "{} not exists!".format(video_root)

	video_list = get_video_list(video_root)
	audio_list = change_root_ext(video_list, video_root, audio_root)

	p = Pool(n_jobs)
	[p.apply_async(extract_wav_thread, (video_list[i::n_jobs], audio_list[i::n_jobs])) for i in range(n_jobs)]
	p.close()
	p.join()


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--video_root', type=str, default="movie_clips")
	parser.add_argument('--audio_root', type=str, default="audio")
	args = parser.parse_args()
	extract_wav(args.video_root, args.audio_root)


if __name__ == "__main__":
	main()