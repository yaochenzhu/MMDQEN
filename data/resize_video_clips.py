import os
import sys
import glob
from multiprocessing import Pool

def get_filepath_list(root, ext=".mp4"):
	assert(os.path.exists(root)), "{} not exists!".format(root)
	filepath_list = glob.glob(os.path.join(root, "*"+ext))
	return filepath_list


def change_root(filepath_list, new_root):
	if not os.path.exists(new_root):
		os.mkdir(new_root)
	filename_list      =  [os.path.split(filepath)[-1] for filepath in filepath_list]
	new_filepath_list  =  [os.path.join(new_root, filename) for filename in filename_list]
	return new_filepath_list


def resize_and_save(ori_filepath_list, new_filepath_list, target_size):
	ffmpeg_cmd_fmt = "ffmpeg -i {} -vf scale="+str(target_size[0])+":"+str(target_size[1])+" {}"
	for ori_filepath, new_filepath in zip(ori_filepath_list, new_filepath_list):
		ffmpeg_cmd = ffmpeg_cmd_fmt.format(ori_filepath, new_filepath)
		os.system(ffmpeg_cmd)


def resize_and_save_multi(ori_root, new_root, target_size, n_jobs):
	ori_filepath_list = get_filepath_list(ori_root)
	new_filepath_list = change_root(ori_filepath_list, new_root)
	p = Pool(n_jobs)
	[p.apply_async(
		resize_and_save,
		(ori_filepath_list[i::n_jobs],
		 new_filepath_list[i::n_jobs],
		 target_size,)
	) for i in range(n_jobs)]
	p.close()
	p.join()


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--ori_root', type=str, default="movie_clips")
	parser.add_argument('--new_root', type=str, default="256_192")
	parser.add_argument('--width' ,   type=int, default=256)
	parser.add_argument('--height',   type=int, default=192)
	parser.add_argument('--n_jobs',   type=int, default=8)
	args = parser.parse_args()
	
	resize_and_save_multi(args.ori_root, args.new_root, (args.width, args.height), args.n_jobs)


if __name__ == "__main__":
	main()