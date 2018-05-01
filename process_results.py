import os

main_dir = './old_png/'
out_result = './old_png/result.txt'

all_files = list(os.listdir(main_dir))
with open(out_result, 'w') as fout:
	for f in all_files:
		if f[-3:] != 'png':
			continue
		f = f[3:-4]
		values = dict()
		all_words = f.split('_')
		for i in range(len(all_words)-1):
			if i % 2 == 0:
				if all_words[i] != 'epoch':
					values[all_words[i]] = float(all_words[i+1])
				else:
					values[all_words[i]] = int(all_words[i+1])
		fout.write('epoch %d loss %.3f best_rank %.2f mean_rank %.2f best_mAP %.4f mAP %.4f \n' \
			% (values['epoch'], values['loss'], values['bestrank'], values['meanrank'], values['bestmAP'], values['mAP']))
