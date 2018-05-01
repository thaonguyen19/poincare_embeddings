import os

main_dir = './old_png/'
out_result = './old_png/result.txt'

for file in os.listdir(main_dir):
	file = file[3:-4]
	values = dict()
	all_words = file.split('_')
	for i in range(len(all_words)):
		if i % 2 == 0:
			if all_words[i] != 'epoch':
				values[all_words[i]] = float(all_words[i+1])
			else:
				values[all_words[i]] = int(all_words[i+1])

		with open(out_result, 'a') as fout:
			fout.write('epoch %d loss %.3f best_rank %.2f mean_rank %.2f best_mAP %.4f mAP %.4f' \
					% (values['epoch'], values['loss'], values['bestrank'], values['meanrank'], values['bestmAP'], values['mAP']))
