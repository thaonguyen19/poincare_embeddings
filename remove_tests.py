# with open('./package_depth_closure_wo_clique/functions_04182018_val', 'r') as fin:
# 	with open('./package_depth_closure_wo_clique/functions_04182018_notests_val', 'w') as fout:
# 		for line in fin:
# 			if 'tests.' not in line:
# 				fout.write(line)
				
import random
with open('./package_depth_closure_wo_clique/functions_04182018_notests_sorted', 'r') as fin:
	with open('./package_depth_closure_wo_clique/functions_04182018_new_val', 'w') as fout:
		all_lines = fin.readlines()
		val_lines = random.sample(all_lines, 983)
		for line in val_lines:
			fout.write(line)