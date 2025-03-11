import os

print('---------------------------------------------------------------------------------')
cmd = f'torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
	--bsz 4 \
	--images images_4 \
	--eval --llffhold 83 \
	-s /data2/jtx/data/rubble \
	-m output/rubble_4_2_test \
	--iterations 200000 \
	--densify_from_iter 1000 \
	--densify_until_iter 50000 \
	--densification_interval 200 \
	--densify_grad_threshold 0.00013 \
	--percent_dense 0.003 \
	--opacity_reset_interval 9000 \
	--test_iterations 7000 30000 50000 80000 110000 120000 140000 160000 180000 200000 \
	--save_iterations 30000 50000 120000 200000 \
	--checkpoint_iterations 50000 120000 200000'
# print(cmd)
# os.system(cmd)

print('---------------------------------------------------------------------------------')
cmd = f'python render.py -m output/rubble_4_2 --iteration 119997 --skip_train --llffhold 83'
#print(cmd)
#os.system(cmd)

print('---------------------------------------------------------------------------------')
cmd = f'python metrics.py -m output/rubble_4_2 --mode test'
#print(cmd)
#os.system(cmd)


# for idx, scene in enumerate(['DJI_qixingpark_crossroad', 'DJI_village_building0_1_sq0', 'DJI_village_building1_1_sq0']):
#     print('---------------------------------------------------------------------------------')
#     cmd = f'torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --bsz 4 -s /data2/liuzhi/Dataset/3DGS_Dataset/input/{scene}/train -m output/{scene}'
#     print(cmd)
#     os.system(cmd)
#
#     print('---------------------------------------------------------------------------------')
#     cmd = f'python render.py -m output/{scene}'
#     print(cmd)
#     os.system(cmd)
#
#     print('---------------------------------------------------------------------------------')
#     cmd = f'python metrics.py -m output/{scene}'
#     print(cmd)
#     os.system(cmd)




# print('---------------------------------------------------------------------------------')
# cmd = f'CUDA_VISIBLE_DEVICES=3 python train.py \
#     --eval \
# 	-s /data2/liuzhi/Dataset/3DGS_Dataset/input/matrixcity/small_city/aerial/pose/block_all \
# 	-m output/matrixcity_all_3dgs \
# 	--iterations 150000 \
# 	--densify_from_iter 2500 \
# 	--densify_until_iter 75000 \
# 	--densification_interval 500 \
# 	--test_iterations 30_000 90_000 150_000 \
# 	--save_iterations 150_000 \
#     --checkpoint_iterations 150_000'
# print(cmd)
# os.system(cmd)
#
# print('---------------------------------------------------------------------------------')
# cmd = f'CUDA_VISIBLE_DEVICES=3 python render.py -m output/matrixcity_all_3dgs --iteration 150_000 --skip_train --eval'
# print(cmd)
# os.system(cmd)
#
# print('---------------------------------------------------------------------------------')
# cmd = f'CUDA_VISIBLE_DEVICES=3 python metrics.py -m output/matrixcity_all_3dgs --mode test'
# print(cmd)
# os.system(cmd)


print('---------------------------------------------------------------------------------')
cmd = f'CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
	--bsz 4 \
	--eval \
	-s /data2/liuzhi/Dataset/3DGS_Dataset/matrixcity/small_city/aerial/pose/block_all \
	-m output/matrixcity_all \
	--iterations 200000 \
	--densify_from_iter 1000 \
	--densify_until_iter 50000 \
	--densification_interval 200 \
	--densify_grad_threshold 0.00013 \
	--percent_dense 0.003 \
	--opacity_reset_interval 9000 \
	--test_iterations 7000 30000 50000 80000 110000 120000 140000 160000 180000 200000 \
	--save_iterations 50000 120000 200000 \
    --start_checkpoint "/data2/liuzhi/3DGS_code/Grendel-GS/output/matrixcity/checkpoints/49997"'
# print(cmd)
# os.system(cmd)

print('---------------------------------------------------------------------------------')
cmd = f'CUDA_VISIBLE_DEVICES=3 python render.py -m output/matrixcity_ --iteration 119997 --skip_train --eval'
# print(cmd)
# os.system(cmd)

print('---------------------------------------------------------------------------------')
cmd = f'CUDA_VISIBLE_DEVICES=3 python metrics.py -m output/matrixcity_ --mode test'
# print(cmd)
# os.system(cmd)




# --densify_grad_threshold: 0.0002
# --percent_dense: 0.01
# /data2/liuzhi/Dataset/3DGS_Dataset/siyue
print('---------------------------------------------------------------------------------')
cmd = f'CUDA_VISIBLE_DEVICES=2 python train.py \
	--images images \
	-s /data2/liuzhi/remote_data/dataset_reality/siyue/0909_airfull \
	-m output/0909_airfull \
	--iterations 60_000 \
	--densify_from_iter 1000 \
	--densify_until_iter 30000 \
	--densification_interval 200 \
	--densify_grad_threshold 0.0002 \
	--percent_dense 0.01 \
	--opacity_reset_interval 6000 \
	--test_iterations 7000 15000 30000 60000 \
	--save_iterations 15000 30000 60000'
print(cmd)
os.system(cmd)

# print('---------------------------------------------------------------------------------')
# cmd = f'CUDA_VISIBLE_DEVICES=3 python render.py -m output/siyue_newsparse_0.00005_0.001 --iteration 299993 --skip_train --skip_test'
# print(cmd)
# os.system(cmd)

print('---------------------------------------------------------------------------------')
# cmd = f'CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py \
# 	--bsz 4 \
# 	--images images \
# 	-s /data2/liuzhi/remote_data/dataset_reality/hnsf_1517 \
# 	-m output/henanshifan \
# 	--iterations 30_000 \
# 	--densify_from_iter 500 \
# 	--densify_until_iter 150_00 \
# 	--densification_interval 100 \
# 	--densify_grad_threshold 0.0002 \
# 	--percent_dense 0.01 \
# 	--opacity_reset_interval 3000 \
# 	--test_iterations 7000 15000 30000\
# 	--save_iterations 7000 30000'
# print(cmd)
# os.system(cmd)
# cmd = f'CUDA_VISIBLE_DEVICES=2 python train.py \
# 	--images images \
# 	-s /data2/liuzhi/remote_data/dataset_reality/hnsf_1517 \
# 	-m output/henanshifan \
# 	--iterations 30_000 \
# 	--densify_from_iter 500 \
# 	--densify_until_iter 150_00 \
# 	--densification_interval 100 \
# 	--densify_grad_threshold 0.0002 \
# 	--percent_dense 0.01 \
# 	--opacity_reset_interval 3000 \
# 	--test_iterations 7000 15000 30000\
# 	--save_iterations 15000 30000'
# print(cmd)
# os.system(cmd)

# print('---------------------------------------------------------------------------------')
# cmd = f'CUDA_VISIBLE_DEVICES=3 python render.py -m output/siyue_fukan_0904 --iteration 29997 --skip_train --skip_test'
# print(cmd)
# os.system(cmd)
