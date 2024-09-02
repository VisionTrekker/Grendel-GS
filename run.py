import os

print('---------------------------------------------------------------------------------')
cmd = f'torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
	--bsz 4 \
	--images images_4 -r 1 \
	--eval --llffhold 83 \
	-s /data2/jtx/data/rubble \
	-m output/rubble_4_2 \
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
#print(cmd)
#os.system(cmd)

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





print('---------------------------------------------------------------------------------')
cmd = f'torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py \
	--bsz 1 \
	--images images \
	-s /data2/liuzhi/Dataset/3DGS_Dataset/siyue \
	-m output/siyue \
	--iterations 200_000 \
	--densify_from_iter 2000 \
	--densify_until_iter 100_000 \
	--densification_interval 400 \
	--densify_grad_threshold 0.0002 \
	--percent_dense 0.01 \
	--opacity_reset_interval 9000 \
	--test_iterations 30000 80000 130000 180000 200000 \
	--save_iterations 80000 130000 200000'
print(cmd)
os.system(cmd)

print('---------------------------------------------------------------------------------')
cmd = f'CUDA_VISIBLE_DEVICES=3 python render.py -m output/matrixcity_ --iteration 119997 --skip_train --eval'
# print(cmd)
# os.system(cmd)

print('---------------------------------------------------------------------------------')
cmd = f'CUDA_VISIBLE_DEVICES=3 python metrics.py -m output/matrixcity_ --mode test'
# print(cmd)
# os.system(cmd)
