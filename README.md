# 运行环境

两种运行环境及测试环境：requirement_1.txt, requirement_2.txt, requirement_test.txt

# 数据处理

首先根据DataProcess.ipynb产生数据集文件。
cd submit
产生V1_3d_4_J.npz，复制到./Model_inference/Mix_GCN/dataset/save_3d_pose下，和./Model_inference/Mix_Former/dataset/save_3d_pose下
产生V1_2d_4_J.npz，复制到./Model_inference/Mix_GCN/dataset/save_2d_pose下
产生train_label.pkl和test_A_label.pkl, 复制到./Model_inference/Mix_GCN/data/uav/xsub1下 //python=3.8.12 numpy==1.22.3
原数据集train_joint.npy和test_A_joint.npy, 复制到./Model_inference/Mix_GCN/data/uav/xsub1下。
产生V1_test_B.npz 复制到./Model_inference/Mix_GCN/dataset/save_3d_pose下，和./Model_inference/Mix_Former/dataset/save_3d_pose下
产生V1_test_B_2d.npz 复制到./Model_inference/Mix_GCN/dataset/save_2d_pose下

最终结构
data/
```
uav
___ xsub1
    ___ test_A_joint.npy
    ___ test_label.pkl
    ___ train_joint.npy
    ___ train_label.pkl

```

dataset
___ save_3d_pose
    ___ train_joint.npz
    ___ V1_3d_4_J.npz
    ___ V1_test_B.npz
___ save_2d_pose
    ___ V1_2d_4_J


环境需在本地安装torchlight:
cd ./Model_inference/Mix_GCN/
pip install -e torchlight


# 训练

    # 环境1：requirement1.txt (python==3.8.12)

    Mix_GCN:
    cd ./Model_inference/Mix_GCN/

        CTR-GCN-J-3d-env1:
        python main.py --config ./config/ctrgcn_V1_J_3d.yaml --device 0
        CTR-GCN-J-2d-env1:
        python main.py --config ./config/ctrgcn_V1_J.yaml --device 0

        CTR_GCN_B_3d_env1:
        python main.py --config ./config/ctrgcn_V1_B_3d.yaml --device 0

        MST_GCN_B_3d_env1:
        python main.py --config ./config/mstgcn_V1_J_3d.yaml --device 0

        TD_GCN_B_2d_env1:
        python main.py --config ./config/tdgcn_V1_J.yaml --device 0

        CTR_GCN_JM_3d_env1:
        python main.py --config ./config/ctrgcn_V1_JM_3d.yaml --device 0




    Mix_Former:
    cd ./Model_inference/Mix_Former/

        Former_J:
        python main.py --config ./config/mixformer_V1_J.yaml --device 0

        Former_B:
        python main.py --config ./config/mixformer_V1_B.yaml --device 0

    TE-GCN:
    cd ./Model_inference/Mix_GCN/

        python main_tegcn.py --config ./config/tegcn_V1_J_3d_train.yaml --device 0

        

    # 环境2：requirement2.txt (python==3.10.12)

    Mix_GCN:
    cd ./Model_inference/Mix_GCN
        
        CTR_GCN_B_3d_env2:
        python main.py --config ./config/ctrgcn_V1_B_3d_newdata.yaml --device 0

        CTR_GCN_J_3d_env2:
        python main.py --config ./config/ctrgcn_V1_J_3d_newdata_newenv.yaml --device 0




# 测试集Ascore获取：//以下权重为最优结果

    在训练结束后，分别选择最优权重文件进行测试。

    # 环境1：requirement1.txt

    Mix_GCN:
    cd ./Model_inference/Mix_GCN/

        CTR-GCN-J-3d-env1:
        python main.py --config ./config/ctrgcn_V1_J_3d.yaml --device 0 --phase test --save-score True --weights ./output/ctrgcn_V1_J_3D_env1/runs-72-45504.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_CTR_J.pkl
        参考：72.3
        
        CTR-GCN-J-2d-env1:
        python main.py --config ./config/ctrgcn_V1_J.yaml --device 0 --phase test --save-score True --weights ./output/ctrgcn_V1_J/runs-57-14592.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_CTR_J_2d.pkl
        参考：63.5

        CTR_GCN_B_3d_env1:
        python main.py --config ./config/ctrgcn_V1_B_3d.yaml --device 0 --phase test --save-score True --weights ./output/ctrgcn_V1_B_3D_env1/runs-73-46136.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_CTR_B_n.pkl
        71.3

        MST_GCN_J_3d_env1:
        python main.py --config ./config/mstgcn_V1_J_3d.yaml --device 0 --phase test --save-score True --weights ./output/mstgcn_V1_J_3d/runs-87-27492.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_MST_J.pkl
        67.85

        TD_GCN_J_2d_env1:
        python main.py --config ./config/tdgcn_V1_J.yaml --device 0 --phase test --save-score True --weights ./output/tdgcn_V1_J/runs-42-10752.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_TD_J.pkl
        63.80 //

        CTR_GCN_JM_3d_env1:
        python main.py --config ./config/ctrgcn_V1_JM_3d.yaml --device 0 --phase test --save-score True --weights ./output/ctrgcn_V1_JM_3D/runs-35-17955.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_CTR_JM.pkl
        54.8




    Mix_Former:
    cd ./Model_inference/Mix_Former/

        Former_J:
        python main.py --config ./config/mixformer_V1_J.yaml --device 0 --phase test --save-score True --weights ./output/mixformer_V1_J/runs-57-7296.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_former_J.pkl
        70.85

        Former_B:
        python main.py --config ./config/mixformer_V1_B.yaml --device 0 --phase test --save-score True --weights ./output/mixformer_V1_B/runs-63-8064.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_former_B.pkl
        66.25

    TE-GCN:
    cd ./Model_inference/Mix_GCN/

        python main_tegcn.py --config ./config/tegcn_V1_J_3d_test.yaml --device 0 --phase test --save-score True --weights ./output/tegcn_V1_J_3d/runs-49-25650.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_TE.pkl
        70.4

        

    # 环境2：requirement2.txt

    Mix_GCN:
    cd ./Model_inference/Mix_GCN
        
        CTR_GCN_B_3d_env2:
        python main.py --config ./config/ctrgcn_V1_B_3d_env2.yaml --device 0 --phase test --save-score True --weights ./output/ctrgcn_V1_B_3D_env2/runs-75-47400.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_CTR_B.pkl
        71.90

        CTR_GCN_J_3d_env2:
        python main.py --config ./config/ctrgcn_V1_J_3d_env2.yaml --device 0 --phase test --save-score True --weights ./output/ctrgcn_V1_J_3D_env2/runs-72-45504.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_CTR_J_n.pkl
        73.45

    以上产生epoch1_test_score{}存于epoch1_test_score文件夹下


# 测试集Bscore获取：//以下权重为最优结果

    # 环境1：requirement1.txt

    Mix_GCN:
    cd ./Model_inference/Mix_GCN/

        CTR-GCN-J-3d-env1:
        python main.py --config ./config_B/ctrgcn_V1_J_3d.yaml --device 0 --phase test --save-score True --weights ./output/ctrgcn_V1_J_3D_env1/runs-72-45504.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_CTR_J.pkl
        
        CTR-GCN-J-2d-env1:
        python main.py --config ./config_B/ctrgcn_V1_J.yaml --device 0 --phase test --save-score True --weights ./output/ctrgcn_V1_J/runs-57-14592.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_CTR_J_2d.pkl

        CTR_GCN_B_3d_env1:
        python main.py --config ./config_B/ctrgcn_V1_B_3d.yaml --device 0 --phase test --save-score True --weights ./output/ctrgcn_V1_B_3D_env1/runs-73-46136.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_CTR_B_n.pkl

        MST_GCN_J_3d_env1:
        python main.py --config ./config_B/mstgcn_V1_J_3d.yaml --device 0 --phase test --save-score True --weights ./output/mstgcn_V1_J_3d/runs-87-27492.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_MST_J.pkl

        TD_GCN_J_2d_env1:
        python main.py --config ./config_B/tdgcn_V1_J.yaml --device 0 --phase test --save-score True --weights ./output/tdgcn_V1_J/runs-42-10752.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_TD_J.pkl

        CTR_GCN_JM_3d_env1:
        python main.py --config ./config_B/ctrgcn_V1_JM_3d.yaml --device 0 --phase test --save-score True --weights ./output/ctrgcn_V1_JM_3D/runs-35-17955.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_CTR_JM.pkl




    Mix_Former:
    cd ./Model_inference/Mix_Former/

        Former_J:
        python main.py --config ./config_B/mixformer_V1_J.yaml --device 0 --phase test --save-score True --weights ./output/mixformer_V1_J/runs-57-7296.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_former_J.pkl

        Former_B:
        python main.py --config ./config_B/mixformer_V1_B.yaml --device 0 --phase test --save-score True --weights ./output/mixformer_V1_B/runs-63-8064.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_former_B.pkl

    TE-GCN:
    cd ./Model_inference/Mix_GCN/

        python main_tegcn.py --config ./config_B/tegcn_V1_J_3d_test.yaml --device 0 --phase test --save-score True --weights ./output/tegcn_V1_J_3d/runs-49-25650.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_TE.pkl

        

    # 环境2：requirement2.txt

    Mix_GCN:
    cd ./Model_inference/Mix_GCN
        
        CTR_GCN_B_3d_env2:
        python main.py --config ./config_B/ctrgcn_V1_B_3d_env2.yaml --device 0 --phase test --save-score True --weights ./output/ctrgcn_V1_B_3D_env2/runs-75-47400.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_CTR_B.pkl

        CTR_GCN_J_3d_env2:
        python main.py --config ./config_B/ctrgcn_V1_J_3d_env2.yaml --device 0 --phase test --save-score True --weights ./output/ctrgcn_V1_J_3D_env2/runs-72-45504.pt
        产生epoch1_test_score.pkl -> epoch1_test_score_CTR_J_n.pkl

    以上产生epoch1_test_score{}存于epoch1_test_score_B文件夹下

运行Ensemble_test.py, 即产生测试集B上的pred.npy (环境：requirement_test.txt)
