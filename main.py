import copy
import random
from mlp_pso import mlp_pso
from read_file import build_samples_from_txt
from fold import make_kfold,flatten_one_level
from normalizer import normalizer

# กำหนด input ที่ต้องการ
layers = [4]
inertia_weight = 0.2

random.seed(1)

path = "AirQualityUCI.txt"
feature_attribute = [3, 6, 8, 10, 11, 12, 13, 14]
desire_output_attribute = 5
horizons = (240,)

sample_list = build_samples_from_txt(
    path=path, 
    feature_indices=feature_attribute,
    target_index=desire_output_attribute,
    horizons = horizons,
    drop_neg200=True
)

# 10% cross validation
random.shuffle(sample_list)
folds = make_kfold(sample_list,10)

for i in range(len(folds)):
    train_sample = copy.deepcopy(folds)
    
    # แบ่ง train/validation sample
    val_sample = train_sample.pop(i)
    train_sample = flatten_one_level(train_sample)

    # normalize ข้อมูล
    new_normalizer = normalizer()
    train_sample_norm = new_normalizer.normalize_sample(train_sample)

    # train
    mlp = mlp_pso(len(feature_attribute),layers,1,20)
    mlp.l_best_algorithm(train_sample_norm,1.5,1.5,inertia_weight,100)

    # predict
    mae_list = []
    for j in range(len(val_sample)):
        val_input, val_output = val_sample[j]
        val_input_norm = new_normalizer.normalize_validation_input(val_input)

        g_best_weight, g_best_bias = mlp.g_best_particle # type: ignore
        pred = mlp.feed_forward(val_input_norm,g_best_weight,g_best_bias)

        val_pred = new_normalizer.denormalize_output(pred)

        mae = sum(abs(p - t) for p, t in zip(val_pred, val_output)) / len(val_output)
        mae_list.append(mae)

    print("Fold ", i+1, ":")
    print(f"AVG MAE:  {sum(mae_list)/len(mae_list):.4f}")
    print("---------------------------------------------")