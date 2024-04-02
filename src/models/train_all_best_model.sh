study_name="skin_lesion_classification_with_HAM10000_dataset"
storage_name="$PROJECT_ROOT/experiments/${study_name}.db"
output_dir="$PROJECT_ROOT/models/best_models"
fit_script="$PROJECT_ROOT/src/models/fit_best_models.py"

base_architectures=("VGG16" "VGG19" "ResNet101V2" "InceptionResNetV2" "Xception" "MobileNetV2")

for arch in "${base_architectures[@]}"; do
    python ${fit_script} --study-name "$study_name" --storage-name "$storage_name" --base-architecture "$arch" --output-dir "$output_dir"
done