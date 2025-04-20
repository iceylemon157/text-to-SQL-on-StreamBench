bench_names=(
    "sql_generation_public"
    "sql_generation_private"
)
model_name="meta-llama/Llama-3.1-8B-Instruct"
device="auto"

echo -e "\033[0;36mRunning Lora StreamICL for SQL Generation:\033[0m"
for bench_name in ${bench_names[@]}; do
    echo -e "\033[0;32mRunning benchmark: $bench_name\033[0m"
    python lora_streamicl-sql.py --bench_name $bench_name --model_name $model_name
done