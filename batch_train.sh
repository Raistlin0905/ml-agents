# Configurable parameters
NUM_RUNS=10  # Number of training runs
COOLING_SECONDS=300  # 5 minutes cooling between runs; increase to 600+ if needed

cd "$(dirname "$0")"  # Ensure in ML-Agents dir

# Sleep prevention
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS: Use caffeinate to prevent idle sleep for the entire script
    caffeinate -i $$ &
    CAFF_PID=$!
    echo "macOS sleep prevented (PID: $CAFF_PID)"
fi

for ((i=1; i<=NUM_RUNS; i++)); do
    echo "Starting run $i/$NUM_RUNS"
    start_time=$(date +%s)
    
    # Run training; waits for completion (handles your CSV data storage automatically)
    python3 data_collection/headless_train.py 3DBall --no-graphics
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Run $i completed in ${duration}s; CSV data appended."
    
    if [ $i -lt $NUM_RUNS ]; then
        echo "Cooling for ${COOLING_SECONDS}s"
        sleep $COOLING_SECONDS
    fi
done

echo "All $NUM_RUNS runs complete. Check training_data.csv for aggregated data."

# Cleanup
if [[ "$(uname)" == "Darwin" ]]; then
    kill $CAFF_PID 2>/dev/null
    echo "macOS caffeinate stopped"
fi