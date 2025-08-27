import submitit
import torch


# Create function that checks how many GPUs are available (if any)
def check_gpu():
    is_gpu = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)

    # Print results
    print(f"Is GPU available: {is_gpu}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"GPU name: {gpu_name}")

    return is_gpu, num_gpus, gpu_name


if __name__ == "__main__":
    # Create executor
    run_name = "check_gpu"
    log_folder = f"logs/{run_name}_%j"

    # TODO: Need to change arguments to be specific with
    # QUEST cluster specifics
    executor = submitit.AutoExecutor(folder=log_folder)

    # Update parameters
    executor.update_parameters(
        slurm_account="p31796",
        name=run_name,
        timeout_min=10,
        slurm_partition="gengpu",
        gpus_per_node=1,
        nodes=1,
        slurm_mem="32G",
    )

    # Submit job
    job = executor.submit(check_gpu)
