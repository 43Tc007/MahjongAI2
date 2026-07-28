from train import train_PPO
import time
import argparse

def benchmark_num_workers(worker_counts=None, n_iters=3, save_checkpoint=False, save_final=False):
    if worker_counts is None:
        worker_counts = [1, 2, 4]

    results = []
    for num_workers in worker_counts:
        print(f"\nBenchmarking num_workers={num_workers}...")
        start = time.perf_counter()
        train_PPO(
            policy_path='',
            critic_path='',
            n_iters=n_iters,
            auto_shutdown=False,
            save_checkpoint=save_checkpoint,
            save_final=save_final,
            num_workers=num_workers,
        )
        elapsed = time.perf_counter() - start
        results.append((num_workers, elapsed))
        print(f"num_workers={num_workers}: {elapsed:.2f} seconds")

    print("\nBenchmark summary:")
    for num_workers, elapsed in results:
        print(f"- num_workers={num_workers}: {elapsed:.2f} seconds")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, default='', help="Policy path")
    parser.add_argument("--critic_path", type=str, default='', help="Critic path")
    parser.add_argument("--n_iters", type=int, default=3, help="Training iterations for each benchmark run")
    parser.add_argument("--auto_shutdown", type=int, default=0, help="Shutdown after training")
    parser.add_argument("--save_checkpoint", type=int, default=0, help="save_checkpoint")
    parser.add_argument("--save_final", type=int, default=0, help="save_final")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of collector workers for a single run")
    parser.add_argument(
        "--benchmark_workers",
        nargs='*',
        type=int,
        default=None,
        help="Run a benchmark over these worker counts (for example: 1 2 4 8)",
    )
    args = parser.parse_args()
    benchmark_num_workers(
        worker_counts=[1, 2, 4, 8, 12],
        n_iters=args.n_iters,
        save_checkpoint=False,
        save_final=False,
    )
