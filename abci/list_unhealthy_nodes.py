import argparse
import os
import subprocess
import tempfile


def is_healthy_nccl_tests(hosts: list[str], slot: int, all_reduce_perf: str) -> bool:
    """
    nccl testを host file に従って実行する。
    hostfile に記載されたホスト間での通信が成功するかを判定する

    Args:
        hosts (List[str]): ["g0240", "g0241", ...]
        slot (int): V100 node なら 4, A100 node なら 8
        all_reduce_perf (str): nccl test all reduce perf のパス

    Returns:
        bool: hosts のホスト間での通信が成功するか
    """
    print(f"nccl-test: {hosts}", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        hostfile: str = os.path.join(tmpdir, "hostfile")
        with open(hostfile, "w") as f:
            f.write("\n".join([f"{host} slots={slot}" for host in hosts]))

        # run nccl test
        try:
            subprocess.run(
                args=[
                    "mpiexec",
                    "--bind-to",
                    "none",
                    "-x",
                    "NCCL_DEBUG=INFO",
                    "-x",
                    "NCCL_ASYNC_ERROR_HANDLING=1",
                    "--hostfile",
                    hostfile,
                    all_reduce_perf,
                    "-b",
                    "1k",
                    "-e",
                    "512k",
                    "-d",
                    "half",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(str(e), flush=True)
            return False

    return True


def nccl_binary_search(
    hosts: list[str], slot: int, all_reduce_perf: str, begin: int, end: int
) -> int:
    """_summary_

    Args:
        hosts (list[str]): ホスト名の配列 ["g0240", "g0241", ...]
        slot (int): V100 node なら 4, A100 node なら 8
        all_reduce_perf (str): nccl test all reduce perf のパス
        begin (int): 2分探索の開始位置
        end (int): 2分探索の終了位置

    Returns:
        int: _description_
    """
    # Find maximum t that pass `_run_nccl_tests([hosts[lower:t]])`
    lower: int = begin
    upper: int = end + 1

    while upper - lower > 1:
        mid: int = (lower + upper) // 2
        if mid - begin < 2:
            # 2ノード未満(1ノード)の場合は、healthy かどうか判定できない
            break
        if is_healthy_nccl_tests(
            hosts=hosts[begin:mid], slot=slot, all_reduce_perf=all_reduce_perf
        ):
            lower = mid
        else:
            upper = mid

    return lower


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unhealthy-node-list-dir",
        type=str,
        default="/groups/gaf51217/fujii/abci-unhealthy-node-list",
    )
    parser.add_argument("--all-reduce-perf", type=str, default="all_reduce_perf")
    parser.add_argument("--hostfile", type=str)

    args: argparse.Namespace = parser.parse_args()

    # create directory
    os.makedirs(args.unhealthy_node_list_dir, exist_ok=True)

    with open(args.hostfile) as f:
        hosts: list[str] = [line for line in f.readlines() if line != ""]
    # g0240 slots=4 のような行を想定 (g0240: ホスト名, 4: スロット数)
    slot = int(hosts[0].split(" ")[1].replace("slots=", ""))
    hosts = [line.split(" ")[0] for line in hosts]

    healthy_nodes: list[str] = []
    unhealthy_nodes: list[str] = []
    unknown_nodes: list[str] = []

    # binary search: 2分探索
    begin = 0
    end: int = len(hosts)

    while end - begin > 1:
        t: int = nccl_binary_search(
            hosts=hosts, slot=slot, all_reduce_perf=args.all_reduce_perf, begin=begin, end=end
        )
        if begin == t:
            # all_reduce_perf が (rank=begin, begin+1) でfailedした
            # このとき、begin の番号のノードが healthy かどうかは判定できない
            unknown_nodes.append(hosts[begin])
            begin: int = begin + 1
        else:
            # all_reduce_per が hosts[begin:t] の間で成功した
            for rank in range(begin, t):
                healthy_nodes.append(hosts[rank])
            if t != end:
                unhealthy_nodes.append(hosts[t])
            begin = t + 1

    assert len(healthy_nodes) != 0

    for unknown_node in unknown_nodes:
        if is_healthy_nccl_tests(
            hosts=healthy_nodes + [unknown_node], slot=slot, all_reduce_perf=args.all_reduce_perf
        ):
            healthy_nodes.append(unknown_node)
        else:
            unhealthy_nodes.append(unknown_node)

    print("Result:\n  Healthy node:")
    for node in healthy_nodes:
        print(f"    {node}")
    print("Unhealthy node:")
    for node in unhealthy_nodes:
        print(f"    {node}")

    # update unhealthy node list
    with open(os.path.join(args.unhealthy_node_list_dir, "hostfile"), "w") as f:
        for unhealthy_node in unhealthy_nodes:
            f.write(f"{unhealthy_node}\n")


if __name__ == "__main__":
    main()
