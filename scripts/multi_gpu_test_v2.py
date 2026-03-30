"""Multi-GPU test v2: Uses standard RunPod pytorch image with Jupyter + torch JIT compilation."""
import json
import subprocess
import time
import os
import sys

API_KEY = os.environ.get("RUNPOD_API_KEY", "")

# Test configs — use standard pytorch image (has Jupyter)
GPU_CONFIGS = [
    {"name": "vram-test-a6000", "gpu": "NVIDIA RTX A6000", "arch": "sm_86", "desc": "Ampere (RTX 3090/A6000)"},
    {"name": "vram-test-l40s", "gpu": "NVIDIA L40S", "arch": "sm_89", "desc": "Ada (RTX 4090/L40S)"},
]

BENCHMARK_CODE_TEMPLATE = '''
import torch
import time
import json

gpu_name = torch.cuda.get_device_name(0)
vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
print(f"GPU: {{gpu_name}}")
print(f"VRAM: {{vram}} GB")
print(f"CUDA: {{torch.version.cuda}}")

# Compile kernel via torch JIT
from torch.utils.cpp_extension import load_inline
import subprocess
subprocess.run(["pip", "install", "-q", "ninja"], capture_output=True)

cuda_src = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void dequant_int8_kernel(
    const int8_t* quantized, const float* scales, float* output,
    int n, int block_size) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = (float)quantized[i] * scales[i / block_size];
}}

torch::Tensor dequantize_int8(torch::Tensor quantized, torch::Tensor scales, int block_size) {{
    auto n = quantized.numel();
    auto output = torch::empty({{n}}, torch::TensorOptions().dtype(torch::kFloat32).device(quantized.device()));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_int8_kernel<<<blocks, threads>>>(
        quantized.data_ptr<int8_t>(), scales.data_ptr<float>(),
        output.data_ptr<float>(), n, block_size);
    return output;
}}
"""

cpp_src = "torch::Tensor dequantize_int8(torch::Tensor quantized, torch::Tensor scales, int block_size);"

print("Compiling CUDA kernel via torch JIT...")
t0 = time.time()
mod = load_inline(name="vram_pager_bench", cpp_sources=cpp_src, cuda_sources=cuda_src,
    functions=["dequantize_int8"], extra_cuda_cflags=["-O2"], verbose=False)
compile_time = time.time() - t0
print(f"Compiled in {{compile_time:.1f}}s")

N = 26_000_000
BLOCK = 128

orig = torch.randn(N, device="cuda")
sc = orig.reshape(-1, BLOCK).abs().max(dim=1).values / 127.0
q = (orig.reshape(-1, BLOCK) / sc.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8).reshape(-1)

# Warmup
for _ in range(10):
    _ = mod.dequantize_int8(q, sc, BLOCK)
torch.cuda.synchronize()

# Benchmark kernel
t0 = time.time()
for _ in range(1000):
    out = mod.dequantize_int8(q, sc, BLOCK)
torch.cuda.synchronize()
kms = (time.time()-t0)/1000*1000

# Correctness
pt = (q.float().reshape(-1, BLOCK) * sc.unsqueeze(1)).reshape(-1)
err = (out - pt).abs().max().item()

# PCIe bandwidth
f32_cpu = torch.randn(N, device="cpu").pin_memory()
i8_cpu = torch.randint(-128, 127, (N,), dtype=torch.int8, device="cpu").pin_memory()

torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    _ = f32_cpu.cuda(non_blocking=True)
    torch.cuda.synchronize()
f32ms = (time.time()-t0)/50*1000

sc_cpu = torch.randn(N//BLOCK, device="cpu").pin_memory()
out_gpu = torch.empty(N, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    gi = i8_cpu.cuda(non_blocking=True)
    gs = sc_cpu.cuda(non_blocking=True)
    torch.cuda.synchronize()
    _ = mod.dequantize_int8(gi, gs, BLOCK)
    torch.cuda.synchronize()
    del gi, gs
i8ms = (time.time()-t0)/50*1000

results = {{
    "gpu": gpu_name, "vram_gb": vram, "cuda": torch.version.cuda,
    "arch": "{ARCH}",
    "compile_s": round(compile_time, 1),
    "kernel_ms": round(kms, 3), "error": err, "correct": err < 1e-5,
    "fp32_transfer_ms": round(f32ms, 1), "int8_total_ms": round(i8ms, 1),
    "speedup": round(f32ms / i8ms, 2),
    "pcie_bandwidth_gbs": round(N*4/1e6/f32ms*1000, 0)
}}
print(f"\\nRESULTS_JSON:{{json.dumps(results)}}")
'''


def graphql(query):
    result = subprocess.run([
        "curl", "-s", "-H", "Content-Type: application/json",
        "-d", json.dumps({"query": query}),
        f"https://api.runpod.io/graphql?api_key={API_KEY}"
    ], capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def create_pod(name, gpu_type):
    query = ('mutation { podFindAndDeployOnDemand(input: { '
             f'name: "{name}", gpuTypeId: "{gpu_type}", gpuCount: 1, '
             'volumeInGb: 10, containerDiskInGb: 20, '
             'imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", '
             'volumeMountPath: "/workspace", '
             'minVcpuCount: 4, minMemoryInGb: 16, ports: "8888/http" '
             '}) { id name machine { gpuDisplayName } } }')
    result = graphql(query)
    if "errors" in result:
        return None, result["errors"][0]["message"]
    pod = result["data"]["podFindAndDeployOnDemand"]
    return pod["id"], pod["machine"]["gpuDisplayName"]


def wait_for_pod(pod_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        query = '{ pod(input: {podId: "' + pod_id + '"}) { runtime { uptimeInSeconds } } }'
        result = graphql(query)
        rt = result["data"]["pod"].get("runtime")
        if rt and rt.get("uptimeInSeconds", 0) > 0:
            return True
        time.sleep(15)
    return False


def destroy_pod(pod_id):
    graphql('mutation { podTerminate(input: {podId: "' + pod_id + '"}) }')


def run_on_pod(pod_id, code, timeout=300):
    """Execute via Jupyter WebSocket."""
    import requests
    import websocket

    jupyter_url = f"https://{pod_id}-8888.proxy.runpod.net"
    token = "dev"

    # Wait for Jupyter to come up
    for attempt in range(30):
        try:
            r = requests.get(f"{jupyter_url}/api?token={token}", timeout=5)
            if r.status_code == 200:
                break
        except:
            pass
        time.sleep(10)
    else:
        print("  Jupyter never started")
        return ""

    # Get/create kernel
    session = requests.Session()
    resp = session.post(f"{jupyter_url}/api/kernels?token={token}",
                       headers={"Content-Type": "application/json"})
    kid = resp.json()["id"]

    ws_url = jupyter_url.replace("https://", "wss://") + f"/api/kernels/{kid}/channels?token={token}"
    ws = websocket.create_connection(ws_url, timeout=30)

    mid = f"run_{time.time()}"
    msg = {
        "header": {"msg_id": mid, "msg_type": "execute_request", "username": "", "session": "", "date": "", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": True, "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        "buffers": [], "channel": "shell"
    }
    ws.send(json.dumps(msg))

    output = ""
    start = time.time()
    while time.time() - start < timeout:
        try:
            ws.settimeout(15)
            r = json.loads(ws.recv())
            if r.get("parent_header", {}).get("msg_id") != mid:
                continue
            if r.get("msg_type") == "stream":
                output += r["content"]["text"]
                print(r["content"]["text"], end="", flush=True)
            elif r.get("msg_type") == "error":
                output += f"ERROR: {r['content'].get('evalue', '')}\n"
                print(f"ERROR: {r['content'].get('evalue', '')}")
            elif r.get("msg_type") == "execute_reply":
                break
        except websocket.WebSocketTimeoutException:
            continue
    ws.close()
    return output


def main():
    all_results = []

    for config in GPU_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Testing: {config['desc']} ({config['gpu']})")
        print(f"{'='*60}")

        pod_id = None
        try:
            # Create pod
            print(f"  Creating pod...")
            pod_id, gpu_name = create_pod(config["name"], config["gpu"])
            if pod_id is None:
                print(f"  FAILED: {gpu_name}")
                continue
            print(f"  Pod: {pod_id} ({gpu_name})")

            # Wait
            print(f"  Waiting for pod...")
            if not wait_for_pod(pod_id, timeout=300):
                print(f"  TIMEOUT")
                continue

            # Run benchmark
            print(f"  Running benchmark...")
            code = BENCHMARK_CODE_TEMPLATE.replace("{ARCH}", config["arch"])
            output = run_on_pod(pod_id, code, timeout=300)

            # Extract results
            if "RESULTS_JSON:" in output:
                json_str = output.split("RESULTS_JSON:")[1].strip().split("\n")[0]
                results = json.loads(json_str)
                all_results.append(results)
                print(f"\n  PASSED: {results['gpu']} — {results['speedup']}x speedup")
            else:
                print(f"\n  NO RESULTS extracted")

        except Exception as e:
            print(f"  EXCEPTION: {e}")

        finally:
            if pod_id:
                print(f"  Destroying pod {pod_id}...")
                destroy_pod(pod_id)
                print(f"  Pod destroyed.")

    # Summary
    print(f"\n\n{'='*60}")
    print(f"MULTI-GPU BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"\n{r['gpu']} ({r['arch']}, {r['vram_gb']}GB):")
        print(f"  Kernel: {r['kernel_ms']:.3f}ms | Correct: {r['correct']} | Error: {r['error']:.2e}")
        print(f"  FP32 transfer: {r['fp32_transfer_ms']}ms | INT8+decompress: {r['int8_total_ms']}ms")
        print(f"  Net speedup: {r['speedup']}x | PCIe: {r['pcie_bandwidth_gbs']} GB/s")

    # Save
    os.makedirs("os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research")", exist_ok=True)
    with open("os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research")/multi_gpu_benchmarks.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to research/multi_gpu_benchmarks.json")


if __name__ == "__main__":
    main()
